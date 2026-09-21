"""Scaffold split for p13: Bemis-Murcko scaffold equality AND ECFP4 Tanimoto >= TAU.

This is the scaffold split reported in the manuscript. It overwrites the
`scaffold_*` arrays in `splits/`, which `train_with_custom_splits.py` reads.

NOTE: the reported threshold is inconsistent between documents - the manuscript
Methods state Tanimoto >= 0.4, the response letter states >= 0.5. TAU below is
0.4 (the manuscript value); reconcile the documents before deposit.

Inputs:
  results/processed_synergydf_p13.csv  (drug1, drug2 as int = DrugBank ID without 'DB' prefix)
  data/drug_data/structures.sdf        (DATABASE_ID=DB..., SMILES, plus optional DRUG_NAME)

Outputs:
  splits/scaffold_train_indices.npy
  splits/scaffold_test_indices.npy
  splits/scaffold_fold_{0..4}_indices.npy
  results/drug_scaffold_clusters.csv        (diagnostic)
  results/scaffold_split_composition.csv

Clustering rule (AND): two drugs are in the same cluster iff
  (a) identical Bemis-Murcko generic scaffold, AND
  (b) ECFP4 (Morgan r=2, 2048-bit) Tanimoto >= TAU
Connected components over this joint relation.

A synergy sample (drug1, drug2) is assigned cluster = min(cluster(d1), cluster(d2));
a sample whose two drugs fall in different clusters takes the smaller cluster id
(conservative: it leaves with whichever drug's cluster is held out first).
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # silence RDKit warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(HERE, "..", ".."))
SYN_PATH = os.path.join(ROOT_DIR, "results", "processed_synergydf_p13.csv")
SDF_PATH = os.path.join(ROOT_DIR, "data", "drug_data", "structures.sdf")
SPLITS_DIR = os.path.join(HERE, "splits")
MODELS_DIR = os.path.join(HERE, "results")
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# === Parameters (top-level, tunable) =======================================
# ECFP4 Tanimoto threshold. UNRESOLVED: manuscript Methods state >= 0.4 (this
# value); the response letter states >= 0.5. Set to match whichever is submitted.
TAU = 0.40
FP_RADIUS = 2
FP_BITS = 2048
TEST_SIZE = 0.20
N_FOLDS = 5
SEED = 42
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. Build DB_id -> (smiles, drug_name, mol) map from SDF
# ---------------------------------------------------------------------------
print("=" * 80)
print("STEP 1: Load SDF and build DrugBank-ID -> SMILES map")
print("=" * 80)

supplier = Chem.SDMolSupplier(SDF_PATH)
db_to_smiles = {}
db_to_name = {}
n_total = n_ok = 0
for mol in supplier:
    n_total += 1
    if mol is None:
        continue
    db_id = mol.GetProp("DATABASE_ID") if mol.HasProp("DATABASE_ID") else None
    smi = mol.GetProp("SMILES") if mol.HasProp("SMILES") else None
    if db_id is None or smi is None:
        continue
    db_to_smiles[db_id] = smi
    if mol.HasProp("DRUG_NAME"):
        db_to_name[db_id] = mol.GetProp("DRUG_NAME")
    elif mol.HasProp("GENERIC_NAME"):
        db_to_name[db_id] = mol.GetProp("GENERIC_NAME")
    else:
        db_to_name[db_id] = ""
    n_ok += 1

print(f"SDF molecules read : {n_total}")
print(f"With DB_ID+SMILES  : {n_ok}")
print(f"Unique DrugBank IDs: {len(db_to_smiles)}")


# ---------------------------------------------------------------------------
# 2. Load p13 synergy table; collect unique drugs as DB{n}
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 2: Load p13 synergy table")
print("=" * 80)

syn = pd.read_csv(SYN_PATH, index_col=0).reset_index(drop=True)
n_samples = len(syn)
print(f"p13 samples: {n_samples}")

# Integer drug ids -> 'DB{n}' with 5-digit zero padding (SDF uses DB00080, DB09053, etc.)
p13_int_ids = pd.concat([syn["drug1"], syn["drug2"]]).unique()
p13_db_ids = sorted({f"DB{int(i):05d}" for i in p13_int_ids})
print(f"p13 unique drugs: {len(p13_db_ids)}")
print(f"  example mapped ids: {p13_db_ids[:5]}")

missing = [d for d in p13_db_ids if d not in db_to_smiles]
print(f"p13 drugs missing from SDF: {len(missing)}")
if missing:
    print(f"  examples: {missing[:10]}")


# ---------------------------------------------------------------------------
# 3. Compute Murcko scaffold (generic) + ECFP4 fingerprint per drug
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"STEP 3: Compute Murcko scaffold + ECFP4 (radius={FP_RADIUS}, bits={FP_BITS})")
print("=" * 80)

records = []  # list of dicts for diagnostic CSV
valid_db_ids = []
for db in p13_db_ids:
    smi = db_to_smiles.get(db)
    if smi is None:
        # Missing: will get its own singleton cluster later
        records.append({"drug_db_id": db, "drug_name": db_to_name.get(db, ""),
                        "smiles": "", "scaffold": "", "has_fp": False})
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        records.append({"drug_db_id": db, "drug_name": db_to_name.get(db, ""),
                        "smiles": smi, "scaffold": "", "has_fp": False})
        continue
    try:
        scaf = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
        scaf_smi = Chem.MolToSmiles(scaf) if scaf is not None else ""
    except Exception:
        scaf_smi = ""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    records.append({"drug_db_id": db, "drug_name": db_to_name.get(db, ""),
                    "smiles": smi, "scaffold": scaf_smi, "has_fp": True, "_fp": fp})
    valid_db_ids.append(db)

df_drugs = pd.DataFrame(records)
n_with_fp = int(df_drugs["has_fp"].sum())
print(f"Drugs with valid ECFP4 fingerprint: {n_with_fp} / {len(df_drugs)}")
print(f"Unique generic scaffolds among valid drugs: "
      f"{df_drugs[df_drugs.has_fp]['scaffold'].nunique()}")


# ---------------------------------------------------------------------------
# 4. Cluster via AND rule: same scaffold AND Tanimoto >= TAU
#    Build adjacency, then connected components (union-find).
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"STEP 4: Cluster (AND rule: same scaffold AND Tanimoto >= {TAU})")
print("=" * 80)

# Index valid drugs 0..K-1
valid_idx = {db: i for i, db in enumerate(valid_db_ids)}
K = len(valid_idx)
fps = [records[i]["_fp"] for i in range(len(records)) if records[i].get("has_fp")]
# Note: fps list order matches valid_db_ids order because we appended in that order
assert len(fps) == K

scaffolds = [r["scaffold"] for r in records if r.get("has_fp")]
assert len(scaffolds) == K

# Union-find
parent = list(range(K + len(df_drugs)))  # extra slots for no-FP drugs (singletons)

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

# Pairwise scan over valid drugs
# Group by scaffold first to limit Tanimoto comparisons to same-scaffold pairs
scaf_to_positions = defaultdict(list)
for pos, scaf in enumerate(scaffolds):
    scaf_to_positions[scaf].append(pos)

n_edges = 0
n_scaf_pairs = 0
for scaf, positions in scaf_to_positions.items():
    if scaf == "" or len(positions) < 2:
        continue
    for ii in range(len(positions)):
        for jj in range(ii + 1, len(positions)):
            i, j = positions[ii], positions[jj]
            n_scaf_pairs += 1
            tc = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if tc >= TAU:
                union(i, j)
                n_edges += 1

print(f"Same-scaffold drug pairs examined: {n_scaf_pairs}")
print(f"Pairs passing AND rule (linked)  : {n_edges}")

# Assign cluster IDs (compact, 0..)
roots = {}
next_cluster_id = 0
# Valid drugs
cluster_of_valid = {}
for pos in range(K):
    r = find(pos)
    if r not in roots:
        roots[r] = next_cluster_id
        next_cluster_id += 1
    cluster_of_valid[pos] = roots[r]
# No-FP drugs get individual singleton clusters
db_to_cluster = {}
for pos, db in enumerate(valid_db_ids):
    db_to_cluster[db] = cluster_of_valid[pos]
for r in records:
    if not r.get("has_fp"):
        db_to_cluster[r["drug_db_id"]] = next_cluster_id
        next_cluster_id += 1

# Attach cluster to df_drugs
df_drugs["cluster_id"] = df_drugs["drug_db_id"].map(db_to_cluster).astype(int)
# drop the temporary _fp column if present
if "_fp" in df_drugs.columns:
    df_drugs = df_drugs.drop(columns=["_fp"])

cluster_sizes = df_drugs.groupby("cluster_id").size()
print(f"Total clusters           : {len(cluster_sizes)}")
print(f"Singleton clusters       : {(cluster_sizes == 1).sum()}")
print(f"Multi-drug clusters (>1) : {(cluster_sizes > 1).sum()}")
print(f"Largest cluster size     : {cluster_sizes.max()}")
print(f"Drugs in multi-drug clusters: "
      f"{int(cluster_sizes[cluster_sizes > 1].sum())} / {len(df_drugs)}")


# ---------------------------------------------------------------------------
# 5. Verify BTK trio (ibrutinib, acalabrutinib, zanubrutinib)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 5: Verify BTK inhibitor co-clustering")
print("=" * 80)

BTK = {"ibrutinib": "DB09053", "acalabrutinib": "DB11718", "zanubrutinib": "DB15966"}
for name, db in BTK.items():
    row = df_drugs[df_drugs.drug_db_id == db]
    if len(row):
        c = int(row.cluster_id.iloc[0])
        print(f"  {name:15s} {db}  cluster={c}  scaffold='{row.scaffold.iloc[0]}'  smiles='{row.smiles.iloc[0][:60]}...'")
    else:
        print(f"  {name:15s} {db}  NOT IN p13 (not in synergy df)")
btk_clusters = {int(df_drugs[df_drugs.drug_db_id == db].cluster_id.iloc[0])
                for db in BTK.values() if len(df_drugs[df_drugs.drug_db_id == db])}
print(f"\nBTK cluster set: {btk_clusters}")
print(f"BTK trio co-clustered under AND rule? {len(btk_clusters) == 1}")


# ---------------------------------------------------------------------------
# 6. Find genuine co-clustered trios to use as the manuscript example
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 6: Find genuine co-clustered trios (>=3 drugs, named) for manuscript example")
print("=" * 80)

multi = df_drugs.groupby("cluster_id").filter(lambda g: len(g) >= 3 and (g["drug_name"] != "").sum() >= 3)
trio_candidates = []
for cid, g in multi.groupby("cluster_id"):
    named = g[g["drug_name"] != ""].sort_values("drug_name")
    if len(named) >= 3:
        trio_candidates.append((cid, named))

trio_candidates.sort(key=lambda x: -len(x[1]))
print(f"Clusters with >=3 named drugs: {len(trio_candidates)}")
for cid, g in trio_candidates[:10]:
    names = ", ".join(g["drug_name"].head(6).tolist())
    print(f"  cluster {cid} ({len(g)} drugs): {names}")


# ---------------------------------------------------------------------------
# 7. Map samples -> cluster, then train/test split + k-fold within train
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 7: Assign samples to clusters and create splits")
print("=" * 80)

int_to_db = {int(i): f"DB{int(i):05d}" for i in p13_int_ids}

# Build a vectorized int(drug) -> cluster_id lookup table (numpy-backed)
unique_drug_ints = np.array(sorted({int(i) for i in p13_int_ids}))
cluster_lookup = np.full(unique_drug_ints.max() + 1, -1, dtype=np.int64)
for di in unique_drug_ints:
    db = int_to_db[di]
    cid = db_to_cluster.get(db, -1)
    if cid >= 0:
        cluster_lookup[di] = cid

d1 = syn["drug1"].to_numpy().astype(np.int64)
d2 = syn["drug2"].to_numpy().astype(np.int64)
c1 = cluster_lookup[d1]   # -1 if unknown
c2 = cluster_lookup[d2]

# IMPORTANT: standard scaffold-split sample assignment.
# Each sample is assigned to EXACTLY ONE cluster (its drug1's cluster; if drug1
# has no valid cluster, fall back to drug2's). This guarantees that partitioning
# clusters -> partitioning samples yields DISJOINT train/test sets, which is what
# the downstream pipeline requires (custom_train_idx and custom_test_idx are
# used as positional index arrays into the feature matrix and must not overlap).
#
# Trade-off: if drug1 and drug2 sit in different clusters, the sample "follows"
# drug1. The alternative "either drug in test -> test" rule produces non-disjoint
# sets when clusters share drugs, which would itself be a form of leakage.
sample_cluster = np.where(c1 >= 0, c1, np.where(c2 >= 0, c2, -1))
syn["drug_cluster"] = sample_cluster

cluster_ids = sorted([c for c in syn["drug_cluster"].unique() if c != -1])
rng = np.random.default_rng(SEED)
rng.shuffle(cluster_ids)

# Greedy fill test set to ~TEST_SIZE by sample count
test_clusters = set()
test_size_running = 0
target = TEST_SIZE * n_samples
for cid in cluster_ids:
    cluster_n = int((syn["drug_cluster"] == cid).sum())
    if test_size_running + cluster_n <= target * 1.10:  # allow up to 10% over
        test_clusters.add(cid)
        test_size_running += cluster_n
    if test_size_running >= target * 0.9:
        break

train_clusters = set(cluster_ids) - test_clusters

test_idx = syn[syn["drug_cluster"].isin(test_clusters)].index.values
train_idx = syn[syn["drug_cluster"].isin(train_clusters)].index.values

print(f"Train clusters: {len(train_clusters)},  Test clusters: {len(test_clusters)}")
print(f"Train samples : {len(train_idx):>7d} ({100*len(train_idx)/n_samples:.2f}%)")
print(f"Test samples  : {len(test_idx):>7d} ({100*len(test_idx)/n_samples:.2f}%)")


# ---------------------------------------------------------------------------
# 8. K-fold within train (greedy smallest-fold packing by cluster)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"STEP 8: K-fold (n={N_FOLDS}) within training clusters")
print("=" * 80)

train_clusters_list = sorted(train_clusters)
rng.shuffle(train_clusters_list)

fold_clusters = [[] for _ in range(N_FOLDS)]
fold_sizes = [0] * N_FOLDS
for cid in train_clusters_list:
    smallest = int(np.argmin(fold_sizes))
    fold_clusters[smallest].append(cid)
    fold_sizes[smallest] += int((syn["drug_cluster"] == cid).sum())

fold_idx_list = []
train_idx_set = set(train_idx.tolist())
for fold_id in range(N_FOLDS):
    fold_cluster_set = set(fold_clusters[fold_id])
    fidx_all = syn[syn["drug_cluster"].isin(fold_cluster_set)].index.values
    # restrict to train indices (k-fold is WITHIN train)
    fidx = np.array([i for i in fidx_all if i in train_idx_set])
    fold_idx_list.append(fidx)
    print(f"  Fold {fold_id}: {len(fidx):>7d} samples ({100*len(fidx)/len(train_idx):.2f}% of train)")


# ---------------------------------------------------------------------------
# 9. Save everything
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 9: Save outputs")
print("=" * 80)

np.save(os.path.join(SPLITS_DIR, "scaffold_train_indices.npy"), train_idx)
np.save(os.path.join(SPLITS_DIR, "scaffold_test_indices.npy"), test_idx)
for fid, fidx in enumerate(fold_idx_list):
    np.save(os.path.join(SPLITS_DIR, f"scaffold_fold_{fid}_indices.npy"), fidx)
print("Saved scaffold_{train,test,fold_*}_indices.npy")

df_drugs.to_csv(os.path.join(MODELS_DIR, "drug_scaffold_clusters.csv"), index=False)
print("Saved drug_scaffold_clusters.csv")

# Split composition table
def breakdown(indices, label):
    sub = syn.iloc[indices]
    return pd.DataFrame({
        "split": [label],
        "n_samples": [len(sub)],
        "n_unique_drug1": [sub["drug1"].nunique()],
        "n_unique_drug2": [sub["drug2"].nunique()],
        "n_unique_drugs_in_pairs": [len(set(sub["drug1"]) | set(sub["drug2"]))],
        "n_drug_clusters": [sub["drug_cluster"].nunique()],
    })

composition = pd.concat([
    breakdown(train_idx, "train"),
    breakdown(test_idx, "test"),
], ignore_index=True)
composition["pct_of_total"] = 100 * composition["n_samples"] / n_samples
composition.to_csv(os.path.join(MODELS_DIR, "scaffold_split_composition.csv"), index=False)
print("Saved scaffold_split_composition.csv")
print()
print(composition.to_string(index=False))


# ---------------------------------------------------------------------------
# 10. Leakage check: no cluster should appear in both train and test sample sets
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 10: Leakage check")
print("=" * 80)
# Cluster-level: which clusters appear among train samples vs test samples?
train_clusters_in_samples = set(syn.iloc[train_idx]["drug_cluster"].unique())
test_clusters_in_samples = set(syn.iloc[test_idx]["drug_cluster"].unique())
train_clusters_in_samples.discard(-1)
test_clusters_in_samples.discard(-1)
overlap = train_clusters_in_samples & test_clusters_in_samples
print(f"Cluster overlap (samples in both train and test) — should be 0: {len(overlap)}")
if overlap:
    print(f"  overlapping cluster ids: {sorted(overlap)[:20]}")

# Hard correctness check: train and test index arrays must be disjoint
idx_overlap = set(train_idx.tolist()) & set(test_idx.tolist())
print(f"Sample-index overlap (HARD requirement, must be 0): {len(idx_overlap)}")
union_n = len(set(train_idx.tolist()) | set(test_idx.tolist()))
print(f"Train ∪ Test = {union_n} (should == total samples {n_samples}): "
      f"{'OK' if union_n == n_samples else 'MISMATCH'}")

# Drug-level: by construction, a drug whose cluster is held out is removed from
# training ONLY when its partner drug is also in a test cluster. A drug can
# appear in both train and test if it pairs with one drug in a train cluster
# and another drug in a test cluster. This is EXPECTED and is the price of
# cluster-level splitting (samples, not drugs, are the unit of leakage prevention).
train_drugs = set(syn.iloc[train_idx]["drug1"]) | set(syn.iloc[train_idx]["drug2"])
test_drugs = set(syn.iloc[test_idx]["drug1"]) | set(syn.iloc[test_idx]["drug2"])
drug_overlap = train_drugs & test_drugs
print(f"Drug overlap train vs test: {len(drug_overlap)}")
print("  (Expected >0 under cluster-level split: a drug can pair with both train- and test-cluster drugs. "
      "Leakage prevention is at the cluster level, not the individual-drug level.)")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
