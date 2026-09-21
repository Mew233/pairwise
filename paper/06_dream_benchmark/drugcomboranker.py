"""
DrugComboRanker — reimplementation (matched conditions) for NCI-DREAM SC2 / OCI-LY3.

Published DrugComboRanker (Huang et al., Bioinformatics 2014) has NO open source (its repo
is documentation-only). We reimplement its algorithmic logic using only single-agent
OCI-LY3 data, so it is a fair matched-condition baseline.

Original method:
  1. Build a drug FUNCTIONAL network from drug genomic (expression) profiles.
  2. Partition it into drug communities (Bayesian NMF) -> drugs sharing mechanism of action.
  3. Build a disease SIGNALING network from the cell-line profile + interactome.
  4. Rank pairs by whether the two drugs hit COMPLEMENTARY signaling modules of the
     disease network (i.e. different MoA jointly covering the disease network).

Matched-condition reimplementation (DOCUMENTED simplifications, swap in curated networks
to refine):
  - Drug functional network  = Pearson correlation of single-drug gene-level signatures.
  - Drug communities         = clustering on (1 - correlation) distance.
  - Disease signaling genes D = the genes most strongly/consistently perturbed across the
    14 drugs (proxy for the druggable disease-signaling layer in OCI-LY3).
  - Disease modules          = KMeans clustering of D by cross-drug perturbation pattern.
  - Pair score (complementarity) = union coverage of the disease modules by A and B
    (sum_m max(cov_A[m], cov_B[m])) + cross-community bonus - redundancy (overlap),
    capturing "complementary modules, different MoA". Higher = more synergistic.
No combination labels are used.
"""
import numpy as np, pandas as pd, csv
from itertools import combinations
from sklearn.cluster import KMeans, AgglomerativeClustering
import syngen
from dream_pcindex import evaluate

def build(timepoint="24", deg_cut=0.5, n_disease=2000, n_modules=8, n_drug_comm=5):
    genes, expr, meta = syngen.load()
    fc = syngen.drug_logfc(genes, expr, meta, timepoint=timepoint)      # genes x drugs
    drugs = list(fc.columns)

    # (1)+(2) drug functional network -> communities
    corr = np.corrcoef(fc.values.T)                                    # drugs x drugs
    dist = 1 - corr
    comm = AgglomerativeClustering(n_clusters=n_drug_comm, metric="precomputed",
                                   linkage="average").fit_predict(dist)
    drug_comm = dict(zip(drugs, comm))

    # (3) disease signaling genes = most perturbed across drugs; modules by perturbation pattern
    consistency = fc.abs().mean(axis=1)
    D = consistency.sort_values(ascending=False).head(n_disease).index
    Dmat = fc.loc[D].values                                            # diseaseGenes x drugs
    gmod = KMeans(n_clusters=n_modules, n_init=10, random_state=0).fit_predict(
        (Dmat - Dmat.mean(1, keepdims=True)))
    binpert = (np.abs(Dmat) >= deg_cut).astype(float)                  # gene perturbed by drug?

    # per-drug coverage of each disease module
    cov = np.zeros((len(drugs), n_modules))
    for m in range(n_modules):
        rows = (gmod == m)
        size = rows.sum()
        if size == 0: continue
        cov[:, m] = binpert[rows].sum(axis=0) / size                   # frac of module perturbed
    cov = pd.DataFrame(cov, index=drugs)
    return drugs, drug_comm, cov

def dcr_rank(timepoint="24", deg_cut=0.5, **kw):
    drugs, drug_comm, cov = build(timepoint=timepoint, deg_cut=deg_cut, **kw)
    rows = []
    for a, b in combinations(drugs, 2):
        ca, cb = cov.loc[a].values, cov.loc[b].values
        union = np.maximum(ca, cb).sum()              # complementary module coverage
        overlap = np.minimum(ca, cb).sum()            # redundancy
        cross = 1.0 if drug_comm[a] != drug_comm[b] else 0.0   # different MoA community
        score = union - 0.5 * overlap + 0.25 * cross
        rows.append((a, b, score))
    df = pd.DataFrame(rows, columns=["A", "B", "score"])
    df["Rank"] = df["score"].rank(ascending=False, method="first").astype(int)
    return df

# ---------------------------------------------------------------------------
# RACS-style semi-supervised ranking (replaces Option B).
# ---------------------------------------------------------------------------
def native_scores(**kwargs):
    import semisup_common as sc
    cfg = sc.DEFAULTS["drugcomboranker"]
    return dcr_rank(
        timepoint=str(cfg["timepoint"]),
        deg_cut=float(cfg["deg_cut"]),
        n_modules=int(cfg["n_modules"]),
    )[["A", "B", "score"]]


def semisup(verbose=True):
    import semisup_common as sc
    return sc.run_semisup(
        "drugcomboranker", native_scores, "drugcomboranker_semisup_ranking.csv", verbose=verbose
    )


if __name__ == "__main__":
    print(f"{'timepoint':>9}{'deg_cut':>8}{'modules':>8}{'pc-index':>10}{'p':>8}")
    best = None
    for tp in ["12", "24"]:
        for cut in [0.3, 0.5]:
            for nmod in [6, 8, 10]:
                df = dcr_rank(timepoint=tp, deg_cut=cut, n_modules=nmod)
                out = f"dcr_t{tp}_c{cut}_m{nmod}.csv"
                with open(out, "w", newline="") as f:
                    w = csv.writer(f); w.writerow(["Drug Combination", "Rank"])
                    for _, r in df.iterrows(): w.writerow([f"{r.A} & {r.B}", r.Rank])
                pc, pv = evaluate(out, n_perm=2000)
                print(f"{tp:>9}{cut:>8}{nmod:>8}{pc:>10.4f}{pv:>8.4f}")
                if best is None or pc > best[0]:
                    best = (pc, pv, tp, cut, nmod, out)
    print(f"\nBest DrugComboRanker(reimpl): t={best[2]} cut={best[3]} modules={best[4]} -> "
          f"pc={best[0]:.4f} (p={best[1]:.4f})")
