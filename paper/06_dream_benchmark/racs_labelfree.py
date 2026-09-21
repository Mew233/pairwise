"""
RACS — label-free reimplementation (matched conditions) for NCI-DREAM SC2 / OCI-LY3.

Published RACS (Sun et al., Nat Commun 2015) is SEMI-SUPERVISED: it ranks pairs by
similarity to KNOWN synergistic seeds in a multi-feature space built from (i) drug
targeting networks, (ii) transcriptomic profiles, (iii) drug pharmacology. Its reported
0.78 used known synergy LABELS and a 78-pair subset.

Per the reviewer's request, this version removes ALL synergy-label supervision so that
RACS uses only the same single-agent (monotherapy) OCI-LY3 information PAIRWISE is
calibrated on. We keep RACS's feature construction but aggregate the features in an
UNSUPERVISED way (equal-weight z-score fusion, no training on outcomes).

DOCUMENTED ASSUMPTIONS (so this is defensible, not a strawman):
  F1 Transcriptomic feature: cosine similarity of the two single-drug gene-level logFC
     signatures over core DEGs. (DREAM finding: concordant single-drug transcriptional
     programs -> synergy; this is the feature RACS draws from transcriptomics.)
  F2 Target-network feature: Jaccard overlap of curated primary targets + a shared
     functional-class term. (Proxy for RACS's drug-target network proximity; no external
     PPI DB is fetched — swap in STRING/your network to refine.)
  Fusion: z-score F1,F2 across the 91 pairs and sum (equal weights). NO labels used.
Targets below are standard pharmacology for the 14 DREAM compounds; refine if you have a
curated source.
"""
import numpy as np, pandas as pd, csv
from itertools import combinations
import syngen
from dream_pcindex import evaluate

# curated primary targets / functional class (standard pharmacology)
TARGETS = {
    "Aclacinomycin A": (["TOP1", "TOP2A"], "DNA"),
    "Blebbistatin":    (["MYH9", "MYH10"], "cytoskeleton"),
    "Camptothecin":    (["TOP1"], "DNA"),
    "Cycloheximide":   ([], "translation"),
    "Doxorubicin":     (["TOP2A", "TOP2B"], "DNA"),
    "Etoposide":       (["TOP2A", "TOP2B"], "DNA"),
    "Geldanamycin":    (["HSP90AA1", "HSP90AB1"], "chaperone"),
    "H-7":             (["PRKCA", "PRKCB", "PRKACA"], "kinase"),
    "Methotrexate":    (["DHFR", "TYMS", "ATIC"], "antimetabolite"),
    "Mitomycin C":     ([], "DNA"),
    "Monastrol":       (["KIF11"], "cytoskeleton"),
    "Rapamycin":       (["MTOR", "FKBP1A"], "kinase"),
    "Trichostatin A":  (["HDAC1", "HDAC2", "HDAC3"], "epigenetic"),
    "Vincristine":     (["TUBB", "TUBA1A"], "cytoskeleton"),
}

def core_signatures(timepoint="24", deg_cut=0.5):
    genes, expr, meta = syngen.load()
    fc = syngen.drug_logfc(genes, expr, meta, timepoint=timepoint)
    sig = fc.copy()
    core = (sig.abs() >= deg_cut).any(axis=1)
    return sig[core]                       # core genes x drugs (signed logFC)

def f1_transcriptomic(sig, a, b):
    va, vb = sig[a].values, sig[b].values
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

def f2_targetnet(a, b):
    ta, ca = TARGETS[a]; tb, cb = TARGETS[b]
    sa, sb = set(ta), set(tb)
    jacc = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    same_class = 1.0 if ca == cb else 0.0
    return 0.5 * jacc + 0.5 * same_class

def racs_labelfree(timepoint="24", deg_cut=0.5):
    sig = core_signatures(timepoint, deg_cut)
    drugs = list(sig.columns)
    rows = []
    for a, b in combinations(drugs, 2):
        rows.append((a, b, f1_transcriptomic(sig, a, b), f2_targetnet(a, b)))
    df = pd.DataFrame(rows, columns=["A", "B", "F1", "F2"])
    # unsupervised equal-weight fusion (no synergy labels)
    z = lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1)
    df["score"] = z(df["F1"]) + z(df["F2"])
    df["Rank"] = df["score"].rank(ascending=False, method="first").astype(int)
    return df

# ---------------------------------------------------------------------------
# RACS-style semi-supervised ranking (replaces Option B).
# ---------------------------------------------------------------------------
def native_scores(**kwargs):
    import semisup_common as sc
    cfg = sc.DEFAULTS["racs"]
    df = racs_labelfree(str(cfg["timepoint"]), float(cfg["deg_cut"]))
    return df[["A", "B", "score"]]


def semisup(verbose=True):
    import semisup_common as sc
    return sc.run_semisup("racs", native_scores, "racs_semisup_ranking.csv", verbose=verbose)


if __name__ == "__main__":
    print(f"{'timepoint':>9}{'deg_cut':>8}{'pc-index':>10}{'p':>8}")
    best = None
    for tp in ["6", "12", "24"]:
        for cut in [0.3, 0.5, 0.8]:
            df = racs_labelfree(tp, cut)
            out = f"racs_lf_t{tp}_c{cut}.csv"
            with open(out, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["Drug Combination", "Rank"])
                for _, r in df.iterrows():
                    w.writerow([f"{r.A} & {r.B}", r.Rank])
            pc, pv = evaluate(out, n_perm=2000)
            print(f"{tp:>9}{cut:>8}{pc:>10.4f}{pv:>8.4f}")
            if best is None or pc > best[0]:
                best = (pc, pv, tp, cut, out)
    # also report transcriptomic-only and target-only ablations at the default config
    df = racs_labelfree("24", 0.5)
    for feat in ["F1", "F2"]:
        d = df.copy(); d["Rank"] = d[feat].rank(ascending=False, method="first").astype(int)
        out = f"racs_lf_{feat}_only.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["Drug Combination", "Rank"])
            for _, r in d.iterrows(): w.writerow([f"{r.A} & {r.B}", r.Rank])
        pc, pv = evaluate(out, n_perm=2000)
        print(f"  ablation {feat}-only (t24,c0.5): pc={pc:.4f} p={pv:.4f}")
    print(f"\nBest label-free RACS: t={best[2]} cut={best[3]} -> pc={best[0]:.4f} (p={best[1]:.4f})")
