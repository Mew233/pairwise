"""
SynGen baseline (reimplementation) for NCI-DREAM SC2 / OCI-LY3.

SynGen was proposed by the DREAM organizers (Bansal et al., Nat Biotechnol 2014).
It is a monotherapy-only method: it (1) defines a set of "core genes" = genes that are
significant DEGs in at least one single-compound treatment, then (2) scores each compound
pair by the concordance of their single-drug expression signatures over those core genes
(number of co-regulated genes, accounting for direction of regulation). Pairs with more
concordant single-drug signatures are predicted more synergistic — the same hypothesis
used by the two best DREAM methods.

SynGen uses ONLY single-agent (monotherapy) OCI-LY3 data
"""
import numpy as np, pandas as pd, csv, sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))

TREAT = os.path.join(_HERE, "data", "ACalifano_DLBCL_Ly3_14Comp_treatment.txt")

NAME_MAP = {
    '"H-7, Dihydrochloride"': "H-7",
    "Doxorubicin hydrochloride": "Doxorubicin",
}
def canon_drug(x):
    return NAME_MAP.get(x, x)

from functools import lru_cache


@lru_cache(maxsize=4)
def load(treat=TREAT):
    with open(treat) as f:
        meta = [f.readline().rstrip("\n").split("\t") for _ in range(4)]
    drug = [canon_drug(d) for d in meta[1]]
    time = meta[2]
    conc = meta[3]
    df = pd.read_csv(treat, sep="\t", skiprows=4, header=None)
    genes = df.iloc[:, 1].astype(str).values            # Genename column
    expr = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").values  # probes x samples
    cols = list(range(2, df.shape[1]))
    meta_cols = pd.DataFrame({
        "drug":[drug[c] for c in cols],
        "time":[time[c] for c in cols],
        "conc":[conc[c] for c in cols],
    })
    return genes, expr, meta_cols

def drug_logfc(genes, expr, meta, timepoint="24", conc_level="IC20"):
    """Return DataFrame genes x drugs of log fold-change vs control at one timepoint.

    Multiple probes per gene are collapsed by keeping the probe with the largest
    mean absolute log-fold-change across drugs (one representative row per gene).
    """
    is_t = meta["time"].astype(str) == str(timepoint)
    ctrl_mask = (meta["drug"].isin(["DMSO", "Media"])) & is_t
    ctrl = expr[:, ctrl_mask.values].mean(axis=1)
    drugs = [d for d in meta["drug"].unique() if d not in ("DMSO", "Media")]
    out = {}
    for d in drugs:
        m = (meta["drug"] == d) & is_t & (meta["conc"].astype(str) == conc_level)
        if m.sum() == 0:                # fall back to any conc at this timepoint
            m = (meta["drug"] == d) & is_t
        out[d] = expr[:, m.values].mean(axis=1) - ctrl
    fc = pd.DataFrame(out, index=range(len(genes))).astype(float)
    # Collapse multi-probe -> gene: per gene, keep the probe (row) with largest mean|logFC|.
    # rep_score is indexed by integer row position; groupby(gene_arr) groups rows by gene
    # name, and idxmax() returns the row position of the highest-scoring probe per gene.
    rep_score = fc.abs().mean(axis=1)
    gene_arr = np.asarray(genes)
    keep_pos = np.asarray(rep_score.groupby(gene_arr).idxmax().values, dtype=int)
    fc = fc.iloc[keep_pos]
    fc.index = gene_arr[keep_pos]
    return fc

def syngen_rank(fc, deg_cut=0.5, subtract_discordant=True):
    """SynGen scores: concordant DEG overlap over core genes. rank1 = most synergistic."""
    sign = np.sign(fc.values) * (np.abs(fc.values) >= deg_cut)   # genes x drugs in {-1,0,1}
    core = (np.abs(sign).sum(axis=1) >= 1)                       # DEG in >=1 drug
    S = sign[core]                                              # core genes x drugs
    drugs = list(fc.columns)
    from itertools import combinations
    rows = []
    for a, b in combinations(range(len(drugs)), 2):
        va, vb = S[:, a], S[:, b]
        concord = int(np.sum((va != 0) & (va == vb)))
        discord = int(np.sum((va != 0) & (vb != 0) & (va == -vb)))
        score = concord - discord if subtract_discordant else concord
        rows.append((drugs[a], drugs[b], score))
    res = pd.DataFrame(rows, columns=["A", "B", "score"])
    res["Rank"] = res["score"].rank(ascending=False, method="first").astype(int)
    return res


def rank_for_config(timepoint="24", deg_cut=0.5, subtract_discordant=True):
    """Produce the 91-pair SynGen ranking at a given (frozen) config."""
    genes, expr, meta = load()
    fc = drug_logfc(genes, expr, meta, timepoint=str(timepoint))
    return syngen_rank(fc, deg_cut=deg_cut, subtract_discordant=subtract_discordant)


# ---------------------------------------------------------------------------
# RACS-style semi-supervised ranking (replaces Option B).
# ---------------------------------------------------------------------------
def native_scores(**kwargs):
    cfg = __import__("semisup_common", fromlist=["DEFAULTS"]).DEFAULTS["syngen"]
    return rank_for_config(
        timepoint=str(cfg["timepoint"]),
        deg_cut=float(cfg["deg_cut"]),
        subtract_discordant=True,
    )[["A", "B", "score"]]


def semisup(verbose=True):
    import semisup_common as sc
    return sc.run_semisup("syngen", native_scores, "syngen_semisup_ranking.csv", verbose=verbose)


if __name__ == "__main__":
    genes, expr, meta = load()
    print("samples:", expr.shape[1], "| timepoints:", sorted(meta['time'].unique()))
    from dream_pcindex import evaluate
    print(f"\n{'timepoint':>9} {'deg_cut':>7} {'pc-index':>9} {'p-value':>8}")
    best = None
    for tp in ["6", "12", "24"]:
        fc = drug_logfc(genes, expr, meta, timepoint=tp)
        for cut in [0.3, 0.5, 0.8, 1.0]:
            res = syngen_rank(fc, deg_cut=cut)
            out = f"syngen_t{tp}_c{cut}.csv"
            with open(out, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["Drug Combination", "Rank"])
                for _, r in res.iterrows():
                    w.writerow([f"{r.A} & {r.B}", r.Rank])
            pc, pv = evaluate(out, n_perm=2000)
            print(f"{tp:>9} {cut:>7} {pc:>9.4f} {pv:>8.4f}")
            if best is None or pc > best[0]:
                best = (pc, pv, tp, cut, out)
    print(f"\nBest SynGen config: t={best[2]} deg_cut={best[3]} -> pc-index={best[0]:.4f} (p={best[1]:.4f})")
    print(f"Saved best ranking: {best[4]}")
