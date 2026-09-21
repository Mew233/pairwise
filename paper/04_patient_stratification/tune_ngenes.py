#!/usr/bin/env python3
"""
Select the LPS gene-set size using LS-INTERNAL out-of-fold survival ONLY.

This is the single hyperparameter tuned, and it is tuned exclusively on the LS
cohort's held-out (OOF) signal. Selection criterion = mean held-out
concordance (Harrell C) of the continuous OOF LPS vs PFS across seeds, with the
dichotomized log-rank p reported alongside for context.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.model_selection import StratifiedKFold

from lps_classifier import LPSClassifier
from run_heldout_survival import (DATA, N_FOLDS, POS, cluster_train_labels,
                                  load_data)

GRID = [25, 50, 75, 100, 150, 200]
SEEDS = [1, 2, 3]


def oof(expr, mshap, samples, events, seed, n_genes):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    grp = pd.Series(index=samples, dtype=object)
    lps = pd.Series(index=samples, dtype=float)
    s = np.array(samples)
    for tr_idx, te_idx in skf.split(s, events):
        tr, te = list(s[tr_idx]), list(s[te_idx])
        labels = cluster_train_labels(mshap, tr)
        clf = LPSClassifier(n_genes=n_genes).fit(expr[tr], labels)
        pred = clf.predict(expr[te])
        grp.loc[te] = pred["group"].values
        lps.loc[te] = pred["lps"].values
    return grp, lps


def run():
    expr, mshap, clin = load_data()
    samples = [s for s in mshap.columns if s in expr.columns and s in clin.index]
    base = clin.loc[samples]
    ok = base["pfs_time"].notna() & base["pfs_event"].notna() & (base["pfs_time"] > 0)
    samples = [s for s, v in zip(samples, ok) if v]
    events = clin.loc[samples, "pfs_event"].astype(int).values
    t = clin.loc[samples, "pfs_time"].values
    e = clin.loc[samples, "pfs_event"].astype(int).values
    print(f"LS analyzable: n={len(samples)} events={int(e.sum())}\n")

    rows = []
    for n in GRID:
        cidx, lps_ps = [], []
        for seed in SEEDS:
            grp, lps = oof(expr, mshap, samples, events, seed, n)
            lz = ((lps - lps.mean()) / lps.std()).values
            # higher LPS = BTKiCombo(+) = better survival -> use -lz as risk
            cidx.append(concordance_index(t, -lz, e))
            g = (grp == POS).astype(int).values
            lr = logrank_test(t[g == 1], t[g == 0], e[g == 1], e[g == 0])
            lps_ps.append(lr.p_value)
        rows.append(dict(n_genes=n, c_index=np.mean(cidx),
                         logrank_p=np.exp(np.mean(np.log(lps_ps)))))
        print(f"  n_genes={n:>3}: mean C={np.mean(cidx):.4f} "
              f"geomean log-rank p={np.exp(np.mean(np.log(lps_ps))):.3g}")
    res = pd.DataFrame(rows)
    best = res.loc[res.c_index.idxmax(), "n_genes"]
    print(f"\nSelected n_genes = {int(best)} (max mean held-out C-index)")
    res.to_csv(f"{DATA}/../results/ngenes_selection.csv", index=False)
    return int(best)


if __name__ == "__main__":
    run()
