"""Task 7 — RWR restart-probability (alpha) sweep metrics, full precision.

Each alpha in {0.1, 0.3, 0.5, 0.7, 0.9} corresponds to a PAIRWISE model trained
with drug-target RWR features propagated at that restart probability
(checkpoints: results/task7_rwr/checkpoints/best_model_pairwise_alpha_<a>.pth).
This script recomputes held-out test AUROC and a bootstrap 95% CI
(percentile, 1000 resamples, seed 0) from the shipped per-alpha test
predictions — no retraining.

Inputs:
    results/task7_rwr/predicts/predicts_pairwise_alpha_<a>_p13.txt
        columns: index, actuals, predicts_pairwise

Outputs:
    paper/03_network_pathway/results/rwr_sweep.csv
    columns: r, AUROC, ci_low, ci_high

Usage:
    python eval_rwr_sweep.py            # from the repo root
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRED_DIR = os.path.join(ROOT_DIR, "results", "task7_rwr", "predicts")
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "rwr_sweep.csv")

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_BOOT = 1000
SEED = 0


def bootstrap_auroc_ci(y, scores, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(roc_auc_score(y[idx], scores[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    rows = []
    for alpha in ALPHAS:
        path = os.path.join(PRED_DIR, f"predicts_pairwise_alpha_{alpha}_p13.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path)
        pred_col = [c for c in df.columns if c.startswith("predicts_")][0]
        y = df["actuals"].to_numpy()
        s = df[pred_col].to_numpy()
        auroc = float(roc_auc_score(y, s))
        lo, hi = bootstrap_auroc_ci(y, s)
        rows.append({"r": alpha, "AUROC": auroc, "ci_low": lo, "ci_high": hi})
        print(f"alpha={alpha}: AUROC={auroc:.6f}  95% CI=[{lo:.6f}, {hi:.6f}]  (n={len(df)})")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
