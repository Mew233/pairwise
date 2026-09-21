#!/usr/bin/env python3
"""

Scored on the common held-out split: 60,927 combinations, 17,686 of them
Loewe-positive, identical for the six models compared here.

The CI is a stratified percentile bootstrap (1000 resamples, seed 0) that keeps
the positive/negative counts fixed.

Input : predictions/predicts_<model>_p13.txt
Output: results/test_auroc_bootstrap.csv
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PRED_GLOB = os.path.join(HERE, "predictions", "predicts_*_p13.txt")
OUT_CSV = os.path.join(HERE, "results", "test_auroc_bootstrap.csv")

# Run on a different held-out split; see the module docstring.
EXCLUDE = {"TGSynergy"}

N_BOOT = 1000
SEED = 0


def bootstrap_ci(y_true, y_score, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)

    draws = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, len(pos), replace=True),
             rng.choice(neg, len(neg), replace=True)]
        )
        draws.append(roc_auc_score(y_true[idx], y_score[idx]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    rows = []
    for path in sorted(glob.glob(PRED_GLOB)):
        model = re.sub(r"^predicts_|_p13\.txt$", "", os.path.basename(path))
        if model in EXCLUDE:
            continue
        table = pd.read_csv(path)
        score_col = [c for c in table.columns if c.startswith("predicts_")][0]
        y_true, y_score = table["actuals"], table[score_col]

        auroc = roc_auc_score(y_true, y_score)
        ci_low, ci_high = bootstrap_ci(y_true, y_score)
        rows.append(
            {
                "model": model,
                "test_auroc": round(auroc, 6),
                "boot_ci_low": round(ci_low, 6),
                "boot_ci_high": round(ci_high, 6),
                "n": len(table),
                "n_positive": int(y_true.sum()),
                "n_boot": N_BOOT,
                "seed": SEED,
            }
        )

    out = (
        pd.DataFrame(rows)
        .sort_values("test_auroc", ascending=False)
        .reset_index(drop=True)
    )
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
