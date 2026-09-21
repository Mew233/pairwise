#!/usr/bin/env python3
"""

Reads results/cv_folds/<model>_seed42.csv -- one row per CV fold, written during
training -- and reports the mean, spread and a normal-approximation 95% CI over
the five folds. Deriving the table this way means every number in it traces back
to a file in the deposit; no value is carried over by hand.

Input : results/cv_folds/*_seed42.csv
Output: results/predictions_auroc.csv
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FOLD_DIR = os.path.join(HERE, "results", "cv_folds")
OUT_CSV = os.path.join(HERE, "results", "predictions_auroc.csv")

DISPLAY_NAMES = {
    "pairwise": "PAIRWISE",
    "transynergy_liu": "Transynergy",
    "multitaskdnn_kim": "Multitask-DNN",
    "deepdds_wang": "DeepDDS",
    "deepsynergy_preuer": "DeepSynergy",
    "matchmaker_brahim": "MatchMaker",
}


def main() -> None:
    rows = []
    for path in sorted(glob.glob(os.path.join(FOLD_DIR, "*_seed42.csv"))):
        tag = os.path.basename(path).replace("_seed42.csv", "")
        auroc = pd.read_csv(path)["AUROC"].values
        n = len(auroc)
        mean = auroc.mean()
        sd = auroc.std(ddof=1)
        se = sd / np.sqrt(n)
        rows.append(
            {
                "model": DISPLAY_NAMES.get(tag, tag),
                "model_tag": tag,
                "n_folds": n,
                "cv_auroc_mean": round(mean, 4),
                "cv_auroc_sd": round(sd, 4),
                "cv_auroc_se": round(se, 4),
                "ci95_low": round(mean - 1.96 * se, 4),
                "ci95_high": round(mean + 1.96 * se, 4),
                "seed": 42,
            }
        )

    table = (
        pd.DataFrame(rows)
        .sort_values("cv_auroc_mean", ascending=False)
        .reset_index(drop=True)
    )
    table.to_csv(OUT_CSV, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
