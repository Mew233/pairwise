#!/usr/bin/env python3
"""Rebuild loewe_sensitivity.csv from the shipped per-threshold prediction files.

Every row is recomputed from results/predicts_<model>_p13_original_thres<t>.txt,
so the table cannot drift from the predictions it claims to summarise. Those
files hold `actuals` (label at that Loewe threshold) and the model's score for
all 60,927 held-out combinations.

The previous table carried two rows that did not follow from any shipped
prediction file: PAIRWISE at threshold 0 read 0.842441 while its own predictions
give 0.812441 (AUPRC matched to 16 digits, AUROC differed by exactly 0.03), and
transynergy_liu read 0.779560 against a recomputed 0.839736. The four remaining
baselines agreed to six decimals, which is what flagged the two outliers.

Input : ../../results/predicts_<model>_p13_original_thres<t>.txt
Output: results/loewe_sensitivity.csv
"""
import glob
import os
import re

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))
PRED_GLOB = os.path.join(ROOT, "results", "predicts_*_p13_original_thres*.txt")
OUT_CSV = os.path.join(HERE, "results", "loewe_sensitivity.csv")

PATTERN = re.compile(r"predicts_(.+)_p13_original_thres(\d+)\.txt$")


def main() -> None:
    rows = []
    for path in sorted(glob.glob(PRED_GLOB)):
        match = PATTERN.search(os.path.basename(path))
        if not match:
            continue
        model, threshold = match.group(1), int(match.group(2))

        table = pd.read_csv(path)
        score_col = [c for c in table.columns if c.startswith("predicts_")][0]
        y_true, y_score = table["actuals"], table[score_col]

        rows.append(
            {
                "model": model,
                "threshold": threshold,
                "AUROC": roc_auc_score(y_true, y_score),
                "AUPRC": average_precision_score(y_true, y_score),
                "n_pos": int(y_true.sum()),
                "n_total": len(table),
            }
        )

    out = pd.DataFrame(rows).sort_values(["model", "threshold"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
