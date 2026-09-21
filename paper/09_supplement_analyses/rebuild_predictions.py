#!/usr/bin/env python3
"""Rebuild predictions.csv on the common held-out split.

The previous predictions.csv pooled three different held-out splits in one file
(60,917 / 17,755 and 60,918 / 17,650 rows depending on the model), so the
per-model AUROC/AUPRC it produced were not comparable: they scored different
models against different test sets. Its `transynergy_liu` rows also carried the
60,917 / 17,755 fingerprint rather than TranSynergy's own split, which is why
`pr_auc_bootstrap_ci.csv` reported TranSynergy (0.7075) above PAIRWISE (0.6625)
while every same-split table has PAIRWISE ahead.

This script writes one long-format file from the six per-model prediction files
on the single 60,927-row split (17,686 Loewe-positive), so downstream bootstrap
CIs compare models on identical rows.

Input : ../../results/predicts_<model>_p13_original_thres0.txt
Output: predictions.csv  (model, split, y_true, y_score)
"""
import glob
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.realpath(os.path.join(HERE, "..", ".."))
PRED_GLOB = os.path.join(ROOT, "results", "predicts_*_p13_original_thres0.txt")
OUT_CSV = os.path.join(HERE, "predictions.csv")

PATTERN = re.compile(r"predicts_(.+)_p13_original_thres0\.txt$")


def main() -> None:
    frames = []
    for path in sorted(glob.glob(PRED_GLOB)):
        match = PATTERN.search(os.path.basename(path))
        if not match:
            continue
        model = match.group(1)

        table = pd.read_csv(path)
        score_col = [c for c in table.columns if c.startswith("predicts_")][0]
        frames.append(
            pd.DataFrame(
                {
                    "model": model,
                    "split": "random",
                    "y_true": table["actuals"].astype(int),
                    "y_score": table[score_col].astype(float),
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)

    sizes = out.groupby("model").size().unique()
    positives = out.groupby("model")["y_true"].sum().unique()
    assert len(sizes) == 1, f"models have different row counts: {sizes}"
    assert len(positives) == 1, f"models have different positive counts: {positives}"

    out.to_csv(OUT_CSV, index=False)
    print(out.groupby("model").agg(n=("y_true", "size"), n_pos=("y_true", "sum")))
    print(f"\nwrote {OUT_CSV} ({len(out)} rows)")


if __name__ == "__main__":
    main()
