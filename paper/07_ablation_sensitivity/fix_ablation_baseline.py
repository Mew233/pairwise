#!/usr/bin/env python3
"""Set the `full` (and `exp`) baseline of ablation.csv from PAIRWISE's own CV folds.

The ablation table compares each variant against the unmodified PAIRWISE model.
That baseline must therefore be PAIRWISE's own score under the same protocol as
the variants, which is the 5-fold cross-validated AUROC recorded per fold in
paper/01_benchmark_p13/results/cv_folds/pairwise_seed42.csv (mean 0.8520).

`exp` is set to the same value: expression is the default cell-omics stream, so
that row is a control restating the unmodified configuration, not a separate
variant.

Input : ../01_benchmark_p13/results/cv_folds/pairwise_seed42.csv
Output: results/ablation.csv (rewritten in place)
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDS = os.path.join(
    HERE, "..", "01_benchmark_p13", "results", "cv_folds", "pairwise_seed42.csv"
)
ABLATION_CSV = os.path.join(HERE, "results", "ablation.csv")

BASELINE_ROWS = ("full", "exp")


def main() -> None:
    folds = pd.read_csv(FOLDS)["AUROC"]
    baseline = float(folds.mean())

    table = pd.read_csv(ABLATION_CSV)
    before = {r.config: r.AUROC for _, r in table.iterrows() if r.config in BASELINE_ROWS}
    table.loc[table["config"].isin(BASELINE_ROWS), "AUROC"] = baseline
    table.to_csv(ABLATION_CSV, index=False)

    print(f"PAIRWISE 5-fold CV baseline = {baseline:.16f} (n_folds={len(folds)})")
    for row in BASELINE_ROWS:
        print(f"  {row}: {before.get(row)} -> {baseline}")

    variants = table[~table["config"].isin(BASELINE_ROWS)]
    above = variants[variants["AUROC"] > baseline]
    if len(above):
        print("\nWARNING: variants scoring above the unmodified baseline:")
        print(above.to_string(index=False))
    else:
        print("\nAll variants fall at or below the baseline, as expected.")


if __name__ == "__main__":
    main()
