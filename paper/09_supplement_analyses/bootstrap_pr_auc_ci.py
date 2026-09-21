"""Bootstrap 95% percentile CI for PR-AUC (average precision) per model x split.

Input:  predictions.csv   (model, split, y_true, y_score) — built by
        rebuild_predictions.py on the common 60,927-row split
Output: pr_auc_bootstrap_ci.csv
          columns: model, split, pr_point, pr_ci_lo, pr_ci_hi, n_boot

Method: stratified bootstrap (B=2000) resampling test-set rows with replacement,
        preserving the positive/negative class ratio. PR-AUC = average_precision_score
        recomputed on each resample. CI = 2.5th and 97.5th percentiles.

pr_point is recomputed from the predictions, so the point estimate and its CI
always describe the same data. An earlier version read pr_point from a separate
test_metrics.csv "reported value" instead, which let the published point drift
away from the predictions it was supposed to summarise.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.abspath(__file__))
PRED_PATH = os.path.join(ROOT, "predictions.csv")
OUT_PATH = os.path.join(ROOT, "pr_auc_bootstrap_ci.csv")

N_BOOT = 2000
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5
SEED = 42

pred = pd.read_csv(PRED_PATH)

rng = np.random.default_rng(SEED)
rows = []
t0 = time.time()
cells = sorted(set(pred[["model", "split"]].itertuples(index=False, name=None)))

for i, (model, split) in enumerate(cells, 1):
    cell = pred[(pred.model == model) & (pred.split == split)].reset_index(drop=True)
    if len(cell) == 0:
        print(f"[{i}/{len(cells)}] {model:<22}{split:<10} MISSING predictions")
        continue

    y = cell["y_true"].to_numpy()
    s = cell["y_score"].to_numpy()
    n = len(cell)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    # stratified bootstrap: sample n_pos positives and n_neg negatives with replacement
    boot_scores = np.empty(N_BOOT)
    for b in range(N_BOOT):
        samp_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        samp_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        samp = np.concatenate([samp_pos, samp_neg])
        boot_scores[b] = average_precision_score(y[samp], s[samp])

    pr_point = float(average_precision_score(y, s))

    ci_lo = float(np.percentile(boot_scores, CI_LO_PCT))
    ci_hi = float(np.percentile(boot_scores, CI_HI_PCT))
    boot_mean = float(boot_scores.mean())

    rows.append({
        "model": model,
        "split": split,
        "pr_point": round(pr_point, 4),
        "pr_ci_lo": round(ci_lo, 4),
        "pr_ci_hi": round(ci_hi, 4),
        "n_boot": N_BOOT,
        "boot_mean": round(boot_mean, 4),
        "n_test": n,
    })
    elapsed = time.time() - t0
    print(f"[{i}/{len(cells)}] {model:<22}{split:<10} point={pr_point:.4f} "
          f"CI=[{ci_lo:.4f}, {ci_hi:.4f}] boot_mean={boot_mean:.4f} "
          f"({elapsed:.1f}s elapsed)")

out = pd.DataFrame(rows).sort_values(["model", "split"]).reset_index(drop=True)
out[["model", "split", "pr_point", "pr_ci_lo", "pr_ci_hi", "n_boot"]].to_csv(
    OUT_PATH, index=False
)
print(f"\nWrote {len(out)} rows to {OUT_PATH}")
print("\nFinal table:")
print(out.to_string(index=False))
