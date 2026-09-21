"""Extract fold-level ROC-AUC and test-set PR-AUC from all 24 DL logs.

Outputs:
  - fold_roc_aucs.csv: model, split, fold, roc_auc (5 folds per model x split)
  - test_metrics.csv:  model, split, roc_auc_test, pr_auc_test (single value)
"""
import re
import glob
import os
import pandas as pd

LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_SUBDIR = os.path.join(LOG_DIR, "logs_dl")

fold_rows = []
test_rows = []

for log in sorted(glob.glob(os.path.join(LOG_SUBDIR, "*_*.log"))):
    name = os.path.basename(log).replace(".log", "")
    if "_" not in name:
        continue
    # split on the LAST underscore: e.g. transynergy_liu_original -> (transynergy_liu, original)
    model, split = name.rsplit("_", 1)
    if split not in {"original", "scaffold", "lineage", "combined"}:
        continue
    with open(log) as f:
        txt = f.read()

    # Fold-level CV AUC: lines like "Fold 0: 0.8439..."
    folds = re.findall(r"^Fold (\d+):\s*([\d.]+)\s*$", txt, re.MULTILINE)
    for f_id, auc in folds:
        fold_rows.append({
            "model": model,
            "split": split,
            "fold": int(f_id) + 1,
            "roc_auc": float(auc),
        })

    # Test-set metrics: "AUC: 0.84...", "PRAUC: 0.70...", "AUROC: ...", "AUPR: ..."
    # Match the LAST occurrence (final eval).
    def last_match(pattern):
        m = re.findall(pattern, txt, re.MULTILINE)
        return float(m[-1]) if m else None

    auc_test = (
        last_match(r"^ROCAUC:\s*([\d.]+)\s*$")
        or last_match(r"^AUC:\s*([\d.]+)\s*$")
        or last_match(r"^AUROC:\s*([\d.]+)\s*$")
    )
    prauc_test = (
        last_match(r"^PRAUC:\s*([\d.]+)\s*$")
        or last_match(r"^AUPR:\s*([\d.]+)\s*$")
        or last_match(r"^pr_auc:\s*([\d.]+)\s*$")
    )

    test_rows.append({
        "model": model,
        "split": split,
        "roc_auc_test": auc_test,
        "pr_auc_test": prauc_test,
    })

fold_df = pd.DataFrame(fold_rows).sort_values(["model", "split", "fold"])
test_df = pd.DataFrame(test_rows).sort_values(["model", "split"])

fold_df.to_csv(os.path.join(LOG_DIR, "fold_roc_aucs.csv"), index=False)
test_df.to_csv(os.path.join(LOG_DIR, "test_metrics.csv"), index=False)

print("=== Fold-level ROC-AUC counts (should be 5 per model x split) ===")
print(fold_df.groupby(["model", "split"]).size().to_string())
print(f"\nTotal fold rows: {len(fold_df)} (expected 120 = 24 x 5)")

print("\n=== Test-set metrics ===")
print(test_df.to_string(index=False))
print(f"\nMissing roc_auc_test: {test_df['roc_auc_test'].isna().sum()}")
print(f"Missing pr_auc_test: {test_df['pr_auc_test'].isna().sum()}")
