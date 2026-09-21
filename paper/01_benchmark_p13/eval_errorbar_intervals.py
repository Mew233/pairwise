"""Bootstrap and 5-fold error bars for the p13 benchmark (no retraining).

Reads ``predictions/predicts_<model>_p13.txt`` (held-out test predictions) and
``results/cv_folds/<model>_seed42.csv`` (per-fold AUROCs replayed from the
out-of-fold metapredicts, see ``cv_utils.py``).

Output table ``results/errorbar_intervals.csv`` keeps the two evaluation
protocols in separate, clearly named columns:

  test_auroc / boot_ci_low / boot_ci_high
      AUROC on the held-out test set with a bootstrap 95% CI
      (percentile method, n_boot resamples, seed fixed).

  cv_mean_auroc / sd_5fold / cv_min / cv_max
      Mean, SD (ddof=0) and range of the 5 CV-fold AUROCs.


Usage:
    python eval_errorbar_intervals.py --synergy_df p13
    python eval_errorbar_intervals.py --synergy_df p13 --write
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cv_utils import (
    fold_aurocs_from_metapredicts,
    get_fold_val_synergy_indices,
    load_training_arrays,
    make_model_args,
)
from pairwise.paths import RESULTS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(HERE, "predictions")
OUT_DIR = os.path.join(HERE, "results")
CV_FOLDS_DIR = os.path.join(OUT_DIR, "cv_folds")
OUTPUT_CSV = os.path.join(OUT_DIR, "errorbar_intervals.csv")

MODELS = [
    "pairwise",
    "transynergy_liu",
    "matchmaker_brahim",
    "deepsynergy_preuer",
    "TGSynergy",
    "multitaskdnn_kim",
    "deepdds_wang",
]


def parse_args(cli_args=None):
    parser = argparse.ArgumentParser(description="p13 error bar intervals")
    parser.add_argument("--synergy_df", type=str, default="p13")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--boot_seed", type=int, default=0)
    parser.add_argument(
        "--skip_folds",
        action="store_true",
        help="only recompute test AUROC and bootstrap CI; leave per-fold columns empty",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write results/errorbar_intervals.csv (default: print only)",
    )
    return parser.parse_args(cli_args)


def load_test_predictions(model, synergy_df):
    path = os.path.join(PREDICTIONS_DIR, f"predicts_{model}_{synergy_df}.txt")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    pred_cols = [c for c in df.columns if c.startswith("predicts_")]
    if not pred_cols:
        raise ValueError(f"No prediction column in {path}")
    out = df[["actuals", pred_cols[0]]].rename(columns={pred_cols[0]: "pred"})
    return out, path


def load_meta_predictions(model, synergy_df):
    path = os.path.join(
        RESULTS_DIR, "meta_clf", f"metapredicts_{model}_{synergy_df}.txt"
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    pred_cols = [c for c in df.columns if c.startswith("metapredicts_")]
    if not pred_cols:
        raise ValueError(f"No metapredict column in {path}")
    out = df[["index", "actuals", pred_cols[0]]].rename(columns={pred_cols[0]: "pred"})
    return out, path


def bootstrap_auroc_ci(y, scores, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores)
    n = len(y)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boot_stats.append(roc_auc_score(y[idx], scores[idx]))
    if not boot_stats:
        raise RuntimeError("All bootstrap samples single-class")
    return float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def fold_statistics(model, synergy_df):
    meta_result = load_meta_predictions(model, synergy_df)
    if meta_result is None:
        print(f"    no metapredicts for {model}; skipping fold statistics")
        return None
    meta_df, _ = meta_result

    fold_path = os.path.join(CV_FOLDS_DIR, f"{model}_seed42.csv")
    if os.path.exists(fold_path):
        fold_aurocs = pd.read_csv(fold_path)["AUROC"].tolist()
        print(f"    loaded cached fold AUROCs from {os.path.basename(fold_path)}")
    else:
        args = make_model_args(model, synergy_df=synergy_df)
        print(f"    replaying CV folds for {model} (prepare_data)...")
        X_cell, X_drug, Y, dummy = load_training_arrays(args)
        fold_val_indices = get_fold_val_synergy_indices(
            model, args, X_cell, X_drug, Y, dummy
        )
        fold_aurocs = fold_aurocs_from_metapredicts(meta_df, fold_val_indices)
        os.makedirs(CV_FOLDS_DIR, exist_ok=True)
        pd.DataFrame(
            {"fold": range(len(fold_aurocs)), "AUROC": fold_aurocs}
        ).to_csv(fold_path, index=False)

    print(f"    5-fold AUROCs: {[round(x, 4) for x in fold_aurocs]}")
    return fold_aurocs


def evaluate_model(model, synergy_df, n_boot, boot_seed, skip_folds):
    test_result = load_test_predictions(model, synergy_df)
    if test_result is None:
        print(f"  [skip] {model}: no predictions file")
        return None
    test_df, test_path = test_result

    test_auroc = float(roc_auc_score(test_df["actuals"], test_df["pred"]))
    boot_lo, boot_hi = bootstrap_auroc_ci(
        test_df["actuals"].values,
        test_df["pred"].values,
        n_boot=n_boot,
        seed=boot_seed,
    )
    print(f"  {model}: test AUROC={test_auroc:.4f} from {os.path.basename(test_path)}")
    print(f"    bootstrap 95% CI: [{boot_lo:.4f}, {boot_hi:.4f}]")

    row = {
        "model": model,
        "test_auroc": test_auroc,
        "boot_ci_low": boot_lo,
        "boot_ci_high": boot_hi,
        "cv_mean_auroc": np.nan,
        "sd_5fold": np.nan,
        "cv_min": np.nan,
        "cv_max": np.nan,
        "note": "",
    }

    if not skip_folds:
        fold_aurocs = fold_statistics(model, synergy_df)
        if fold_aurocs:
            folds = np.asarray(fold_aurocs, dtype=float)
            row["cv_mean_auroc"] = float(folds.mean())
            row["sd_5fold"] = float(folds.std(ddof=1))
            row["cv_min"] = float(folds.min())
            row["cv_max"] = float(folds.max())

    if model == "TGSynergy":
        row["note"] = "no 5-fold CV replay available (graph model); test-set metrics only"
    return row


def main(cli_args=None):
    args = parse_args(cli_args)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"p13 error bars — synergy_df={args.synergy_df} (no retraining)")
    print(f"predictions read from {PREDICTIONS_DIR}\n")

    rows = []
    for model in MODELS:
        row = evaluate_model(
            model, args.synergy_df, args.n_boot, args.boot_seed,
            args.skip_folds,
        )
        if row:
            rows.append(row)

    df = pd.DataFrame(rows).sort_values("test_auroc", ascending=False)

    print("\n" + "=" * 78)
    print(df.round(4).to_string(index=False))
    if args.write:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Wrote {OUTPUT_CSV}")
    else:
        print(f"\n(shipped table left unchanged: {OUTPUT_CSV}; pass --write to replace)")
    if df.empty:
        raise RuntimeError("No rows produced")
    return df


if __name__ == "__main__":
    main()
