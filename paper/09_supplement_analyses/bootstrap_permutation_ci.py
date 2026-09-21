"""Bootstrap 95% percentile CIs for the counterfactual delta-AUROC.

For each swap scheme (same-tissue, cross-tissue), the statistic is
    delta_AUROC = AUROC(y, pred_swapped) - AUROC(y, pred_original),
i.e. the change in AUROC when the transcriptome is replaced while the drug
pair is held fixed.

Method: paired stratified bootstrap (B=2000). On each iteration we resample
n_pos positives and n_neg negatives with replacement from the SAME row indices
for original and swapped predictions (so the pairing between pred_original and
pred_swapped is preserved). The CI is the 2.5th / 97.5th percentile of the
bootstrap distribution of delta_AUROC.

Baseline note: `pred_original` here scores 0.8164, not the 0.8444 reported in
01_benchmark_p13. Both are PAIRWISE on the same 60,927-row split with identical
labels and row order, but they come from different inference runs. The delta is
only meaningful if the original and swapped predictions are produced by the same
run, so this analysis uses its own paired baseline rather than the benchmark's.

An earlier version of this table used a baseline with the 60,917 / 17,755
fingerprint — a different model's split — which made the reported deltas
incomparable with every other table in the deposit.

Input:  results/permutation_pairwise.csv  (y, pred_original, pred_same_tissue,
                                             pred_cross_tissue)
Output: results/permutation_delta_ci.csv
        columns: scheme, auroc_original, auroc_swapped, delta_auroc,
                 delta_ci_lo, delta_ci_hi, n_boot, n_pos, n_neg
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
IN_PATH = os.path.join(RES, "permutation_pairwise.csv")
OUT_PATH = os.path.join(RES, "permutation_delta_ci.csv")

N_BOOT = 2000
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5
SEED = 42

SCHEMES = [
    ("same_tissue", "pred_same_tissue"),
    ("cross_tissue", "pred_cross_tissue"),
]


def _auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main() -> None:
    df = pd.read_csv(IN_PATH)
    y = df["y"].to_numpy()
    s_orig = df["pred_original"].to_numpy()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    rng = np.random.default_rng(SEED)

    orig_point = _auc(y, s_orig)
    print(f"n={len(df)}  n_pos={n_pos}  n_neg={n_neg}")
    print(f"AUROC_original = {orig_point:.6f}\n")

    rows = []
    t0 = time.time()
    for scheme, col in SCHEMES:
        s_swap = df[col].to_numpy()
        swap_point = _auc(y, s_swap)
        delta_point = swap_point - orig_point

        boot_delta = np.empty(N_BOOT)
        for b in range(N_BOOT):
            samp_pos = rng.choice(pos_idx, size=n_pos, replace=True)
            samp_neg = rng.choice(neg_idx, size=n_neg, replace=True)
            idx = np.concatenate([samp_pos, samp_neg])
            yb = y[idx]
            boot_delta[b] = _auc(yb, s_swap[idx]) - _auc(yb, s_orig[idx])

        lo = float(np.percentile(boot_delta, CI_LO_PCT))
        hi = float(np.percentile(boot_delta, CI_HI_PCT))
        boot_mean = float(boot_delta.mean())
        boot_se = float(boot_delta.std(ddof=1))
        elapsed = time.time() - t0
        print(
            f"{scheme:<12} AUROC_swap={swap_point:.4f}  "
            f"delta={delta_point:+.4f}  "
            f"95% CI=[{lo:+.4f}, {hi:+.4f}]  "
            f"boot_mean={boot_mean:+.4f}  SE={boot_se:.4f}  "
            f"({elapsed:.1f}s)"
        )
        rows.append({
            "scheme": scheme,
            "auroc_original": round(orig_point, 6),
            "auroc_swapped": round(swap_point, 6),
            "delta_auroc": round(delta_point, 6),
            "delta_ci_lo": round(lo, 6),
            "delta_ci_hi": round(hi, 6),
            "n_boot": N_BOOT,
            "boot_mean": round(boot_mean, 6),
            "boot_se": round(boot_se, 6),
            "n_pos": n_pos,
            "n_neg": n_neg,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
