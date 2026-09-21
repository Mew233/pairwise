import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.realpath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(ROOT, 'results')
SYNERGY_CSV = os.path.join(RESULTS, 'processed_synergydf_p13.csv')

MODELS = [
    'pairwise',
    'deepsynergy_preuer',
    'matchmaker_brahim',
    'multitaskdnn_kim',
    'deepdds_wang',
    'transynergy_liu',
]
THRESHOLDS = [0, 5, 10, 20]
SPLIT = 'original'


def _bootstrap_metrics(y_true, y_pred, n_boot=200, seed=42, alpha=0.05):
    """Percentile bootstrap CIs for AUROC and AUPRC.

    Resamples the (y_true, y_pred) pairs with replacement n_boot times and
    returns the (1-alpha) percentile interval for each metric. Stratification
    is preserved implicitly by drawing row indices uniformly.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auroc_boot = np.empty(n_boot)
    auprc_boot = np.empty(n_boot)
    pos_mask = y_true == 1
    n_pos = int(pos_mask.sum())
    n_neg = n - n_pos
    for i in range(n_boot):
        # Sample positives and negatives separately so each bootstrap
        # iteration preserves the original class balance (stratified bootstrap).
        pos_idx = rng.choice(np.where(pos_mask)[0], size=n_pos, replace=True)
        neg_idx = rng.choice(np.where(~pos_mask)[0], size=n_neg, replace=True)
        idx = np.concatenate([pos_idx, neg_idx])
        yt = y_true[idx]
        yp = y_pred[idx]
        # Skip degenerate draws where one class is missing
        if yt.sum() == 0 or yt.sum() == len(yt):
            auroc_boot[i] = np.nan
            auprc_boot[i] = np.nan
        else:
            auroc_boot[i] = roc_auc_score(yt, yp)
            auprc_boot[i] = average_precision_score(yt, yp)
    lo = 100 * (alpha / 2)
    hi = 100 * (1 - alpha / 2)
    return (
        float(np.nanpercentile(auroc_boot, lo)),
        float(np.nanpercentile(auroc_boot, hi)),
        float(np.nanpercentile(auprc_boot, lo)),
        float(np.nanpercentile(auprc_boot, hi)),
    )


def evaluate_one(model, threshold, split, synergy_df, n_boot=200):
    """Return metrics + bootstrap CIs for one (model, threshold) cell.

    Reads the retrained model's predictions on its test fold. Labels are
    re-derived from the continuous score at this threshold so that the
    n_pos count is consistent with the binarization rule used during training.
    """
    pred_path = os.path.join(RESULTS, f'predicts_{model}_p13_{split}_thres{threshold}.txt')
    if not os.path.exists(pred_path):
        return None

    pred_df = pd.read_csv(pred_path, index_col=0)
    idx = pred_df['index'].astype(int).values
    scores = synergy_df.loc[idx, 'score'].values
    y_true = (scores > threshold).astype(int)
    y_pred = pred_df[f'predicts_{model}'].values

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        # Degenerate label distribution -> AUROC undefined
        return {
            'model': model,
            'threshold': threshold,
            'AUROC': float('nan'),
            'AUPRC': float('nan'),
            'AUROC_lo': float('nan'),
            'AUROC_hi': float('nan'),
            'AUPRC_lo': float('nan'),
            'AUPRC_hi': float('nan'),
            'n_pos': int(y_true.sum()),
            'n_total': int(len(y_true)),
        }

    auroc_lo, auroc_hi, auprc_lo, auprc_hi = _bootstrap_metrics(
        y_true, y_pred, n_boot=n_boot
    )
    return {
        'model': model,
        'threshold': threshold,
        'AUROC': float(roc_auc_score(y_true, y_pred)),
        'AUPRC': float(average_precision_score(y_true, y_pred)),
        'AUROC_lo': auroc_lo,
        'AUROC_hi': auroc_hi,
        'AUPRC_lo': auprc_lo,
        'AUPRC_hi': auprc_hi,
        'n_pos': int(y_true.sum()),
        'n_total': int(len(y_true)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--thresholds', nargs='+', type=int, default=THRESHOLDS)
    parser.add_argument('--split', type=str, default=SPLIT)
    parser.add_argument('--n_boot', type=int, default=200,
                        help='Number of bootstrap iterations for CIs (default 200)')
    parser.add_argument('--out', type=str,
                        default=os.path.join(RESULTS, 'loewe_sensitivity_retrained.csv'))
    parser.add_argument('--overwrite_legacy', action='store_true',
                        help='Also overwrite results/loewe_sensitivity.csv '
                             '(the legacy post-hoc CSV). Use with care.')
    args = parser.parse_args()

    if not os.path.exists(SYNERGY_CSV):
        sys.exit(f'Missing {SYNERGY_CSV}')

    synergy_df = pd.read_csv(SYNERGY_CSV, index_col=0)
    print(f'Loaded synergy scores: {len(synergy_df)} rows  (n_boot={args.n_boot})')

    rows = []
    missing = []
    for t in args.thresholds:
        for m in args.models:
            r = evaluate_one(m, t, args.split, synergy_df, n_boot=args.n_boot)
            if r is None:
                missing.append(f'{m}_thres{t}')
                continue
            rows.append(r)
            print(f"  {m:24s} thres={t:>2d}  AUROC={r['AUROC']:.4f} "
                  f"[{r['AUROC_lo']:.4f}, {r['AUROC_hi']:.4f}]  "
                  f"AUPRC={r['AUPRC']:.4f}  n_pos={r['n_pos']:>6d}/{r['n_total']}")

    if not rows:
        sys.exit('No prediction files found. Run train_loewe_threshold_sweep.sh first.')

    cols = ['model', 'threshold', 'AUROC', 'AUROC_lo', 'AUROC_hi',
            'AUPRC', 'AUPRC_lo', 'AUPRC_hi', 'n_pos', 'n_total']
    out_df = pd.DataFrame(rows, columns=cols)
    out_df.to_csv(args.out, index=False)
    print(f'\nWrote {len(out_df)} rows -> {args.out}')

    if missing:
        print('\nMissing (run sweep first):')
        for m in missing:
            print(f'  - {m}')

    if args.overwrite_legacy:
        legacy = os.path.join(RESULTS, 'loewe_sensitivity.csv')
        # Back up the legacy post-hoc CSV before overwriting.
        backup = legacy + '.posthoc_backup'
        if not os.path.exists(backup):
            os.rename(legacy, backup)
            print(f'Backed up legacy CSV -> {backup}')
        out_df.to_csv(legacy, index=False)
        print(f'Overwrote {legacy} with retrained values.')


if __name__ == '__main__':
    main()
