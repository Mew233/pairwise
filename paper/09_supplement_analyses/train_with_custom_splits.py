"""Train models on the custom splits (scaffold / lineage / combined), with W&B logging.

Reads the index arrays from splits/ (see create_all_splits.py and
scaffold_split_real.py).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

from pairwise.prepare_data import *
from pairwise.pipeline import *
from pairwise.utilitis import configuration_from_json

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed")

SPLITS_DIR = os.path.join(HERE, 'splits')
RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model list based on memory
MODELS = [
    'pairwise',
    'deepsynergy_preuer',
    'matchmaker_brahim',
    'multitaskdnn_kim',
    'deepdds_wang',
    'transynergy_liu',
    'RF',
    'ERT'
]

SPLIT_TYPES = ['original', 'scaffold', 'lineage', 'combined']


def load_split_indices(split_type, fold_id=None):
    """Load train/test or fold indices for a given split type."""
    if fold_id is None:
        train_path = os.path.join(SPLITS_DIR, f'{split_type}_train_indices.npy')
        test_path = os.path.join(SPLITS_DIR, f'{split_type}_test_indices.npy')
        train_idx = np.load(train_path)
        test_idx = np.load(test_path)
        return train_idx, test_idx
    else:
        fold_path = os.path.join(SPLITS_DIR, f'{split_type}_fold_{fold_id}_indices.npy')
        fold_idx = np.load(fold_path)
        return fold_idx


def apply_custom_split_to_data(X_cell, X_drug, Y, Y_ic1, Y_ic2, train_idx, test_idx, model_name):
    """
    Apply custom train/test split to data based on model requirements.
    This mimics the logic in pipeline.py but with custom indices.
    """
    if model_name == 'multitaskdnn_kim':
        return (
            X_cell[train_idx], X_cell[test_idx],
            X_drug['morgan_fingerprint_1'][train_idx], X_drug['morgan_fingerprint_1'][test_idx],
            X_drug['morgan_fingerprint_2'][train_idx], X_drug['morgan_fingerprint_2'][test_idx],
            X_drug['drug_target_1'][train_idx], X_drug['drug_target_1'][test_idx],
            X_drug['drug_target_2'][train_idx], X_drug['drug_target_2'][test_idx],
            Y[train_idx], Y[test_idx]
        )
    elif model_name == 'deepsynergy_preuer':
        X = np.concatenate([X_cell, X_drug], axis=1)
        return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]
    elif model_name in ['RF', 'ERT', 'LR', 'XGBOOST']:
        X = np.concatenate([X_cell, X_drug], axis=1)
        return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]
    else:
        # For graph-based models like pairwise, deepdds, transynergy
        # Return indices for later processing
        return train_idx, test_idx


def train_model_with_custom_split(model_name, split_type, args, use_wandb=True):
    """
    Train a single model with a custom split.
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name} with {split_type} split")
    print(f"{'='*80}")

    # Initialize W&B
    run_name = f"{model_name}_{split_type}"
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                'model': model_name,
                'split_type': split_type,
                **vars(args)
            },
            reinit=True
        )

    # Update args for this model
    args.model = model_name

    # Load configuration
    _configs_ = configuration_from_json(args)
    config = _configs_[model_name]
    for key, value in config.items():
        if key != 'model_name' and not hasattr(args, key):
            setattr(args, key, value)

    # Prepare data
    print("Loading data...")
    X_cell, X_drug, Y, Y_ic1, Y_ic2 = prepare_data(args)

    # Load custom split indices
    train_idx, test_idx = load_split_indices(split_type)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Train based on model type
    if model_name in ['RF', 'ERT', 'LR', 'XGBOOST']:
        X = np.concatenate([X_cell, X_drug], axis=1)
        X_trainval, X_test = X[train_idx], X[test_idx]
        Y_trainval, Y_test = Y[train_idx], Y[test_idx]

        from sklearn.model_selection import KFold
        from pairwise.get_model import get_model
        import joblib

        model = get_model(model_name)
        kfold = KFold(n_splits=5, random_state=42, shuffle=True)

        cv_results = cross_validate(model, X_trainval, Y_trainval, cv=kfold,
                                    scoring='roc_auc', return_estimator=True)
        scores = cv_results['test_score']
        best_model = cv_results['estimator'][np.argmax(scores)]

        # Save model
        model_path = os.path.join(RESULTS_DIR, f'model_{model_name}_{split_type}.pth')
        joblib.dump(best_model, model_path)

        # Evaluate on test set
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
        test_preds = best_model.predict_proba(X_test)[:, 1]
        test_auroc = roc_auc_score(Y_test, test_preds)
        test_auprc = average_precision_score(Y_test, test_preds)

        print(f"\nResults:")
        print(f"  CV AUROC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"  Test AUROC: {test_auroc:.4f}")
        print(f"  Test AUPRC: {test_auprc:.4f}")

        # Log to W&B
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'cv_auroc_mean': np.mean(scores),
                'cv_auroc_std': np.std(scores),
                'test_auroc': test_auroc,
                'test_auprc': test_auprc
            })

        results = {
            'model': model_name,
            'split_type': split_type,
            'cv_auroc_mean': np.mean(scores),
            'cv_auroc_std': np.std(scores),
            'test_auroc': test_auroc,
            'test_auprc': test_auprc
        }

    else:
        # This script only trains the scikit-learn baselines directly. The deep
        # models are trained through pairwise.main; the split indices written to
        # splits/ are what that run consumes.
        print(f"  {model_name} is trained via pairwise.main, not here")
        print(f"  Split indices saved to splits/ for that run")

        results = {
            'model': model_name,
            'split_type': split_type,
            'status': 'splits_created_manual_training_required'
        }

    # Save results
    results_df = pd.DataFrame([results])
    results_path = os.path.join(RESULTS_DIR, f'results_{model_name}_{split_type}.csv')
    results_df.to_csv(results_path, index=False)

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['RF', 'ERT'],
                       help='Models to train')
    parser.add_argument('--split_types', nargs='+', default=SPLIT_TYPES,
                       help='Split types to evaluate')
    parser.add_argument('--wandb_project', type=str, default='pairwise-custom-splits',
                       help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='W&B entity')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use W&B for logging')
    parser.add_argument('--synergy_df', type=str, default='p13')
    parser.add_argument('--cell_df', type=str, default='CCLE')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--synergy_thres', type=int, default=0)
    parser.add_argument('--ri_thres', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_test_mode', type=str, default='train')
    parser.add_argument('--external_validation', type=bool, default=False)
    parser.add_argument('--SHAP_analysis', type=bool, default=False)

    args = parser.parse_args()

    print("="*80)
    print("REBUTTAL TRAINING - CUSTOM SPLITS")
    print("="*80)
    print(f"Models: {args.models}")
    print(f"Split types: {args.split_types}")
    print(f"W&B: {args.use_wandb}")

    # Collect all results
    all_results = []

    for model_name in args.models:
        for split_type in args.split_types:
            try:
                result = train_model_with_custom_split(
                    model_name, split_type, args, use_wandb=args.use_wandb
                )
                all_results.append(result)
            except Exception as e:
                print(f"ERROR training {model_name} with {split_type}: {e}")
                import traceback
                traceback.print_exc()

    # Save combined results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_path = os.path.join(RESULTS_DIR, 'combined_results.csv')
        combined_df.to_csv(combined_path, index=False)
        print(f"\nSaved combined results to {combined_path}")

        # Print summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(combined_df.to_string(index=False))

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
