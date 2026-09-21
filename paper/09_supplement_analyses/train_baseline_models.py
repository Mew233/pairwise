"""Train the scikit-learn baselines (RF, ERT) on the custom splits.

Runs independently of the deep-learning pipeline.
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

from pairwise.paths import RESULTS_DIR as PW_RESULTS

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(HERE, 'splits')
RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_synergy_data():
    """Load processed synergy data."""
    print("Loading synergy data...")
    synergy_path = os.path.join(PW_RESULTS, 'processed_synergydf_p13.csv')
    synergy_df = pd.read_csv(synergy_path, index_col=0)
    return synergy_df

def load_features():
    """Load pre-computed features for baseline models."""
    print("Loading drug and cell features...")

    # Load drug features (target interactions)
    drug_target_path = os.path.join(PW_RESULTS, 'proessed_dpi.csv')
    if os.path.exists(drug_target_path):
        drug_target = pd.read_csv(drug_target_path, index_col=0)
        print(f"  Drug features: {drug_target.shape}")
    else:
        print(f"Drug target file not found at {drug_target_path}")
        return None, None

    # Load cell features (expression) - use CSV instead of npy
    cell_exp_path = os.path.join(ROOT_DIR, 'data', 'cell_line_data', 'CCLE', 'CCLE_exp.csv')
    if os.path.exists(cell_exp_path):
        cell_exp = pd.read_csv(cell_exp_path, index_col=0)
        print(f"  Cell features: {cell_exp.shape} (cells x genes)")
    else:
        print(f"Cell expression file not found at {cell_exp_path}")
        return None, None

    # Convert drug_target columns to int for matching
    drug_target.columns = drug_target.columns.astype(int)
    print(f"  Converted drug IDs to int: {list(drug_target.columns[:5])}")

    return drug_target, cell_exp

def prepare_X_Y(synergy_df, drug_target, cell_exp):
    """Prepare feature matrix X and label vector Y."""
    print("Preparing feature matrix...")

    n_samples = len(synergy_df)

    # Get dimensions
    n_drug_features = drug_target.shape[0]  # protein targets
    n_cell_features = cell_exp.shape[1]  # genes (columns)

    # Initialize feature matrix
    X = np.zeros((n_samples, n_cell_features + 2 * n_drug_features))

    print(f"Feature dimensions: cell={n_cell_features}, drug={n_drug_features}")
    print(f"Total features: {X.shape[1]}")

    # Cell IDs are in the index of cell_exp DataFrame
    cell_index = cell_exp.index.tolist()

    # Track matching stats
    drug1_matched = 0
    drug2_matched = 0
    cell_matched = 0

    # Fill features
    for i, (idx, row) in enumerate(synergy_df.iterrows()):
        # Drug features
        drug1_id = int(row['drug1'])
        drug2_id = int(row['drug2'])

        if drug1_id in drug_target.columns:
            X[i, :n_drug_features] = drug_target[drug1_id].values
            drug1_matched += 1
        if drug2_id in drug_target.columns:
            X[i, n_drug_features:2*n_drug_features] = drug_target[drug2_id].values
            drug2_matched += 1

        # Cell features - cell_exp is (cells x genes), index=cells, columns=genes
        cell_id = row['cell']
        if cell_id in cell_index:
            X[i, 2*n_drug_features:] = cell_exp.loc[cell_id].values
            cell_matched += 1

        if i % 50000 == 0:
            print(f"  Processed {i}/{n_samples} samples")

    # Labels
    Y = (synergy_df['score'] > 0).astype(int).values

    # Print matching statistics
    print(f"\nFeature matching statistics:")
    print(f"  Drug1 matched: {drug1_matched}/{n_samples} ({100*drug1_matched/n_samples:.1f}%)")
    print(f"  Drug2 matched: {drug2_matched}/{n_samples} ({100*drug2_matched/n_samples:.1f}%)")
    print(f"  Cell matched: {cell_matched}/{n_samples} ({100*cell_matched/n_samples:.1f}%)")
    print(f"  Non-zero features: {(X != 0).sum()}/{X.size} ({100*(X != 0).sum()/X.size:.2f}%)")

    print(f"\nFeatures shape: {X.shape}")
    print(f"Labels shape: {Y.shape}")
    print(f"Positive class ratio: {Y.mean():.3f}")

    return X, Y

def train_model(model_name, X_train, Y_train, X_test, Y_test):
    """Train and evaluate a model."""
    print(f"\nTraining {model_name}...")

    # Initialize model
    if model_name == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    elif model_name == 'ERT':
        model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 5-fold cross-validation on training set
    print("  Running 5-fold CV...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_train, Y_train,
        cv=kfold,
        scoring='roc_auc',
        return_estimator=True,
        n_jobs=-1
    )

    cv_scores = cv_results['test_score']
    print(f"  CV AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Get best model
    best_model = cv_results['estimator'][np.argmax(cv_scores)]

    # Evaluate on test set
    print("  Evaluating on test set...")
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    test_auroc = roc_auc_score(Y_test, y_pred_proba)
    test_auprc = average_precision_score(Y_test, y_pred_proba)
    test_acc = accuracy_score(Y_test, y_pred)
    test_prec = precision_score(Y_test, y_pred, zero_division=0)
    test_rec = recall_score(Y_test, y_pred, zero_division=0)

    print(f"  Test AUROC: {test_auroc:.4f}")
    print(f"  Test AUPRC: {test_auprc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    return {
        'cv_auroc_mean': cv_scores.mean(),
        'cv_auroc_std': cv_scores.std(),
        'test_auroc': test_auroc,
        'test_auprc': test_auprc,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec
    }, best_model

def main():
    print("="*80)
    print("REBUTTAL TRAINING - BASELINE MODELS")
    print("="*80)

    # Load data
    synergy_df = load_synergy_data()
    drug_target, cell_exp = load_features()

    if drug_target is None or cell_exp is None:
        print("ERROR: Could not load features")
        return

    # Prepare features
    X, Y = prepare_X_Y(synergy_df, drug_target, cell_exp)

    # Split types and models
    split_types = ['original', 'scaffold', 'lineage', 'combined']
    models = ['RF', 'ERT']

    all_results = []

    for split_type in split_types:
        print(f"\n{'='*80}")
        print(f"SPLIT TYPE: {split_type.upper()}")
        print(f"{'='*80}")

        # Load split indices
        train_idx = np.load(os.path.join(SPLITS_DIR, f'{split_type}_train_indices.npy'))
        test_idx = np.load(os.path.join(SPLITS_DIR, f'{split_type}_test_indices.npy'))

        # Filter out any indices that are out of bounds
        train_idx = train_idx[train_idx < len(X)]
        test_idx = test_idx[test_idx < len(X)]

        print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        for model_name in models:
            try:
                metrics, trained_model = train_model(model_name, X_train, Y_train, X_test, Y_test)

                # Save results
                result = {
                    'model': model_name,
                    'split_type': split_type,
                    **metrics
                }
                all_results.append(result)

                # Save individual result
                result_df = pd.DataFrame([result])
                result_path = os.path.join(RESULTS_DIR, f'results_{model_name}_{split_type}.csv')
                result_df.to_csv(result_path, index=False)

                # Save model
                model_path = os.path.join(RESULTS_DIR, f'model_{model_name}_{split_type}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(trained_model, f)

            except Exception as e:
                print(f"ERROR training {model_name} on {split_type}: {e}")
                import traceback
                traceback.print_exc()

    # Save combined results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_path = os.path.join(RESULTS_DIR, 'combined_results.csv')
        combined_df.to_csv(combined_path, index=False)

        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(combined_df.to_string(index=False))

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()
