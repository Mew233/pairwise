"""Build the split index arrays used by train_with_custom_splits.py.

Writes into splits/ (positional indices into results/processed_synergydf_p13.csv):

1. original_* - random 80/20 + 5-fold KFold (in-distribution reference)
2. scaffold_* - NOT built here; produced by scaffold_split_real.py
   (Bemis-Murcko scaffold equality AND ECFP4 Tanimoto >= TAU there)
3. lineage_*  - whole cell-line lineages held out via GroupShuffleSplit /
   GroupKFold grouped on tissue_name, so no lineage spans train and test
4. combined_* - union of the scaffold and lineage test partitions
   (most stringent out-of-distribution setting); requires (2) to exist first

Run order: scaffold_split_real.py, then this script.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, KFold, GroupKFold, GroupShuffleSplit
)

from pairwise.paths import RESULTS_DIR as PW_RESULTS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
SPLITS_DIR = os.path.join(HERE, 'splits')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

print("="*80)
print("BUILD SPLIT INDEX ARRAYS (original / lineage / combined)")
print("="*80)

print("\nLoading synergy data...")
synergy_path = os.path.join(PW_RESULTS, 'processed_synergydf_p13.csv')
synergy_df = pd.read_csv(synergy_path, index_col=0)
# IMPORTANT: prepare_data() builds X_cell/X_drug/Y by iterating with iloc[i],
# i.e. by ROW POSITION (0..N-1), NOT by DataFrame index label. The CSV's
# index labels are sparse (0..373823 with gaps from filtering), so we must
# reset_index here so .index.values yields 0..N-1 positional indices that
# align with how X_cell/X_drug/Y are sliced in custom_train_test_split.
synergy_df = synergy_df.reset_index(drop=True)
print(f"Loaded {len(synergy_df)} synergy samples (index range 0..{len(synergy_df)-1})")

n_samples = len(synergy_df)
all_indices = np.arange(n_samples)

# Parameters
TEST_SIZE = 0.2
N_FOLDS = 5
RANDOM_STATE = 42

#=============================================================================
# 1. ORIGINAL RANDOM SPLITS (baseline)
#=============================================================================
print("\n" + "="*80)
print("1. ORIGINAL RANDOM SPLITS (Baseline)")
print("="*80)

train_idx, test_idx = train_test_split(
    all_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

np.save(os.path.join(SPLITS_DIR, 'original_train_indices.npy'), train_idx)
np.save(os.path.join(SPLITS_DIR, 'original_test_indices.npy'), test_idx)
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# K-fold on training set
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
for fold_id, (_, val_idx) in enumerate(kfold.split(train_idx)):
    fold_indices = train_idx[val_idx]
    np.save(os.path.join(SPLITS_DIR, f'original_fold_{fold_id}_indices.npy'), fold_indices)
    print(f"  Fold {fold_id}: {len(fold_indices)} samples")

#=============================================================================
# 2. SCAFFOLD-BASED SPLITS -> see scaffold_split_real.py
#=============================================================================
# The scaffold split is chemistry-based (Bemis-Murcko scaffold equality AND an
# ECFP4 Tanimoto cutoff TAU) and is produced by `scaffold_split_real.py`, which
# writes scaffold_{train,test,fold_*}_indices.npy into the same splits/ directory.
# It is not regenerated here. NOTE: the reported TAU differs between documents
# (manuscript Methods >= 0.4, response letter >= 0.5) - see scaffold_split_real.py.
print("\n" + "="*80)
print("2. SCAFFOLD-BASED SPLITS -> run scaffold_split_real.py")
print("="*80)
print("Skipped: scaffold_*_indices.npy are written by scaffold_split_real.py")

#=============================================================================
# 3. LINEAGE SPLITS (whole lineages held out)
#=============================================================================
print("\n" + "="*80)
print("3. LINEAGE-BASED SPLITS (Leave-Cell-Cluster-Out)")
print("="*80)

# Whole lineages are held out: the grouping variable is the cell line's
# lineage annotation, and GroupKFold guarantees a lineage never appears in
# both the training and the test partition of the same fold.
lineage = synergy_df['tissue_name'].astype(str)
n_tissues = lineage.nunique()

print(f"Total unique cells: {synergy_df['cell'].nunique()}")
print(f"Total lineages: {n_tissues}")

# Held-out test partition: whole lineages, ~TEST_SIZE of the samples.
# GroupShuffleSplit keeps every lineage intact across the train/test boundary.
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_pos, test_pos = next(gss.split(all_indices, groups=lineage))
train_idx = all_indices[train_pos]
test_idx = all_indices[test_pos]

print(f"Train lineages: {lineage.iloc[train_idx].nunique()}, "
      f"Test lineages: {lineage.iloc[test_idx].nunique()}")
print(f"Train samples: {len(train_idx)} ({100*len(train_idx)/n_samples:.1f}%)")
print(f"Test samples: {len(test_idx)} ({100*len(test_idx)/n_samples:.1f}%)")

np.save(os.path.join(SPLITS_DIR, 'lineage_train_indices.npy'), train_idx)
np.save(os.path.join(SPLITS_DIR, 'lineage_test_indices.npy'), test_idx)

# GroupKFold over the training partition, grouped by lineage
gkf = GroupKFold(n_splits=N_FOLDS)
train_lineage = lineage.iloc[train_idx]
for fold_id, (_, val_pos) in enumerate(gkf.split(train_idx, groups=train_lineage)):
    fold_indices = train_idx[val_pos]
    np.save(os.path.join(SPLITS_DIR, f'lineage_fold_{fold_id}_indices.npy'), fold_indices)
    print(f"  Fold {fold_id}: {len(fold_indices)} samples, "
          f"{lineage.iloc[fold_indices].nunique()} lineages")

#=============================================================================
# 4. COMBINED SPLITS (scaffold test partition UNION lineage test partition)
#=============================================================================
print("\n" + "="*80)
print("4. COMBINED SPLITS (Leave-Drug-AND-Cell-Cluster-Out)")
print("="*80)

# Test set: samples held out by EITHER the scaffold split or the lineage split.
# The scaffold indices come from scaffold_split_real.py (chemistry-based).
scaffold_test_path = os.path.join(SPLITS_DIR, 'scaffold_test_indices.npy')
if not os.path.exists(scaffold_test_path):
    raise FileNotFoundError(
        f"{scaffold_test_path} not found - run scaffold_split_real.py first, "
        "it produces the chemistry-based scaffold split used by the combined split."
    )
scaffold_test_idx = np.load(scaffold_test_path)
lineage_test_idx = test_idx  # from section 3

test_idx = np.unique(np.concatenate([scaffold_test_idx, lineage_test_idx]))
train_mask = np.ones(n_samples, dtype=bool)
train_mask[test_idx] = False
train_idx = all_indices[train_mask]

print(f"Train samples: {len(train_idx)} ({100*len(train_idx)/n_samples:.1f}%)")
print(f"Test samples: {len(test_idx)} ({100*len(test_idx)/n_samples:.1f}%)")

np.save(os.path.join(SPLITS_DIR, 'combined_train_indices.npy'), train_idx)
np.save(os.path.join(SPLITS_DIR, 'combined_test_indices.npy'), test_idx)

# K-fold on training set (random shuffle)
train_idx_shuffled = train_idx.copy()
np.random.shuffle(train_idx_shuffled)
samples_per_fold = len(train_idx) // N_FOLDS

for fold_id in range(N_FOLDS):
    fold_indices = train_idx_shuffled[fold_id * samples_per_fold:(fold_id + 1) * samples_per_fold]
    np.save(os.path.join(SPLITS_DIR, f'combined_fold_{fold_id}_indices.npy'), fold_indices)
    print(f"  Fold {fold_id}: {len(fold_indices)} samples")

print("\n" + "="*80)
print("ALL SPLITS CREATED SUCCESSFULLY")
print(f"Splits saved to: {SPLITS_DIR}")
print("="*80)
