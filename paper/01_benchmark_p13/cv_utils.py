"""Replay the train/test and KFold(val) splits used by pipeline.py, without training.

Used to recover per-fold AUROC from the out-of-fold `metapredicts_*` files.
Splits are reproduced by calling the same `prepare_data` the training run used,
so the row ordering that `train_test_split` sees is identical.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split

from pairwise.paths import RESULTS_DIR
from pairwise.pipeline import prepare_data
from pairwise.seeding import DEFAULT_SEED
from pairwise.utilitis import configuration_from_json

CV_SEED = DEFAULT_SEED
K_FOLDS = 5
TEST_SIZE = 0.2


def make_model_args(model, synergy_df="p13", synergy_thres=0):
    args = argparse.Namespace(
        model=model,
        synergy_df=synergy_df,
        synergy_thres=synergy_thres,
        seed=CV_SEED,
        batch_size=256,
        epochs=50,
        train_test_mode="train",
        external_validation=False,
        SHAP_analysis=False,
        ablate="none",
        ri_thres=10,
        cell_df="CCLE",
    )
    config = configuration_from_json(args)
    for key, value in config[model].items():
        if key != "model_name":
            setattr(args, key, value)
    return args


def load_training_arrays(args):
    X_cell, X_drug, Y, _y_ic1, _y_ic2 = prepare_data(args)
    synergy_path = os.path.join(
        RESULTS_DIR, f"processed_synergydf_{args.synergy_df}.csv"
    )
    processed = pd.read_csv(synergy_path, index_col=0)
    dummy = np.array(processed.index)
    return X_cell, X_drug, Y, dummy


def _kfold_on_rows(dummy_trainval, n_rows):
    kfold = KFold(n_splits=K_FOLDS, random_state=CV_SEED, shuffle=True)
    dummy_trainval = np.asarray(dummy_trainval)
    folds = []
    for _, val_ids in kfold.split(np.arange(n_rows)):
        folds.append(dummy_trainval[val_ids].astype(int))
    return folds


def get_fold_val_synergy_indices(model, args, X_cell, X_drug, Y, dummy):
    """Return list of K_FOLDS arrays of synergy indices (validation fold each)."""
    if model == "multitaskdnn_kim":
        splits = train_test_split(
            X_cell,
            X_drug["morgan_fingerprint_1"],
            X_drug["morgan_fingerprint_2"],
            X_drug["drug_target_1"],
            X_drug["drug_target_2"],
            Y,
            dummy,
            test_size=TEST_SIZE,
            random_state=CV_SEED,
        )
        dummy_trainval = splits[12]
        return _kfold_on_rows(dummy_trainval, len(dummy_trainval))

    if model == "deepsynergy_preuer":
        X = np.concatenate([X_cell, X_drug], axis=1)
        _, _, _, _, dummy_trainval, _ = train_test_split(
            X, Y, dummy, test_size=TEST_SIZE, random_state=CV_SEED
        )
        return _kfold_on_rows(dummy_trainval, len(dummy_trainval))

    if model == "matchmaker_brahim":
        text = args.drug_omics[0]
        if args.get_drugfeature_concated:
            text = "drug"
        splits = train_test_split(
            X_cell,
            X_drug[f"{text}_1"],
            X_drug[f"{text}_2"],
            Y,
            dummy,
            test_size=TEST_SIZE,
            random_state=CV_SEED,
        )
        dummy_trainval = splits[8]
        return _kfold_on_rows(dummy_trainval, len(dummy_trainval))

    if model == "deepdds_wang":
        splits = train_test_split(
            X_cell,
            X_drug["smiles2graph_1"],
            X_drug["smiles2graph_2"],
            Y,
            dummy,
            test_size=TEST_SIZE,
            random_state=CV_SEED,
        )
        dummy_trainval = splits[8]
        return _kfold_on_rows(dummy_trainval, len(dummy_trainval))

    if model == "pairwise":
        X, X_fp = [], []
        for d1, d2, cell, fp1, fp2 in zip(
            X_drug["drug_target_rwr_1"],
            X_drug["drug_target_rwr_2"],
            X_cell,
            X_drug["smiles_grover_1"],
            X_drug["smiles_grover_2"],
        ):
            X.append(torch.from_numpy(np.vstack((d1, d2, cell))).float())
            X_fp.append(torch.from_numpy(np.vstack((fp1, fp2))).float())
        _, _, _, _, dummy_trainval, _, _, _ = train_test_split(
            X, Y, dummy, X_fp, test_size=TEST_SIZE, random_state=CV_SEED
        )
        return _kfold_on_rows(dummy_trainval, len(dummy_trainval))

    if model == "transynergy_liu":
        X, X2, C = [], [], []
        for d1, d2, cell, fp1, fp2 in zip(
            X_drug["drug_target_1"],
            X_drug["drug_target_2"],
            X_cell,
            X_drug["smiles_grover_1"],
            X_drug["smiles_grover_2"],
        ):
            X.append(torch.from_numpy(np.vstack((d1, d2))).float())
            X2.append(torch.from_numpy(np.vstack((fp1, fp2))).float())
            C.append(torch.from_numpy(cell).float())
        _, _, _, _, dummy_trainval, _, _, _, _, _ = train_test_split(
            X, Y, dummy, X2, C, test_size=TEST_SIZE, random_state=CV_SEED
        )
        return _kfold_on_rows(dummy_trainval, len(dummy_trainval))

    raise ValueError(f"Unsupported model for fold replay: {model}")


def fold_aurocs_from_metapredicts(meta_df, fold_val_indices, pred_col="pred"):
    from sklearn.metrics import roc_auc_score

    meta = meta_df.copy()
    meta["index"] = meta["index"].astype(int)
    aurocs = []
    for fold, val_idx in enumerate(fold_val_indices):
        val_set = set(val_idx.tolist())
        sub = meta[meta["index"].isin(val_set)]
        if sub["actuals"].nunique() < 2:
            raise ValueError(f"Fold {fold}: single class in metapredicts val set")
        aurocs.append(float(roc_auc_score(sub["actuals"], sub[pred_col])))
    return aurocs
