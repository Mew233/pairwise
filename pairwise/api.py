"""High-level API for PAIRWISE

This module provides ``loadpairwise()`` and a ``Trainer`` class so that the
example notebook and downstream scripts can use a clean three-method workflow:

    model = loadpairwise('checkpoint.pth', map_location='cpu')
    pred, df_pred = model.predict(features)
    model.extract(features)
"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .get_model import get_model
from .seeding import seed_everything, DEFAULT_SEED


def loadpairwise(checkpoint_path, map_location="cpu", model_name="pairwise"):
    """Load a trained PAIRWISE model from a checkpoint.

    Mirrors ``compass.loadcompass``.  Returns a ``Pairwise`` model with weights
    loaded, ready for ``.predict()`` and ``.extract()`` calls.

    Parameters
    ----------
    checkpoint_path : str
        Path to a ``.pth`` file (PyTorch ``state_dict``).  Can be a relative
        path under ``PAIRWISE_WEIGHTS_DIR`` or an absolute path.
    map_location : str or torch.device
        Device to map the weights to.  Use ``'cpu'`` if no GPU is available.
    model_name : str
        Model architecture to instantiate (default ``'pairwise'``).

    Returns
    -------
    Pairwise
        Model instance with loaded weights, on the specified device.
    """
    model = get_model(model_name)

    # Resolve checkpoint path
    if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
        from .paths import WEIGHTS_DIR
        candidate = os.path.join(WEIGHTS_DIR, checkpoint_path)
        if os.path.exists(candidate):
            checkpoint_path = candidate

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Download from Zenodo or run training first."
        )

    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model = model.to(map_location)
    model.eval()
    return model


class Trainer:
    """High-level training wrapper for PAIRWISE.

    Mirrors the COMPASS ``PreTrainer`` pattern: configure hyperparameters at
    construction time, then call ``.train()`` to run the full 5-fold
    cross-validation training loop.

    Parameters
    ----------
    model_name : str
        Architecture to train (default ``'pairwise'``).
    synergy_df : str
        Benchmark dataset (``'p13'`` or ``'Customized'``).
    lr : float
        Learning rate for Adam (default ``1e-4``, matching the manuscript).
    batch_size : int
        Mini-batch size (default 256).
    epochs : int
        Maximum number of epochs per fold (default 50).
    cv_folds : int
        Number of cross-validation folds (default 5).
    seed : int
        Random seed controlling all stochasticity (default 42).
    device : str or torch.device
        Compute device.  Defaults to CUDA if available, else CPU.
    work_dir : str
        Working directory holding ``data/``, ``results/``, ``weights/``.

    Example
    -------
    >>> trainer = Trainer(lr=1e-4, epochs=50)
    >>> model, checkpoint = trainer.train()
    >>> val_results = trainer.evaluate(model, checkpoint)
    """

    def __init__(
        self,
        model_name="pairwise",
        synergy_df="p13",
        lr=1e-4,
        weight_decay=0.0,
        batch_size=256,
        epochs=50,
        cv_folds=5,
        seed=42,
        device=None,
        gpu=0,
    ):
        self.model_name = model_name
        self.synergy_df = synergy_df
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.cv_folds = cv_folds
        self.seed = seed
        self.gpu = gpu

        if device is None:
            self.device = torch.device(
                f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        seed_everything(seed)

    def _make_args(self, train_test_mode="train"):
        """Build an argparse.Namespace compatible with the pipeline."""
        import argparse
        from .utilitis import configuration_from_json

        args = argparse.Namespace(
            model=self.model_name,
            synergy_df=self.synergy_df,
            synergy_thres=0,
            ri_thres=10,
            batch_size=self.batch_size,
            epochs=self.epochs,
            cv_folds=self.cv_folds,
            seed=self.seed,
            lr=self.lr,
            weight_decay=self.weight_decay,
            gpu=self.gpu,
            device=self.device,
            train_test_mode=train_test_mode,
            external_validation=False,
            SHAP_analysis=False,
            custom_split=None,
            suffix_threshold=False,
        )
        # Fill feature settings from config JSON
        from .main import apply_model_config
        args = apply_model_config(args)
        return args

    def train(self):
        """Run the full training pipeline.

        Returns
        -------
        model : nn.Module
            The trained model architecture (weights loaded from best fold).
        checkpoint_path : str
            Path to the best checkpoint on disk.
        val_results : dict
            Evaluation metrics on the held-out test set.
        """
        from .pipeline import prepare_data, training, evaluate

        args = self._make_args(train_test_mode="train")

        print(f"PAIRWISE Trainer")
        print(f"  model: {args.model}")
        print(f"  seed: {self.seed}")
        print(f"  device: {self.device}")
        print(f"  lr: {self.lr}  batch_size: {self.batch_size}  epochs: {self.epochs}")
        print(f"  drug_omics: {args.drug_omics}  cell_filtered_by: {args.cell_filtered_by}")

        from .main import write_config
        write_config(args)

        X_cell, X_drug, Y, Y_ic1, Y_ic2 = prepare_data(args)
        print("data loaded")

        if args.model in ["LR", "XGBOOST", "RF", "ERT"]:
            from .pipeline import training_baselines
            model, scores, test_loader, train_val_dataset = training_baselines(
                X_cell, X_drug, Y, args
            )
        else:
            model, checkpoint_path, test_loader, train_val_dataset = training(
                X_cell, X_drug, Y, Y_ic1, Y_ic2, args
            )
            print("training finished")

        val_results = evaluate(model, checkpoint_path, test_loader, train_val_dataset, args)
        print(f"ROCAUC: {round(val_results['AUC'], 4)}, PRAUC: {round(val_results['AUPR'], 4)}")

        return model, checkpoint_path, val_results

    def evaluate(self, model, checkpoint_path):
        """Evaluate a trained model on the held-out test set."""
        from .pipeline import prepare_data, evaluate as _evaluate

        args = self._make_args(train_test_mode="test")
        from .main import write_config
        write_config(args)

        X_cell, X_drug, Y, Y_ic1, Y_ic2 = prepare_data(args)
        _, test_loader, train_val_dataset = None, None, None

        # Re-derive test loader via the pipeline
        if args.model in ["combonet", "pairwise"]:
            drug_feat = args.drug_omics[0]
            from .seeding import get_seed
            from sklearn.model_selection import train_test_split
            X, X2, X3 = [], [], []
            for d1, d2, cell, fp1, fp2 in zip(
                X_drug[f"{drug_feat}_1"], X_drug[f"{drug_feat}_2"],
                X_cell, X_drug["smiles_grover_1"], X_drug["smiles_grover_2"],
            ):
                import numpy as np
                t = torch.from_numpy(np.vstack((d1, d2))).float()
                t2 = torch.from_numpy(np.vstack((fp1, fp2))).float()
                t3 = torch.from_numpy(np.array(cell)).float()
                X.append(t)
                X2.append(t2)
                X3.append(t3)
            from .paths import RESULTS_DIR
            import pandas as pd
            synergy = pd.read_csv(
                os.path.join(RESULTS_DIR, f"processed_synergydf_{args.synergy_df}.csv"),
                index_col=0,
            )
            dummy = np.array(synergy.index)
            test_size = 0.2
            _, _, _, _, _, _, X2_trainval, X2_test, _, X3_test = train_test_split(
                X, Y, dummy, X2, X3, test_size=test_size, random_state=get_seed(args)
            )
            # Build test loader
            from .dataloader import evaluator_graph_combonet
            # For pairwise, the test set is the second element of the split
            # We need the full split to get test indices
            splits = train_test_split(
                X, Y, dummy, X2, X3,
                test_size=test_size, random_state=get_seed(args),
            )
            X_test = splits[1]
            Y_test = splits[3]
            dummy_test = splits[5]
            X2_test = splits[7]
            X3_test = splits[9]

            # Build temp_loader_test format
            temp_loader_test = [X_test, Y_test.tolist(), dummy_test.tolist(), X2_test, X3_test]
            val_results_raw = evaluator_graph_combonet(
                model, checkpoint_path, None, temp_loader_test, args
            )
            actuals, predictions = val_results_raw[0], val_results_raw[1]
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = roc_auc_score(y_true=actuals, y_score=predictions)
            ap = average_precision_score(y_true=actuals, y_score=predictions)
            return {"AUC": auc, "AUPR": ap}
        else:
            raise NotImplementedError(
                f"Evaluate not yet implemented for {args.model}. "
                f"Use python -m pairwise.main --model {args.model} --train_test_mode test"
            )
