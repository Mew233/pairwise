#!/usr/bin/env python3
"""Run PAIRWISE and all requested baselines on the common Griner TMD8 set."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import spearmanr

from griner_tmd8_common import (
    DATA_DIR,
    IBRUTINIB_ID,
    METRICS_DIR,
    PRED_DIR,
    ROOT,
    THRESHOLDS,
    build_common_eval,
    cell_vector,
    compute_morgan,
    dti_vector,
    ensure_dirs,
    file_sha256,
    load_feature_context,
    load_p13,
    make_deepdds_graph,
    rwr_vector,
)

from pairwise.get_model import get_model  # noqa: E402
from pairwise.models.transynergy_liu import AE as LegacyAE  # noqa: E402
from pairwise.models.transynergy_liu import Encoder as LegacyEncoder  # noqa: E402
from pairwise.utilitis import OutputFeedForward  # noqa: E402


PRIMARY_MODELS = [
    "pairwise",
    "transynergy_liu",
    "matchmaker_brahim",
    "deepsynergy_preuer",
    "multitaskdnn_kim",
    "deepdds_wang",
    "RF",
    "ERT",
]


def _torch_load(path: str, device):
    return torch.load(path, map_location=device)


def _as_tensor(x, device):
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def _clip_probs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.clip(x, 0.0, 1.0)


def _model_path(model: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", f"best_model_{model}.pth")


def get_current_pairwise_model():
    """Load the source-tree PAIRWISE class matching best_model_pairwise.pth."""
    src_root = os.path.realpath(os.path.join(ROOT, "..", "pairwise", "pairwise"))
    src = os.path.join(src_root, "models", "transynergy_liu.py")
    spec = importlib.util.spec_from_file_location("current_pairwise_transynergy_module", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load PAIRWISE source module from {src}")
    old_path = list(sys.path)
    old_utilitis = sys.modules.pop("utilitis", None)
    try:
        sys.path.insert(0, src_root)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Transynergy_Liu(d_input=3460, d_model=256, n_feature_type=1, N=1, heads=4, dropout=0.2)
    finally:
        sys.path[:] = old_path
        if old_utilitis is not None:
            sys.modules["utilitis"] = old_utilitis


class LegacyTransynergyNoPreload(nn.Module):

    def __init__(self):
        super().__init__()
        self.ae = LegacyAE(4298, 256)
        self.cnn = nn.Sequential()
        self.cnn.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.cnn.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.cnn.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.cnn.fc1 = nn.Linear(29184, 512)
        self.cnn.fc2 = nn.Linear(512, 512)
        self.reduction2 = nn.Linear(3285, 256, bias=True)
        self.atte = LegacyEncoder(256, 1, 4, 0.2)
        self.out = OutputFeedForward(1280, 3, d_layers=[64, 32, 1])

    def _cnn_forward(self, x):
        x = torch.relu(self.cnn.conv1(x))
        x = torch.relu(self.cnn.conv2(x))
        x = torch.relu(self.cnn.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.cnn.fc1(x))
        x = self.cnn.fc2(x)
        return x.view(-1, 2, 256)

    def forward(self, src, fp, cell):
        src = self._cnn_forward(src)
        fp = self.reduction2(fp)
        cell = self.ae.encode(cell).unsqueeze(1)
        x = torch.cat((src, fp, cell), dim=1)
        e = self.atte(x)
        return self.out(e.reshape(e.size(0), -1))


class FeatureBuilder:
    def __init__(self):
        self.ctx = load_feature_context()
        self.cell_all = cell_vector(self.ctx, self.ctx.cell_all_genes)
        self.cell_dti = cell_vector(self.ctx, self.ctx.cell_dti_genes)
        self.cell_var = cell_vector(self.ctx, self.ctx.cell_variance_genes)

    def drug_ids(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return df["drug1_drugbank_id"].astype(int).values, df["drug2_drugbank_id"].astype(int).values

    def dti(self, drug_id: int) -> np.ndarray:
        return dti_vector(self.ctx.dti, int(drug_id))[0]

    def rwr(self, drug_id: int) -> np.ndarray:
        return rwr_vector(self.ctx.rwr, self.ctx.dti, int(drug_id))[0]

    def rwr_pairwise(self, drug_id: int) -> np.ndarray:
        vals = self.rwr(drug_id)
        return pd.Series(vals, index=self.ctx.rwr.index).reindex(self.ctx.cell_dti_genes).fillna(0.0).values.astype(np.float32)

    def grover(self, drug_id: int) -> np.ndarray:
        return self.ctx.grover[int(drug_id)].astype(np.float32)

    def morgan(self, drug_id: int) -> np.ndarray:
        drug_id = int(drug_id)
        fp = self.ctx.drug_bank.get("morgan_fingerprint")
        if hasattr(fp, "columns") and drug_id in fp.columns:
            return fp[drug_id].values.astype(np.float32)
        return compute_morgan(self.ctx.smiles[drug_id])

    def graph(self, drug_id: int):
        drug_id = int(drug_id)
        graphs = self.ctx.drug_bank.get("smiles2graph")
        if isinstance(graphs, dict) and drug_id in graphs and graphs[drug_id] is not None:
            return graphs[drug_id]
        return make_deepdds_graph(self.ctx.smiles[drug_id])

    def arrays_for_external(self, df: pd.DataFrame) -> Dict[str, object]:
        d1, d2 = self.drug_ids(df)
        n = len(df)
        dti1 = np.vstack([self.dti(x) for x in d1]).astype(np.float32)
        dti2 = np.vstack([self.dti(x) for x in d2]).astype(np.float32)
        rwr1 = np.vstack([self.rwr(x) for x in d1]).astype(np.float32)
        rwr2 = np.vstack([self.rwr(x) for x in d2]).astype(np.float32)
        rwr_pairwise1 = np.vstack([self.rwr_pairwise(x) for x in d1]).astype(np.float32)
        rwr_pairwise2 = np.vstack([self.rwr_pairwise(x) for x in d2]).astype(np.float32)
        grover1 = np.vstack([self.grover(x) for x in d1]).astype(np.float32)
        grover2 = np.vstack([self.grover(x) for x in d2]).astype(np.float32)
        morgan1 = np.vstack([self.morgan(x) for x in d1]).astype(np.float32)
        morgan2 = np.vstack([self.morgan(x) for x in d2]).astype(np.float32)
        return {
            "dti1": dti1,
            "dti2": dti2,
            "rwr1": rwr1,
            "rwr2": rwr2,
            "rwr_pairwise1": rwr_pairwise1,
            "rwr_pairwise2": rwr_pairwise2,
            "grover1": grover1,
            "grover2": grover2,
            "morgan1": morgan1,
            "morgan2": morgan2,
            "graph1": [self.graph(x) for x in d1],
            "graph2": [self.graph(x) for x in d2],
            "cell_all": np.tile(self.cell_all, (n, 1)).astype(np.float32),
            "cell_dti": np.tile(self.cell_dti, (n, 1)).astype(np.float32),
            "cell_var": np.tile(self.cell_var, (n, 1)).astype(np.float32),
        }

    def features_for_rf_train(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        print(f"[rf] Building tabular P13 features for {len(df)} rows", flush=True)
        d1 = df["drug1"].astype(int).values
        d2 = df["drug2"].astype(int).values
        cells = df["cell"].astype(str).values
        # RF/ERT use the same tabular representation as DeepSynergy: cell expression + summed Morgan FP.
        drug_cache = {int(d): self.morgan(int(d)) for d in np.unique(np.concatenate([d1, d2]))}
        print(f"[rf] Cached {len(drug_cache)} drug Morgan vectors", flush=True)
        m1 = np.vstack([drug_cache[int(x)] for x in d1]).astype(np.float32)
        m2 = np.vstack([drug_cache[int(x)] for x in d2]).astype(np.float32)
        cell_cache = {}
        for c in np.unique(cells):
            if c not in self.ctx.ccle.index:
                continue
            cell_cache[c] = self.ctx.ccle.loc[c].reindex(self.ctx.cell_all_genes).fillna(0.0).astype(float).values.astype(np.float32)
        print(f"[rf] Cached {len(cell_cache)} cell vectors", flush=True)
        keep = np.array([c in cell_cache for c in cells])
        cell = np.vstack([cell_cache[c] for c in cells[keep]]).astype(np.float32)
        x = np.concatenate([cell, m1[keep] + m2[keep]], axis=1).astype(np.float32)
        y = (df.loc[keep, "score"].values > 0).astype(int)
        return x, y

    def features_for_rf_external(self, f: Dict[str, object]) -> np.ndarray:
        return np.concatenate([f["cell_all"], f["morgan1"] + f["morgan2"]], axis=1).astype(np.float32)


def predict_deep_models(fb: FeatureBuilder, f: Dict[str, object], models: List[str], device) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        if "pairwise" in models:
            model = get_current_pairwise_model().to(device)
            model.load_state_dict(_torch_load(_model_path("pairwise"), device))
            model.eval()
            dts = _as_tensor(np.stack([f["rwr_pairwise1"], f["rwr_pairwise2"]], axis=1), device)
            fps = _as_tensor(np.stack([f["grover1"], f["grover2"]], axis=1), device)
            cell = _as_tensor(f["cell_dti"], device).unsqueeze(1)
            src = torch.cat([dts, cell], dim=1)
            preds["pairwise"] = _clip_probs(model(src, fps).detach().cpu().numpy())

        if "transynergy_liu" in models:
            model = LegacyTransynergyNoPreload().to(device)
            model.load_state_dict(_torch_load(_model_path("transynergy_liu"), device))
            model.eval()
            dts = _as_tensor(np.stack([f["dti1"], f["dti2"]], axis=1), device)
            fps = _as_tensor(np.stack([f["grover1"], f["grover2"]], axis=1), device)
            cell = _as_tensor(f["cell_all"], device)
            preds["transynergy_liu"] = _clip_probs(model(dts, fps, cell).detach().cpu().numpy())

        if "deepsynergy_preuer" in models:
            x = np.concatenate([f["cell_all"], f["morgan1"] + f["morgan2"]], axis=1)
            model = get_model("deepsynergy_preuer", x.shape[1]).to(device)
            model.load_state_dict(_torch_load(_model_path("deepsynergy_preuer"), device))
            model.eval()
            preds["deepsynergy_preuer"] = _clip_probs(model(_as_tensor(x, device)).detach().cpu().numpy())

        if "matchmaker_brahim" in models:
            model = get_model("matchmaker_brahim", f["cell_all"].shape[1], f["morgan1"].shape[1]).to(device)
            model.load_state_dict(_torch_load(_model_path("matchmaker_brahim"), device))
            model.eval()
            preds["matchmaker_brahim"] = _clip_probs(
                model(_as_tensor(f["morgan1"], device), _as_tensor(f["morgan2"], device), _as_tensor(f["cell_all"], device))
                .detach()
                .cpu()
                .numpy()
            )

        if "multitaskdnn_kim" in models:
            model = get_model("multitaskdnn_kim", f["cell_all"].shape[1], f["morgan1"].shape[1], f["dti1"].shape[1]).to(device)
            model.load_state_dict(_torch_load(_model_path("multitaskdnn_kim"), device))
            model.eval()
            preds["multitaskdnn_kim"] = _clip_probs(
                model(
                    _as_tensor(f["morgan1"], device),
                    _as_tensor(f["dti1"], device),
                    _as_tensor(f["morgan2"], device),
                    _as_tensor(f["dti2"], device),
                    _as_tensor(f["cell_all"], device),
                )
                .detach()
                .cpu()
                .numpy()
            )

        if "deepdds_wang" in models:
            from torch_geometric.data import Batch

            model = get_model("deepdds_wang").to(device)
            model.load_state_dict(_torch_load(_model_path("deepdds_wang"), device))
            model.eval()
            b1 = Batch.from_data_list(f["graph1"]).to(device)
            b2 = Batch.from_data_list(f["graph2"]).to(device)
            cell_np = f["cell_var"].astype(np.float32)
            cell_arg = [cell_np[:, j] for j in range(cell_np.shape[1])]
            preds["deepdds_wang"] = _clip_probs(
                model(b1.x, b1.edge_index, b2.x, b2.edge_index, cell_arg, b1.batch, b2.batch).detach().cpu().numpy()
            )
    return preds


def _patch_legacy_sklearn_pkl(model: object) -> object:
    for est in getattr(model, "estimators_", []):
        if not hasattr(est, "monotonic_cst"):
            est.monotonic_cst = None
        value = est.tree_.value
        sums = value.sum(axis=2, keepdims=True)
        if value.size and not np.allclose(sums[sums > 0], 1.0):
            np.divide(value, sums, out=value, where=sums > 0)
    return model


def train_or_load_rf_ert(fb: FeatureBuilder, model_name: str, args) -> object:
    out_path = os.path.join(DATA_DIR, f"model_{model_name}_p13_retrained.pkl")
    if os.path.exists(out_path) and not args.force_retrain_rf:
        with open(out_path, "rb") as f:
            return _patch_legacy_sklearn_pkl(pickle.load(f))
    p13 = load_p13()
    if args.rf_train_max_rows and args.rf_train_max_rows < len(p13):
        p13 = p13.sample(n=args.rf_train_max_rows, random_state=42)
    x, y = fb.features_for_rf_train(p13)
    print(f"[rf] Training {model_name}: X={x.shape}, positives={int(y.sum())}/{len(y)}", flush=True)
    if model_name == "RF":
        model = RandomForestClassifier(n_estimators=args.rf_trees, max_depth=10, random_state=42, n_jobs=args.n_jobs)
    elif model_name == "ERT":
        model = ExtraTreesClassifier(n_estimators=args.rf_trees, max_depth=10, random_state=42, n_jobs=args.n_jobs)
    else:
        raise ValueError(model_name)
    model.fit(x, y)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    return model


def merge_metric_rows(new_rows: List[Dict[str, object]], metrics_path: str, models: List[str], endpoint: str) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(metrics_path):
        old = pd.read_csv(metrics_path)
        if "endpoint" in old.columns:
            old = old[~((old["endpoint"] == endpoint) & (old["model"].isin(models)))]
        else:
            old = old[~old["model"].isin(models)]
        return pd.concat([old, new_df], ignore_index=True)
    return new_df


def merge_predictions(new_pred: pd.DataFrame, pred_path: str, models: List[str]) -> pd.DataFrame:
    if os.path.exists(pred_path):
        old = pd.read_csv(pred_path)
        drop_cols = [f"score_{m}" for m in models if f"score_{m}" in old.columns]
        old = old.drop(columns=drop_cols)
        key_cols = ["block_id", "drug1_name", "drug1_drugbank_id", "drug2_name", "drug2_drugbank_id"]
        score_cols = [f"score_{m}" for m in models if f"score_{m}" in new_pred.columns]
        return old.merge(new_pred[key_cols + score_cols], on=key_cols, how="left")
    return new_pred


def metrics_for_scores(y: np.ndarray, scores: np.ndarray, threshold: int, excess_hsa: np.ndarray) -> Dict[str, float]:
    pred = (scores >= 0.5).astype(int)
    out = {
        "threshold": threshold,
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "spearman_endpoint": float(spearmanr(excess_hsa, scores).correlation),
    }
    for k in [5, 10, 20]:
        top = np.argsort(-scores)[: min(k, len(scores))]
        out[f"precision_at_{k}"] = float(y[top].mean()) if len(top) else float("nan")
    return out


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        ss = scores[idx]
        aucs.append(roc_auc_score(yy, ss))
        aps.append(average_precision_score(yy, ss))
    def ci(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))
    auc_lo, auc_hi = ci(aucs)
    ap_lo, ap_hi = ci(aps)
    return {"auroc_ci_low": auc_lo, "auroc_ci_high": auc_hi, "auprc_ci_low": ap_lo, "auprc_ci_high": ap_hi}


def init_wandb(args, model_name: str, config: Dict[str, object]):
    if not args.use_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"griner-tmd8-{model_name}",
            config=config,
            mode=args.wandb_mode,
            reinit=True,
        )
        return run
    except Exception as e:
        print(f"[wandb] init failed for {model_name}: {e}")
        return None


def log_wandb(run, pred_path: str, metrics: List[Dict[str, object]]) -> None:
    if run is None:
        return
    try:
        import wandb
        for row in metrics:
            run.log({f"hsa_gt_{row['threshold']}/{k}": v for k, v in row.items() if k != "threshold"})
        art = wandb.Artifact(name=f"{run.name}-predictions", type="predictions")
        art.add_file(pred_path)
        run.log_artifact(art)
        run.finish()
    except Exception as e:
        print(f"[wandb] logging failed for {run.name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=PRIMARY_MODELS)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="pairwise-griner-tmd8")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--rf_trees", type=int, default=100)
    parser.add_argument("--rf_train_max_rows", type=int, default=0, help="0 means use all P13 rows")
    parser.add_argument("--force_retrain_rf", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--endpoint", choices=["loewe", "hsa", "zip", "bliss"], default="loewe")
    args = parser.parse_args()

    ensure_dirs()
    common_path = os.path.join(DATA_DIR, "griner_tmd8_common_eval.csv")
    if not os.path.exists(common_path):
        raise FileNotFoundError(f"Missing {common_path}; run prepare_griner_tmd8.py first")
    df = pd.read_csv(common_path)
    if df.empty:
        raise RuntimeError("Common evaluation set is empty")
    print(f"[benchmark] Common evaluation set: {len(df)} rows")

    fb = FeatureBuilder()
    f = fb.arrays_for_external(df)
    device = torch.device(args.device)

    requested = list(args.models)
    deep_models = [m for m in requested if m not in {"RF", "ERT"}]
    all_preds: Dict[str, np.ndarray] = {}
    if deep_models:
        print(f"[benchmark] Predicting deep models: {deep_models}")
        all_preds.update(predict_deep_models(fb, f, deep_models, device))

    if "RF" in requested or "ERT" in requested:
        x_ext = fb.features_for_rf_external(f)
        for model_name in [m for m in ["RF", "ERT"] if m in requested]:
            print(f"[benchmark] Training/loading {model_name} on P13; evaluating on common TMD8 set")
            model = train_or_load_rf_ert(fb, model_name, args)
            all_preds[model_name] = _clip_probs(model.predict_proba(x_ext)[:, 1])

    metric_rows: List[Dict[str, object]] = []
    endpoint_col = f"{args.endpoint}_synergy"
    if endpoint_col not in df.columns:
        raise RuntimeError(f"Endpoint column {endpoint_col} not found in common evaluation data")
    combined_pred = df[
        [
            "block_id",
            "drug1_name",
            "drug1_drugbank_id",
            "drug2_name",
            "drug2_drugbank_id",
            "loewe_synergy",
            "hsa_synergy",
            "zip_synergy",
            "bliss_synergy",
        ]
    ].copy()
    for model_name in requested:
        scores = all_preds[model_name]
        if len(scores) != len(df):
            raise RuntimeError(f"{model_name} produced {len(scores)} predictions for {len(df)} common rows")
        if not np.all(np.isfinite(scores)):
            raise RuntimeError(f"{model_name} produced non-finite predictions")
        if float(np.std(scores)) == 0.0:
            raise RuntimeError(f"{model_name} produced constant predictions")
        pred_df = combined_pred.copy()
        pred_df["score"] = scores
        for th in THRESHOLDS:
            pred_df[f"label_{args.endpoint}_gt_{th}"] = df[f"label_{args.endpoint}_gt_{th}"].astype(int).values
        pred_path = os.path.join(PRED_DIR, f"predictions_{model_name}_{args.endpoint}.csv")
        pred_df.to_csv(pred_path, index=False)
        combined_pred[f"score_{model_name}"] = scores

        rows_this_model = []
        for th in THRESHOLDS:
            y = df[f"label_{args.endpoint}_gt_{th}"].astype(int).values
            row = {
                "model": model_name,
                "endpoint": args.endpoint,
                **metrics_for_scores(y, scores, th, df[endpoint_col].values),
            }
            row.update(bootstrap_ci(y, scores, n_boot=args.bootstrap, seed=42 + th))
            metric_rows.append(row)
            rows_this_model.append(row)

        config = {
            "model": model_name,
            "common_eval_n": int(len(df)),
            "checkpoint": _model_path(model_name) if model_name not in {"RF", "ERT"} else os.path.join(DATA_DIR, f"model_{model_name}_p13_retrained.pkl"),
            "checkpoint_sha256_prefix": file_sha256(_model_path(model_name))[:16] if model_name not in {"RF", "ERT"} else "retrained_p13",
            "external_dataset": "Griner TMD8 DLBCL HTS",
            "labels": f"{args.endpoint} synergy thresholds",
            "primary_endpoint": args.endpoint,
            "griner_labels_used_for_training": False,
            "rf_train_max_rows": args.rf_train_max_rows,
        }
        run = init_wandb(args, model_name, config)
        log_wandb(run, pred_path, rows_this_model)
        print(f"[benchmark] {model_name}: {args.endpoint}>0 AUROC={rows_this_model[0]['auroc']:.4f} AUPRC={rows_this_model[0]['auprc']:.4f}")

    metrics_path = os.path.join(METRICS_DIR, f"griner_tmd8_metrics_{args.endpoint}.csv")
    combined_pred_path = os.path.join(PRED_DIR, f"predictions_all_models_{args.endpoint}.csv")
    combined_pred = merge_predictions(combined_pred, combined_pred_path, requested)
    combined_pred.to_csv(combined_pred_path, index=False)
    metrics = merge_metric_rows(metric_rows, metrics_path, requested, args.endpoint)
    metrics.to_csv(metrics_path, index=False)
    print(f"[benchmark] Combined predictions -> {combined_pred_path}")
    print(f"[benchmark] Metrics -> {metrics_path}")

    # Aggregate W&B run.
    run = init_wandb(args, "aggregate", {"common_eval_n": int(len(df)), "models": requested})
    if run is not None:
        try:
            import wandb
            tbl = wandb.Table(dataframe=metrics)
            run.log({"metrics_table": tbl})
            art = wandb.Artifact(name="griner-tmd8-all-model-predictions", type="predictions")
            art.add_file(combined_pred_path)
            art.add_file(metrics_path)
            run.log_artifact(art)
            run.finish()
        except Exception as e:
            print(f"[wandb] aggregate logging failed: {e}")


if __name__ == "__main__":
    main()

