"""Patient-level PAIRWISE BTKi combo inference for survival stratification."""
from __future__ import annotations

import copy
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

from pairwise.paths import WEIGHTS_DIR  # noqa: E402
from pairwise.utilitis import Layer, Norm, OutputFeedForward  # noqa: E402

BTKI_DRUG2_ID = 9053
HERE = os.path.dirname(os.path.abspath(__file__))
DRUG_ALIAS_CSV = os.path.join(HERE, "drug_aliases.csv")
MSHAP_PANEL = os.path.join(HERE, "cooclassifier-main", "mshap_lstaudt.csv")


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class _Encoder(nn.Module):
    def __init__(self, d_model, n, heads, dropout):
        super().__init__()
        self.layers = _get_clones(Layer(d_model, heads, dropout), n)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        x = src
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class _AE(nn.Module):
    def __init__(self, input_dim, latent_dim=256, hidden_dims=None, dop=0.1):
        super().__init__()
        hidden_dims = hidden_dims or [512]
        enc = [nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dop))]
        for i in range(len(hidden_dims) - 1):
            enc.append(nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU(), nn.Dropout(dop)))
        enc += [nn.Dropout(dop), nn.Linear(hidden_dims[-1], latent_dim)]
        self.encoder = nn.Sequential(*enc)
        dec = [nn.Sequential(nn.Linear(latent_dim, hidden_dims[-1]), nn.ReLU(), nn.Dropout(dop))]
        hd = hidden_dims.copy()
        hd.reverse()
        for i in range(len(hd) - 1):
            dec.append(nn.Sequential(nn.Linear(hd[i], hd[i + 1]), nn.ReLU(), nn.Dropout(dop)))
        self.decoder = nn.Sequential(*dec)
        self.final_layer = nn.Sequential(
            nn.Linear(hd[-1], hd[-1]), nn.ReLU(), nn.Dropout(dop), nn.Linear(hd[-1], input_dim)
        )

    def encode(self, x):
        return self.encoder(x)


class LegacyPairwiseInfer(nn.Module):
    """best_model_transynergy_liu.pth — drug_target + Grover + AE4298."""

    def __init__(self):
        super().__init__()
        self.ae = _AE(4298, 256)
        self.reduction = nn.Linear(3645, 256)
        self.reduction2 = nn.Linear(3285, 256)
        self.encoder = _Encoder(256, 1, 4, 0.2)
        self.out = OutputFeedForward(1280, 3, d_layers=[64, 32, 1])

    def forward(self, d1_tgt, d2_tgt, d1_fp, d2_fp, cell):
        src = torch.stack([self.reduction(d1_tgt), self.reduction(d2_tgt)], dim=1)
        fp = torch.stack([self.reduction2(d1_fp), self.reduction2(d2_fp)], dim=1)
        cell_e = self.ae.encode(cell).unsqueeze(1)
        x = torch.cat([src, fp, cell_e], dim=1)
        e = self.encoder(x)
        return self.out(e.reshape(e.size(0), -1))


class StackedPairwiseInfer(nn.Module):
    """best_model_pairwise.pth — drug_target_rwr + Grover + AE3460."""

    def __init__(self):
        super().__init__()
        self.ae = _AE(3460, 256)
        self.reduction2 = nn.Linear(3285, 256)
        self.reduction3 = nn.Linear(3460, 256)
        self.encoder = _Encoder(256, 1, 1, 0.2)
        self.encoder2 = _Encoder(256, 1, 4, 0.2)
        self.out = OutputFeedForward(1536, 1, d_layers=[512, 1])

    def forward(self, d1_tgt, d2_tgt, d1_fp, d2_fp, cell):
        fp = torch.stack([self.reduction2(d1_fp), self.reduction2(d2_fp)], dim=1)
        cell_e = self.ae.encode(cell).unsqueeze(1)
        coarse = torch.cat([fp, cell_e], dim=1)
        e1 = self.encoder(coarse)
        flat1 = e1.reshape(e1.size(0), -1)
        gene = torch.stack([self.reduction3(d1_tgt), self.reduction3(d2_tgt), self.reduction3(cell)], dim=1)
        e2 = self.encoder2(gene)
        flat2 = e2.reshape(e2.size(0), -1)
        return self.out(torch.cat([flat1, flat2], dim=1))


@dataclass(frozen=True)
class ModelSpec:
    cli_name: str
    tag: str
    checkpoint: str
    cell_filter: str
    cell_dim: int
    drug_target_key: str
    module_cls: type

    @property
    def weight_path(self) -> str:
        return os.path.join(WEIGHTS_DIR, self.checkpoint)


MODEL_SPECS: Dict[str, ModelSpec] = {
    "transynergy_liu": ModelSpec(
        cli_name="transynergy_liu",
        tag="current_transynergy_liu",
        checkpoint="best_model_transynergy_liu.pth",
        cell_filter="all",
        cell_dim=4298,
        drug_target_key="drug_target",
        module_cls=LegacyPairwiseInfer,
    ),
    "pairwise": ModelSpec(
        cli_name="pairwise",
        tag="current_pairwise",
        checkpoint="best_model_pairwise.pth",
        cell_filter="dti",
        cell_dim=3460,
        drug_target_key="drug_target_rwr",
        module_cls=StackedPairwiseInfer,
    ),
}


def normalize_drug_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"^\(\+\/-\)-", "", s)
    s = re.sub(r"^\(-\)-", "", s)
    s = re.sub(r"\s+(citrate|sodium|phosphate|hydrochloride|mesylate|dimaleate|besylate)$", "", s)
    return s.strip()


def _load_sdf_name_map() -> Dict[str, int]:
    sdf = os.path.join(PAIRWISE_ROOT, "data", "drug_data", "structures.sdf")
    out: Dict[str, int] = {}
    for mol in Chem.SDMolSupplier(sdf, sanitize=False):
        if mol is None:
            continue
        db = int(mol.GetProp("DATABASE_ID").replace("DB", ""))
        for prop in ("GENERIC_NAME", "PREF_NAME", "TRADE_NAME", "COMMON_NAME", "NAME"):
            try:
                out[normalize_drug_name(mol.GetProp(prop))] = db
            except KeyError:
                pass
    return out


def load_drug_aliases() -> Dict[str, int]:
    mapping = _load_sdf_name_map()
    if os.path.exists(DRUG_ALIAS_CSV):
        df = pd.read_csv(DRUG_ALIAS_CSV)
        for _, row in df.iterrows():
            mapping[normalize_drug_name(row["drug_name"])] = int(row["drugbank_id"])
    for k, v in {
        "blebbistatin": 1944, "ethyl apovincaminate": 6874, "6-mercaptopurine": 427,
        "mercaptopurine": 427, "flavopiridol": 534, "vincristine": 541, "ibrutinib": BTKI_DRUG2_ID,
        "sildenafil": 1236, "fluvastatin": 1239, "rucaparib": 12389, "y-27632": 12389,
        "luminespib": 11967, "danusertib": 11967, "dacomitinib": 11967,
    }.items():
        mapping[normalize_drug_name(k)] = v
    return mapping


def map_drug_names(names: Sequence[str]) -> Dict[str, int]:
    aliases = load_drug_aliases()
    out: Dict[str, int] = {}
    for name in names:
        key = normalize_drug_name(name)
        if key in aliases:
            out[name] = aliases[key]
    return out


def load_mshap_panel() -> List[str]:
    full = pd.read_csv(MSHAP_PANEL, index_col=0)
    return list(full.index.astype(str))


def _patch_pairwise_root():
    import pairwise.prepare_data as pd_mod
    import pairwise.select_features as sf_mod

    pd_mod.ROOT_DIR = PAIRWISE_ROOT
    sf_mod.ROOT_DIR = PAIRWISE_ROOT


def get_gene_panel(cell_filter: str) -> List[str]:
    _patch_pairwise_root()
    import pairwise.prepare_data as pd_mod
    import pairwise.select_features as sf_mod

    class _Args:
        train_test_mode = "test"
        cell_omics = ["exp"]

    cf = pd_mod.load_cellline_features("CCLE", _Args())
    _, _, genes = sf_mod.get_cell(cf, ["ACH-000236"], ["exp"], cell_filter, True)
    return [str(g) for g in genes]


def build_patient_matrix(expr: pd.DataFrame, genes: Sequence[str]) -> pd.DataFrame:
    expr = expr.copy()
    expr.index = expr.index.astype(str)
    expr = expr[~expr.index.duplicated(keep="first")]
    med = expr.median(axis=1)
    rows = []
    for g in genes:
        if g in expr.index:
            rows.append(expr.loc[g].astype(float))
        else:
            rows.append(pd.Series(float(med.get(g, 0.0)), index=expr.columns))
    return pd.DataFrame(rows, index=list(genes)).T.fillna(0.0)


class PairwisePatientPredictor:
    def __init__(self, model_key: str, device: Optional[str] = None):
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model {model_key}; choose from {list(MODEL_SPECS)}")
        self.spec = MODEL_SPECS[model_key]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.genes = get_gene_panel(self.spec.cell_filter)
        if len(self.genes) != self.spec.cell_dim:
            raise RuntimeError(f"Gene panel {len(self.genes)} != {self.spec.cell_dim}")
        self.model = self.spec.module_cls()
        sd = torch.load(self.spec.weight_path, map_location="cpu")
        self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()
        self._drug_data = None
        self._grover: Dict[int, np.ndarray] = {}

    def _ensure_drug_data(self):
        if self._drug_data is not None:
            return
        _patch_pairwise_root()
        import pairwise.prepare_data as pd_mod

        self._drug_data = pd_mod.load_drug_features()

    def _drug_target_vec(self, drug_id: int) -> np.ndarray:
        self._ensure_drug_data()
        dt = self._drug_data[self.spec.drug_target_key]
        if drug_id in dt.columns:
            return dt[drug_id].values.astype(np.float32)
        return np.zeros(len(dt.index), dtype=np.float32)

    def _grover_vec(self, drug_id: int) -> np.ndarray:
        if drug_id in self._grover:
            return self._grover[drug_id]
        self._ensure_drug_data()
        g = self._drug_data["smiles_grover"]
        if drug_id in g:
            arr = np.asarray(g[drug_id], dtype=np.float32)
        else:
            arr = np.zeros(3285, dtype=np.float32)
        if arr.size < 3285:
            arr = np.pad(arr, (0, 3285 - arr.size))
        elif arr.size > 3285:
            arr = arr[:3285]
        self._grover[drug_id] = arr
        return arr

    @torch.no_grad()
    def predict_batch(self, drug1_id: int, cell_matrix: np.ndarray, batch_size: int = 128) -> np.ndarray:
        d2 = BTKI_DRUG2_ID
        d1_tgt = self._drug_target_vec(drug1_id)
        d2_tgt = self._drug_target_vec(d2)
        d1_fp = self._grover_vec(drug1_id)
        d2_fp = self._grover_vec(d2)
        n = cell_matrix.shape[0]
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            bs = sl.stop - sl.start
            cell = torch.from_numpy(cell_matrix[sl]).float().to(self.device)
            d1t = torch.from_numpy(np.tile(d1_tgt, (bs, 1))).float().to(self.device)
            d2t = torch.from_numpy(np.tile(d2_tgt, (bs, 1))).float().to(self.device)
            d1f = torch.from_numpy(np.tile(d1_fp, (bs, 1))).float().to(self.device)
            d2f = torch.from_numpy(np.tile(d2_fp, (bs, 1))).float().to(self.device)
            pred = self.model(d1t, d2t, d1f, d2f, cell).detach().cpu().numpy().ravel()
            out[sl] = pred[:bs]
        return out

    def predict_matrix(
        self,
        expr: pd.DataFrame,
        drug_names: Sequence[str],
        drug_map: Dict[str, int],
        resume_path: Optional[str] = None,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        samples = list(expr.columns)
        cell_mat = build_patient_matrix(expr, self.genes).loc[samples].values.astype(np.float32)
        mapped = [(n, drug_map[n]) for n in drug_names if n in drug_map]
        if resume_path and os.path.exists(resume_path):
            out = pd.read_csv(resume_path, index_col=0)
            out = out.reindex(index=[n for n, _ in mapped], columns=samples)
        else:
            out = pd.DataFrame(index=[n for n, _ in mapped], columns=samples, dtype=float)
        for i, (name, did) in enumerate(mapped):
            if out.loc[name].notna().all() and not out.loc[name].isna().any():
                continue
            out.loc[name] = self.predict_batch(did, cell_mat, batch_size=batch_size)
            if resume_path and (i + 1) % 10 == 0:
                out.to_csv(resume_path)
                print(f"  checkpoint {i+1}/{len(mapped)} drugs")
        if resume_path:
            out.to_csv(resume_path)
        return out
