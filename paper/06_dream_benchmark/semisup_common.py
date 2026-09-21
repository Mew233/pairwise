"""
Unified RACS-style semi-supervised ranking for NCI-DREAM SC2 / OCI-LY3.

Protocol:
  * Bait (y=1 in manifold ranking) = EoB>0 pairs in the TRAIN fold only.
    Test positives are treated as unlabeled during ranking (no label leakage).
  * Stratified 5-fold pair CV (seed=42).
  * Hyperparameter selection maximizes pc-index on TRAIN folds only.
  * Reported pc-index / AUC / TPR are computed on HELD-OUT folds only.

Pipeline per method:
  1. Method-native pair score (zero-shot).
  2. Manifold ranking in [native_score, F1..F7] vs train-set positive baits.
  3. Secondary transcriptomic concordance filter.
"""
from __future__ import annotations

import csv
import os
import re
from itertools import combinations
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

import dream_pcindex as scorer
import drugcomboranker as dcr
import racs_labelfree as racs
import syngen

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_PATH = os.path.join(HERE, "data", "NCI-DREAMSubchallenge2_GoldStandard.txt")

BUDGET_N = 16
SPLIT_SEED = 42
N_HOLDOUT = 13  # random hold-out pairs (78 train), mirrors published RACS ~78-pair eval
N_FOLDS = 5
_DEG_CUTS_8 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]

DEFAULTS: Dict[str, dict] = {
    "syngen": {"timepoint": "24", "deg_cut": 0.5, "alpha": 0.85},
    "racs": {"timepoint": "24", "deg_cut": 0.5, "alpha": 0.85},
    "drugcomboranker": {"timepoint": "24", "deg_cut": 0.5, "n_modules": 8, "alpha": 0.85},
    "pairwise": {"timepoint": "24", "deg_cut": 0.5, "alpha": 0.85},
    "digre": {"fold": 0.6, "timepoint": "24", "deg_cut": 0.5, "alpha": 0.85},
}

PRE_REG: Dict[str, dict] = {
    "syngen": {
        "grid": [{"timepoint": tp, "deg_cut": c, "alpha": 0.85}
                 for tp in ("12", "24") for c in _DEG_CUTS_8],
        "fixed": {"subtract_discordant": True},
    },
    "racs": {
        "grid": [{"timepoint": tp, "deg_cut": c, "alpha": 0.85}
                 for tp in ("12", "24") for c in _DEG_CUTS_8],
        "fixed": {"w_f1": 1.0, "w_f2": 1.0},
    },
    "drugcomboranker": {
        "grid": [{"timepoint": tp, "deg_cut": c, "n_modules": m, "alpha": 0.85}
                 for tp in ("12", "24") for c in (0.3, 0.5)
                 for m in (4, 6, 8, 10)],
        "fixed": {"n_drug_comm": 5, "overlap_coef": 0.5, "cross_coef": 0.25},
    },
    "pairwise": {
        "grid": [{"timepoint": tp, "deg_cut": c, "alpha": 0.85}
                 for tp in ("12", "24") for c in _DEG_CUTS_8],
        "fixed": {},
    },
    "digre": {
        "grid": [{"fold": round(0.2 + 0.1 * i, 2), "timepoint": "24", "deg_cut": 0.5, "alpha": 0.85}
                 for i in range(BUDGET_N)],
        "fixed": {"pathway": "GeneNet", "network": "geneNetLymph"},
    },
}
for _m, _spec in PRE_REG.items():
    assert len(_spec["grid"]) == BUDGET_N, f"{_m} grid != {BUDGET_N}"

# Cached gold for repeated selection loops
_GOLD_CACHE: Tuple[List[str], np.ndarray, np.ndarray, object] | None = None
_BAIT_DICT: Dict[str, bool] | None = None
_SPLIT_CACHE: Tuple[List[str], List[str]] | None = None
_ACTIVE_SPLIT: Tuple[List[str], List[str]] | None = None


def load_gold_labels(path: str = GOLD_PATH) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return (canonical_pairs, negated_eob_sorted, err_sorted, bait_mask).

    bait_mask[i] = True when raw EoB > 0 (synergistic bait). Pairs are in the same order
    as dream_pcindex.read_gold (negated EoB, ascending = most synergistic first).
    """
    drug1, drug2, raw_eob, err = [], [], [], []
    with open(path, encoding="latin-1") as f:
        raw = f.read()
    lines = [ln for ln in re.split(r"[\r\n]+", raw) if ln.strip()]
    for line in lines[1:]:
        p = line.split("\t")
        if len(p) < 4:
            continue
        drug1.append(p[0])
        drug2.append(p[1])
        raw_eob.append(float(p[2]))
        err.append(float(p[3]))
    raw_eob = np.array(raw_eob)
    err = np.array(err)
    neg = raw_eob * -1.0
    idx = np.argsort(neg, kind="stable")
    pairs = [scorer.canon(drug1[i], drug2[i]) for i in idx]
    bait = raw_eob[idx] > 0
    return pairs, neg[idx], err[idx], bait


def eob_dict() -> Dict[str, float]:
    """Map canonical pair key -> raw EoB (higher = more synergistic)."""
    pairs, neg, _, _ = load_gold_labels()
    return {p: float(-neg[i]) for i, p in enumerate(pairs)}


def bait_dict() -> Dict[str, bool]:
    global _BAIT_DICT
    if _BAIT_DICT is None:
        pairs, _, _, bait = load_gold_labels()
        _BAIT_DICT = {p: bool(bait[i]) for i, p in enumerate(pairs)}
    return _BAIT_DICT


def train_test_split() -> Tuple[List[str], List[str]]:
    """Random 78/13 split of the 91 gold pairs (fixed seed). Returns (train, test)."""
    if _ACTIVE_SPLIT is not None:
        return _ACTIVE_SPLIT
    global _SPLIT_CACHE
    if _SPLIT_CACHE is None:
        pairs, _, _, _ = load_gold_labels()
        rng = np.random.default_rng(SPLIT_SEED)
        perm = rng.permutation(len(pairs))
        test = [pairs[i] for i in perm[:N_HOLDOUT]]
        train = [pairs[i] for i in perm[N_HOLDOUT:]]
        _SPLIT_CACHE = (train, test)
    return _SPLIT_CACHE


def set_active_split(train: List[str] | None = None, test: List[str] | None = None) -> None:
    """Set the train/test split used by bait masks and selection; pass None to reset."""
    global _ACTIVE_SPLIT
    _ACTIVE_SPLIT = None if train is None or test is None else (list(train), list(test))


def kfold_splits(n_splits: int = N_FOLDS, seed: int = SPLIT_SEED) -> List[Tuple[List[str], List[str]]]:
    """Stratified pair-level K-fold splits by EoB>0 status."""
    pairs, _, _, bait = load_gold_labels()
    pos = [p for p, b in zip(pairs, bait) if b]
    neg = [p for p, b in zip(pairs, bait) if not b]
    rng = np.random.default_rng(seed)
    pos = [pos[i] for i in rng.permutation(len(pos))]
    neg = [neg[i] for i in rng.permutation(len(neg))]
    pos_folds = np.array_split(np.array(pos, dtype=object), n_splits)
    neg_folds = np.array_split(np.array(neg, dtype=object), n_splits)
    all_pairs = set(pairs)
    splits: List[Tuple[List[str], List[str]]] = []
    for i in range(n_splits):
        test = list(pos_folds[i]) + list(neg_folds[i])
        train = [p for p in pairs if p not in set(test)]
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == all_pairs
        splits.append((train, test))
    return splits


def drugs_in_pair(pair: str) -> Tuple[str, str]:
    """Split a canonical pair key (A_B) into its two drug names."""
    a, b = pair.split("_", 1)
    return a, b


def leave_one_drug_out_splits() -> List[Tuple[str, List[str], List[str]]]:
    """Leave-one-drug-out CV over the 14 DREAM compounds.

    Returns a list of (held_out_drug, train, test) where `test` contains every
    pair that includes the held-out drug, and `train` contains all other pairs.
    A model evaluated on `test` must predict synergy for combinations involving
    a drug it never saw at training time, which is the scientifically honest test
    of generalization on this benchmark.
    """
    pairs, _, _, _ = load_gold_labels()
    drugs = sorted({d for p in pairs for d in drugs_in_pair(p)})
    splits: List[Tuple[str, List[str], List[str]]] = []
    for held in drugs:
        test = [p for p in pairs if held in drugs_in_pair(p)]
        train = [p for p in pairs if p not in set(test)]
        assert set(train).isdisjoint(set(test))
        splits.append((held, train, test))
    return splits


def train_bait_dict() -> Dict[str, bool]:
    """Bait = EoB>0 AND in train split. Test positives are NOT baits (avoid leakage)."""
    baits = bait_dict()
    train, _ = train_test_split()
    train_set = set(train)
    return {p: baits.get(p, False) for p in train_set if baits.get(p, False)}


def bait_mask_for_pairs(pair_keys: List[str]) -> np.ndarray:
    """y=1 only for train-set EoB>0 baits; all other pairs (incl. test positives) = 0."""
    train_baits = train_bait_dict()
    return np.array([1.0 if pair_key in train_baits else 0.0 for pair_key in pair_keys])


def _gold_scoring_context():
    global _GOLD_CACHE
    if _GOLD_CACHE is None:
        pairs, eob, err, _ = load_gold_labels()
        pmat = scorer.probability_matrix(eob, err)
        _GOLD_CACHE = (pairs, eob, err, pmat)
    return _GOLD_CACHE


def zscore(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    sd = s.std()
    return (s - s.mean()) / (sd if sd > 0 else 1.0)


def pair_key(a: str, b: str) -> str:
    return scorer.canon(a, b)


def build_pair_features(timepoint: str = "24", deg_cut: float = 0.5,
                        n_modules: int = 8) -> Tuple[List[Tuple[str, str]], np.ndarray, np.ndarray]:
    """Build 7-dim feature matrix for all 91 unordered pairs.

    Approximate mapping to Sun et al. 2015 network features:
      F1 transcriptomic cosine (MI proxy)
      F2 target-network Jaccard + class (Dis/DCI proxy)
      F3 SynGen concordance (DEG overlap)
      F4 module complementarity (Eff proxy)
      F5 drug signature Pearson correlation
      F6 cross-community bonus
      F7 union module coverage (pathway coverage proxy)
    """
    genes, expr, meta = syngen.load()
    fc = syngen.drug_logfc(genes, expr, meta, timepoint=str(timepoint))
    drugs = list(fc.columns)
    sig = racs.core_signatures(timepoint=str(timepoint), deg_cut=deg_cut)

    _, drug_comm, cov = dcr.build(timepoint=str(timepoint), deg_cut=deg_cut, n_modules=n_modules)
    corr = np.corrcoef(fc.values.T)

    sign = np.sign(fc.values) * (np.abs(fc.values) >= deg_cut)
    core = (np.abs(sign).sum(axis=1) >= 1)
    S = sign[core]

    rows: List[Tuple[str, str]] = []
    feats: List[List[float]] = []
    tx_concord: List[float] = []

    for i, a in enumerate(drugs):
        for j in range(i + 1, len(drugs)):
            b = drugs[j]
            va, vb = S[:, i], S[:, j]
            concord = int(np.sum((va != 0) & (va == vb)))
            discord = int(np.sum((va != 0) & (vb != 0) & (va == -vb)))
            syngen_sc = concord - discord

            ca, cb = cov.loc[a].values, cov.loc[b].values
            union_cov = float(np.maximum(ca, cb).sum())
            overlap = float(np.minimum(ca, cb).sum())
            cross = 1.0 if drug_comm[a] != drug_comm[b] else 0.0
            dcr_sc = union_cov - 0.5 * overlap + 0.25 * cross

            rows.append((a, b))
            feats.append([
                racs.f1_transcriptomic(sig, a, b),
                racs.f2_targetnet(a, b),
                float(syngen_sc),
                dcr_sc,
                float(corr[i, j]),
                cross,
                union_cov,
            ])
            tx_concord.append(float(syngen_sc))

    X = np.array(feats, dtype=float)
    X = np.column_stack([zscore(X[:, k]) for k in range(X.shape[1])])
    return rows, X, np.array(tx_concord, dtype=float)


def _affinity_matrix(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Gaussian affinity on z-scored feature rows."""
    n = X.shape[0]
    if n <= 1:
        return np.eye(n)
    dist2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    if sigma is None:
        sigma = float(np.median(dist2[dist2 > 0])) if np.any(dist2 > 0) else 1.0
        sigma = max(sigma, 1e-6)
    W = np.exp(-dist2 / (2.0 * sigma))
    np.fill_diagonal(W, 0.0)
    return W


def manifold_rank(X: np.ndarray, bait_mask: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """Graph-based manifold ranking (Zhou et al. style) with positive baits."""
    n = X.shape[0]
    y = bait_mask.astype(float).copy()
    if y.sum() == 0:
        return np.zeros(n)
    W = _affinity_matrix(X)
    d = W.sum(axis=1)
    d[d == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    S = D_inv_sqrt @ W @ D_inv_sqrt
    M = np.eye(n) - alpha * S
    try:
        f = np.linalg.solve(M, y)
    except np.linalg.LinAlgError:
        f = np.linalg.lstsq(M, y, rcond=None)[0]
    return f


def transcriptomic_filter(
    scores: np.ndarray,
    tx_concord: np.ndarray,
    deg_cut: float = 0.5,
    strength: float = 1.0,
) -> np.ndarray:
    """Secondary filter: demote pairs with low/negative transcriptomic concordance."""
    z_tx = zscore(tx_concord)
    z_sc = zscore(scores)
    # Pairs failing concordance gate get a penalty; concordant pairs keep boost.
    # gate uses deg_cut directly; cap at 1.0 to avoid degenerate all-zero gate at deg_cut=1.2.
    gate = (tx_concord >= min(1.0, deg_cut)).astype(float)
    penalty = (z_tx < 0).astype(float) * strength
    return z_sc + 0.35 * z_tx * gate - 0.5 * penalty


def semisup_racs_rank(
    native_df: pd.DataFrame,
    timepoint: str = "24",
    deg_cut: float = 0.5,
    alpha: float = 0.85,
    n_modules: int = 8,
) -> pd.DataFrame:
    """Apply unified RACS semi-sup pipeline to a native-score dataframe (A, B, score)."""
    pairs, X7, tx_concord = build_pair_features(timepoint, deg_cut, n_modules)
    key_to_idx = {pair_key(a, b): i for i, (a, b) in enumerate(pairs)}

    native = []
    order = []
    for r in native_df.itertuples(index=False):
        k = pair_key(r.A, r.B)
        if k not in key_to_idx:
            continue
        order.append((r.A, r.B))
        native.append(float(r.score))
    native = np.array(native, dtype=float)
    idx = [key_to_idx[pair_key(a, b)] for a, b in order]
    X_feat = X7[idx]
    X_combined = np.column_stack([zscore(native), X_feat])

    bait_mask = bait_mask_for_pairs([pair_key(a, b) for a, b in order])

    f = manifold_rank(X_combined, bait_mask, alpha=alpha)
    f = transcriptomic_filter(f, tx_concord[idx], deg_cut=deg_cut)

    out = pd.DataFrame({"A": [a for a, _ in order], "B": [b for _, b in order], "score": f})
    out["Rank"] = out["score"].rank(ascending=False, method="first").astype(int)
    return out


def pc_index_from_df(df: pd.DataFrame, pairs_subset: List[str] | None = None) -> float:
    """pc-index on `pairs_subset` (default: all 91 gold pairs)."""
    gold_pairs, eob, err, pmat = _gold_scoring_context()
    subset = pairs_subset if pairs_subset is not None else gold_pairs
    pred = {}
    for r in df.itertuples(index=False):
        pred[pair_key(r.A, r.B)] = int(r.Rank)
    # gold order restricted to subset, preserving global EoB ranking among subset
    sub_set = set(subset)
    sub_gold = [p for p in gold_pairs if p in sub_set]
    if len(sub_gold) < 2:
        return float("nan")
    idx_map = {p: i for i, p in enumerate(gold_pairs)}
    sub_idx = [idx_map[p] for p in sub_gold]
    ranks = np.array([pred[p] for p in sub_gold], dtype=float)
    # concordance vs EoB order within subset (ranks 1..len(subset) by negated EoB)
    gold_ranks = np.arange(1, len(sub_gold) + 1, dtype=float)
    sub_pmat = pmat[np.ix_(sub_idx, sub_idx)]
    return float(scorer.concordance(ranks, gold_ranks, sub_pmat))


def write_ranking(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = df.sort_values("Rank")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Drug Combination", "Rank"])
        for _, r in d.iterrows():
            w.writerow([f"{r.A} & {r.B}", int(r.Rank)])


def _rank_with_cfg(method: str, native_fn: Callable[..., pd.DataFrame], cfg: dict) -> pd.DataFrame:
    full = dict(cfg, **PRE_REG[method]["fixed"])
    native_kw = {k: full[k] for k in ("fold",) if k in full}
    native = native_fn(**native_kw)
    return semisup_racs_rank(
        native,
        timepoint=str(full.get("timepoint", "24")),
        deg_cut=float(full.get("deg_cut", 0.5)),
        alpha=float(full.get("alpha", 0.85)),
        n_modules=int(full.get("n_modules", 8)),
    )


def select_config(
    method: str,
    native_fn: Callable[..., pd.DataFrame],
    verbose: bool = True,
) -> dict:
    """Select config maximizing pc-index on TRAIN pairs (78), not hold-out."""
    grid = PRE_REG[method]["grid"]
    train, test = train_test_split()
    results = []
    for cfg in grid:
        rank_df = _rank_with_cfg(method, native_fn, cfg)
        pc_train = pc_index_from_df(rank_df, train)
        results.append((pc_train, cfg))
        if verbose:
            print(f"  [{method}] {cfg} -> train pc-index={pc_train:.4f}")
    results.sort(key=lambda r: (-(r[0] if r[0] == r[0] else -1e9)))
    best_pc, best_cfg = results[0]
    if verbose:
        print(f"  [{method}] BUDGET_N={len(grid)} | selected {best_cfg} "
              f"(train pc-index={best_pc:.4f}, hold-out n={len(test)})")
    return {"config": best_cfg, "pc_index_train": best_pc, "n_trials": len(grid),
            "train_pairs": len(train), "test_pairs": len(test)}


def run_semisup(
    method: str,
    native_fn: Callable[..., pd.DataFrame],
    out_name: str,
    verbose: bool = True,
) -> dict:
    """Select on train split; write full 91 ranking; return hold-out metrics."""
    sel = select_config(method, native_fn, verbose=verbose)
    cfg = dict(sel["config"], **PRE_REG[method]["fixed"])
    rank_df = _rank_with_cfg(method, native_fn, sel["config"])
    train, test = train_test_split()
    pc_test = pc_index_from_df(rank_df, test)
    out = os.path.join(HERE, "predictions", out_name)
    write_ranking(rank_df, out)
    if verbose:
        print(f"  [{method}] wrote {out}")
        print(f"  [{method}] hold-out pc-index={pc_test:.4f} (n={len(test)})")
    return {
        "method": method,
        "selection": sel,
        "n_trials": sel["n_trials"],
        "ranking": out,
        "config": cfg,
        "pc_index_test": pc_test,
        "pc_index_train": sel["pc_index_train"],
        "test_pairs": test,
        "train_pairs": train,
    }
