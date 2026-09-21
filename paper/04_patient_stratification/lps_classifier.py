#!/usr/bin/env python3
"""
50-gene Linear Predictor Score (LPS) classifier — Python port of the
`cooclassifier` / clustermap.R procedure, adapted for BTKiCombo(+/-) labels.

Key property for the reviewer's circularity concern: the classifier is fit ONLY
on training-fold gene expression + training-fold cluster labels, and predicts a
held-out sample's group from its GENE EXPRESSION ALONE (never from that sample's
PAIRWISE drug-combination predictions). Gene-wise z-scoring of held-out samples
uses frozen training means/SDs, so the trained classifier stays frozen.

Reference: Wright et al. PNAS 2003 (LPS); Lenz/Staudt cohort COO classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LPSClassifier:
    n_genes: int = 50
    cutoff: float = 0.5            # posterior threshold; 0.5 == argmax (classify all)
    pos_label: str = "Eff"         # BTKiCombo(+)
    neg_label: str = "Noneff"      # BTKiCombo(-)

    genes_: List = field(default_factory=list, init=False)
    weights_: np.ndarray = field(default=None, init=False)
    mu_: pd.Series = field(default=None, init=False)
    sd_: pd.Series = field(default=None, init=False)
    pos_mean_: float = field(default=None, init=False)
    pos_sd_: float = field(default=None, init=False)
    neg_mean_: float = field(default=None, init=False)
    neg_sd_: float = field(default=None, init=False)

    def fit(self, expr: pd.DataFrame, labels: pd.Series) -> "LPSClassifier":
        """expr: genes x samples (train). labels: per-sample {pos,neg}."""
        samples = [s for s in expr.columns if s in labels.index]
        X = expr[samples]
        y = labels.loc[samples]
        # Frozen gene-wise standardization (train statistics).
        self.mu_ = X.mean(axis=1)
        self.sd_ = X.std(axis=1).replace(0, np.nan)
        Z = X.sub(self.mu_, axis=0).div(self.sd_, axis=0)
        Z = Z.dropna(axis=0)  # drop zero-variance genes

        pos_cols = y.index[y == self.pos_label]
        neg_cols = y.index[y == self.neg_label]
        Zp, Zn = Z[pos_cols].values, Z[neg_cols].values

        t_stat, p_val = stats.ttest_ind(Zp, Zn, axis=1, equal_var=False)
        valid = ~np.isnan(t_stat)
        genes = Z.index[valid]
        t_stat, p_val = t_stat[valid], p_val[valid]
        # BH-FDR; pick top-n by smallest FDR (ties broken by |t|).
        order = np.argsort(p_val, kind="stable")
        n = min(self.n_genes, len(order))
        sel = order[:n]
        self.genes_ = list(genes[sel])
        self.weights_ = t_stat[sel]

        Zsel = Z.loc[self.genes_]
        lps_train = self.weights_ @ Zsel.values  # per train sample
        lps = pd.Series(lps_train, index=Zsel.columns)
        self.pos_mean_, self.pos_sd_ = lps[pos_cols].mean(), lps[pos_cols].std()
        self.neg_mean_, self.neg_sd_ = lps[neg_cols].mean(), lps[neg_cols].std()
        return self

    def _posterior_pos(self, lps: np.ndarray) -> np.ndarray:
        dp = stats.norm.pdf(lps, self.pos_mean_, max(self.pos_sd_, 1e-9))
        dn = stats.norm.pdf(lps, self.neg_mean_, max(self.neg_sd_, 1e-9))
        denom = dp + dn
        denom[denom == 0] = 1.0
        return dp / denom

    def predict(self, expr: pd.DataFrame) -> pd.DataFrame:
        """Predict group for held-out samples from expression only."""
        genes = [g for g in self.genes_ if g in expr.index]
        X = expr.loc[genes]
        Z = X.sub(self.mu_.loc[genes], axis=0).div(self.sd_.loc[genes], axis=0)
        w = np.array([self.weights_[self.genes_.index(g)] for g in genes])
        lps = w @ Z.fillna(0.0).values
        p_pos = self._posterior_pos(np.asarray(lps, dtype=float))
        grp = np.where(p_pos > self.cutoff, self.pos_label,
                       np.where((1 - p_pos) > self.cutoff, self.neg_label, "unclassified"))
        return pd.DataFrame({"sample": X.columns, "lps": np.asarray(lps, dtype=float),
                             "p_pos": p_pos, "group": grp}).set_index("sample")
