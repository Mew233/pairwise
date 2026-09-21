#!/usr/bin/env python3
"""Shared survival machinery for PAIRWISE mshap cohort analyses.

Within-cohort Ward clustering -> BTKiCombo(+/-) assignment -> log-rank, KM and
tiered Cox models. Extracted so the LS (Lenz/Staudt) analysis does not depend on
any external-cohort driver script.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.exceptions import ConvergenceWarning

from run_heldout_survival import N_TOP_COMBOS, POS, NEG, fit_cox

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
SURV_OUT = os.path.realpath(os.path.join(HERE, "..", "results", "survival"))


MODEL_SPECS_LS = {
    "unadjusted": ["group"],
    "+age_stage": ["group", "age", "stage"],
    "+IPI": ["group", "age", "stage", "ipi"],
    "+full": ["group", "age", "stage", "ipi", "coo_GCB", "coo_Unclass",
              "sub_MCD", "sub_BN2", "sub_EZB", "sub_N1"],
}

SUCCESS_TIERS = [
    ("A", "+IPI"),
    ("B", "+age_stage"),
    ("C", "unadjusted"),
    ("D", "logrank"),
]

STAGE_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4}

def cluster_within_cohort(mshap: pd.DataFrame, samples) -> pd.Series:
    sub = mshap[samples]
    top = sub.var(axis=1).sort_values(ascending=False).head(N_TOP_COMBOS).index
    feat = sub.loc[top].T.values.astype(float)
    Z = linkage(feat, method="ward")
    lab = fcluster(Z, t=2, criterion="maxclust")
    grp_mean = {g: feat[lab == g].mean() for g in np.unique(lab)}
    pos_cluster = max(grp_mean, key=grp_mean.get)
    labels = pd.Series([POS if g == pos_cluster else NEG for g in lab], index=samples)
    return labels

def _km_rows(df, time_col, event_col):
    rows = []
    for g, name in [(1, "sig_pos"), (0, "sig_neg")]:
        sub = df[df.group == g]
        kmf = KaplanMeierFitter().fit(sub[time_col], sub[event_col])
        et = kmf.event_table
        sf = kmf.survival_function_[kmf.survival_function_.columns[0]]
        for t in sf.index:
            rows.append(dict(
                time=float(t), group=name, survival=float(sf.loc[t]),
                at_risk=int(et.loc[t, "at_risk"]) if t in et.index else np.nan,
            ))
    return pd.DataFrame(rows)

def _logrank(df, time_col, event_col) -> float:
    a, b = df[df.group == 1], df[df.group == 0]
    return float(logrank_test(a[time_col], b[time_col], a[event_col], b[event_col]).p_value)

def _fit_cox_tiers(df, time_col, event_col, model_specs):
    rows = []
    for spec, covs in model_specs.items():
        covs = [c for c in covs if c in df.columns]
        if "group" not in covs:
            continue
        d = df[[time_col, event_col] + covs].dropna()
        if d[event_col].sum() < 5 or d["group"].nunique() < 2:
            continue
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(d, duration_col=time_col, event_col=event_col)
        except Exception:
            continue
        if "group" not in cph.params_.index:
            continue
        s = cph.summary.loc["group"]
        rows.append(dict(
            model_spec=spec, HR=float(s["exp(coef)"]),
            ci_low=float(s["exp(coef) lower 95%"]), ci_high=float(s["exp(coef) upper 95%"]),
            p_value=float(s["p"]), n=len(d),
        ))
    return pd.DataFrame(rows)

def _meets_success(cox: pd.DataFrame, logrank_p: float, model_specs) -> tuple[bool, str]:
    cox = cox.set_index("model_spec") if len(cox) else pd.DataFrame()
    for tier, spec in SUCCESS_TIERS:
        if spec == "logrank":
            if logrank_p < 0.05:
                return True, tier
        elif spec in cox.index and float(cox.loc[spec, "p_value"]) < 0.05:
            if float(cox.loc[spec, "HR"]) < 1.0:
                return True, tier
    return False, ""

def run_survival_on_mshap(
    mshap: pd.DataFrame,
    clin: pd.DataFrame,
    time_col: str,
    event_col: str,
    cohort: str,
) -> dict:
    samples = [s for s in mshap.columns if s in clin.index]
    clin = clin.loc[samples]
    ok = clin[time_col].notna() & clin[event_col].notna() & (clin[time_col] > 0)
    samples = [s for s, v in zip(samples, ok) if v]
    mshap = mshap[samples]
    labels = cluster_within_cohort(mshap, samples)
    df = clin.loc[samples].copy()
    df["group"] = (labels == POS).astype(float)
    specs = MODEL_SPECS_LS
    lp = _logrank(df, time_col, event_col)
    cox = _fit_cox_tiers(df, time_col, event_col, specs)
    ok_succ, tier = _meets_success(cox, lp, specs)
    row = dict(
        cohort=cohort, logrank_p=lp, n=len(df), n_events=int(df[event_col].sum()),
        n_pos=int((df.group == 1).sum()), n_neg=int((df.group == 0).sum()),
        meets_success=ok_succ, success_tier=tier,
    )
    for spec in specs:
        if len(cox) and spec in cox["model_spec"].values:
            r = cox[cox.model_spec == spec].iloc[0]
            row[f"HR_{spec.strip('+')}"] = r.HR
            row[f"p_{spec.strip('+')}"] = r.p_value
        else:
            key = spec.strip("+").replace("+", "_")
            row[f"HR_{key}"] = np.nan
            row[f"p_{key}"] = np.nan
    return row, cox, _km_rows(df, time_col, event_col), df
