#!/usr/bin/env python3
"""Validate PAIRWISE patient inference vs mshap_lstaudt + LS PFS head-to-head."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lifelines.exceptions import ConvergenceWarning

from pairwise_patient_inference import (
    MODEL_SPECS,
    PairwisePatientPredictor,
    load_mshap_panel,
    map_drug_names,
)
from mshap_survival_common import run_survival_on_mshap
from run_heldout_survival import DATA, load_data

warnings.simplefilter("ignore", ConvergenceWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
SURV_OUT = os.path.realpath(os.path.join(HERE, "..", "results", "survival"))
GOLD = os.path.join(HERE, "cooclassifier-main", "mshap_lstaudt.csv")


def validate_model(model_key: str, max_samples: int | None = None, max_drugs: int | None = None):
    spec = MODEL_SPECS[model_key]
    gold = pd.read_csv(GOLD, index_col=0)
    expr = pd.read_csv(os.path.join(DATA, "orginal_lstaudt_DLBC_exp.csv"), index_col=0)
    expr.index = expr.index.astype(str)
    samples = [c for c in gold.columns if c in expr.columns]
    if max_samples:
        samples = samples[:max_samples]
    panel = load_mshap_panel()
    if max_drugs:
        panel = panel[:max_drugs]
    drug_map = map_drug_names(panel)
    print(f"[{model_key}] samples={len(samples)} drugs mapped={len(drug_map)}/{len(panel)}")

    pred = PairwisePatientPredictor(model_key)
    sub_expr = expr[samples]
    resume = os.path.join(SURV_OUT, f"_resume_validate_{spec.tag}.csv")
    new = pred.predict_matrix(sub_expr, panel, drug_map, resume_path=resume)
    new = new.reindex(columns=samples)

    rows = []
    for drug in new.index:
        if drug not in gold.index:
            continue
        a = gold.loc[drug, samples].astype(float).values
        b = new.loc[drug].astype(float).values
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            continue
        r, p = spearmanr(a[ok], b[ok])
        rows.append(dict(drug=drug, spearman=r, p_value=p, n=int(ok.sum())))
    corr = pd.DataFrame(rows)
    out_path = os.path.join(SURV_OUT, f"validate_mshap_lstaudt_{spec.tag}.csv")
    corr.to_csv(out_path, index=False)
    med = float(corr["spearman"].median()) if len(corr) else float("nan")
    print(f"  median Spearman={med:.3f} over {len(corr)} drugs -> {out_path}")

    _, _, clin = load_data()
    ls_samples = [s for s in new.columns if s in clin.index]
    summary, _, _, _ = run_survival_on_mshap(
        new[ls_samples], clin.loc[ls_samples], time_col="pfs_time", event_col="pfs_event", cohort="LS"
    )
    summary["model"] = spec.tag
    return med, corr, summary


def run():
    os.makedirs(SURV_OUT, exist_ok=True)
    ls_rows = []
    for key in MODEL_SPECS:
        med, _, surv = validate_model(key)
        ls_rows.append(surv)
        if key == "transynergy_liu" and med < 0.85:
            print(f"  WARNING: transynergy_liu Spearman {med:.3f} < 0.85 gate")
    pd.DataFrame(ls_rows).to_csv(os.path.join(SURV_OUT, "ls_pairwise_model_comparison.csv"), index=False)
    print(f"Wrote {SURV_OUT}/ls_pairwise_model_comparison.csv")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_SPECS), default=None)
    p.add_argument("--quick", action="store_true", help="20 samples x 30 drugs smoke test")
    args = p.parse_args()
    if args.model:
        ms = 20 if args.quick else None
        md = 30 if args.quick else None
        validate_model(args.model, max_samples=ms, max_drugs=md)
    else:
        run()
