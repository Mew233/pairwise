#!/usr/bin/env python3
"""Multivariable Cox model for the LS held-out cohort (n=229).

PFS ~ BTKiCombo + age + COO + genetic subtype, on all 229 patients.

IPI is not included as a covariate: it is missing for 37 of the 229 patients, so
adjusting for it would drop the cohort by 16% on a non-random subset. Ann Arbor
stage is missing for 3 and is likewise omitted from this model; the stage-adjusted
variant is reported in the `+age_stage` rows of the by-seed tables (n=226).

Reference levels: COO = ABC, genetic subtype = Other, BTKiCombo = BTKiCombo(-).

Input : results/heldout_patient_metadata.csv
Output: results/ls_cox_n229.csv
"""
import os

import pandas as pd
from lifelines import CoxPHFitter

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(HERE, "results", "heldout_patient_metadata.csv")
OUT_CSV = os.path.join(HERE, "results", "ls_cox_n229.csv")

LABELS = {
    "group": "BTKiCombo(+)",
    "age": "Age (per year)",
    "coo_GCB": "COO: GCB (vs ABC)",
    "coo_Unclass": "COO: Unclassified (vs ABC)",
    "sub_MCD": "Genetic subtype: MCD (vs Other)",
    "sub_BN2": "Genetic subtype: BN2 (vs Other)",
    "sub_EZB": "Genetic subtype: EZB (vs Other)",
    "sub_N1": "Genetic subtype: N1 (vs Other)",
}


def build_design(df):
    X = pd.DataFrame(
        {
            "time": df["pfs_time"].astype(float),
            "event": df["pfs_event"].astype(int),
            "group": df["group"].astype(int),
            "age": df["age"].astype(float),
            "coo_GCB": (df["coo"] == "GCB").astype(int),
            "coo_Unclass": (df["coo"] == "Unclass").astype(int),
        }
    )
    for subtype in ("MCD", "BN2", "EZB", "N1"):
        X[f"sub_{subtype}"] = (df["subtype"] == subtype).astype(int)
    return X.dropna().reset_index(drop=True)


def main() -> None:
    X = build_design(pd.read_csv(IN_CSV))
    summary = CoxPHFitter().fit(X, duration_col="time", event_col="event").summary

    out = pd.DataFrame(
        {
            "cohort": "LS_heldout",
            "outcome": "PFS",
            "variable": summary.index,
            "label": [LABELS[v] for v in summary.index],
            "HR": summary["exp(coef)"].values,
            "ci_low": summary["exp(coef) lower 95%"].values,
            "ci_high": summary["exp(coef) upper 95%"].values,
            "p_value": summary["p"].values,
            "n": len(X),
            "n_events": int(X["event"].sum()),
        }
    )
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
