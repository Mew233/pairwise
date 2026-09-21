"""
RACS semi-supervised (Sun et al. 2015 style) for NCI-DREAM SC2 / OCI-LY3.

Native score = label-free F1+F2 fusion at default config; unified semi-sup layer
(manifold ranking vs EoB>0 baits + transcriptomic filter) applied on all 91 pairs.
"""
from __future__ import annotations

import os

import semisup_common as sc
import racs_labelfree as racs

_HERE = os.path.dirname(os.path.abspath(__file__))


def native_scores(**kwargs) -> "pd.DataFrame":
    import pandas as pd

    cfg = sc.DEFAULTS["racs"]
    df = racs.racs_labelfree(timepoint=str(cfg["timepoint"]), deg_cut=float(cfg["deg_cut"]))
    return df[["A", "B", "score"]].copy()


def semisup(verbose: bool = True) -> dict:
    return sc.run_semisup("racs", native_scores, "racs_semisup_ranking.csv", verbose=verbose)


if __name__ == "__main__":
    semisup()
