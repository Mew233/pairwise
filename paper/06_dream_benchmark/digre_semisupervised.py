"""DIGRE semi-supervised adapter."""
from __future__ import annotations

import semisup_common as sc
from digre_native import native_scores


def semisup(verbose: bool = True) -> dict:
    return sc.run_semisup("digre", native_scores, "digre_semisup_ranking.csv", verbose=verbose)
