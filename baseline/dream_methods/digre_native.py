"""DIGRE native score loader for semi-supervised pipeline.

The native score is the output of the DIGREsyn R package applied end-to-end to
the OCI-LY3 transcriptomic, dose-response and gene-network inputs. It is purely
label-free, so it is safe to use DIGRE as a feature inside the matched-L2R / LODO pipelines.

This module also caches the R toolchain output per `fold` value, so calling
`native_scores(fold=0.6)` repeatedly across folds does not re-run DIGRE.
"""
from __future__ import annotations

import os
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RANK1 = os.path.join(HERE, "data", "Team Predictions", "Rank1.csv")
NATIVE_CACHE = os.path.join(HERE, "results", "digre_native_scores.csv")

# The DIGREsyn R package labels two OCI-LY3 drugs with their full salt forms
# while the NCI-DREAM gold standard uses the short names. Translate so pair
# keys line up with every other baseline / scorer.
_DIGRE_NAME_MAP = {
    "Doxorubicin hydrochloride": "Doxorubicin",
    "H-7, Dihydrochloride": "H-7",
}

# In-process cache: fold -> normalized DataFrame. Avoids re-running the R
# toolchain when the matched-L2R orchestrator calls native_scores() many times.
_R_CACHE: dict[float, pd.DataFrame] = {}


def _normalize_name(x: str) -> str:
    return _DIGRE_NAME_MAP.get(x.strip(), x.strip())


def _from_rank1() -> pd.DataFrame:
    rows = []
    with open(RANK1, encoding="latin-1") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i == 0:
                continue
            combo, rank = line.rsplit(",", 1)
            if " & " not in combo:
                continue
            a, b = combo.split(" & ", 1)
            rows.append({"A": a.strip(), "B": b.strip(), "score": float(92 - int(float(rank)))})
    return pd.DataFrame(rows)


def _read_cached_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["A"] = df["A"].map(_normalize_name)
    df["B"] = df["B"].map(_normalize_name)
    return df


def _from_r(fold: float = 0.6, digresyn_path: str = "/tmp/DIGREsyn",
            force_rerun: bool = False) -> pd.DataFrame | None:
    """Run R to get DIGRE scores at a given fold; return None if unavailable.

    Uses an in-process cache (`_R_CACHE`) keyed by `fold` so we only invoke the
    R toolchain once per fold per Python process.
    """
    if not force_rerun and fold in _R_CACHE:
        return _R_CACHE[fold]

    # First, see whether a cache file on disk already has the right fold. The
    # default digre_native.R writes fold=0.6 to results/digre_native_scores.csv.
    fold_cache = os.path.join(HERE, "results", f"digre_native_scores_fold{fold:.2f}.csv")
    if not force_rerun and os.path.isfile(fold_cache):
        df = _read_cached_csv(fold_cache)
        _R_CACHE[fold] = df
        return df

    rscript = os.path.join(HERE, "digre_native.R")
    if not os.path.isfile(rscript):
        return None
    import shutil
    if not shutil.which("Rscript"):
        return None

    # The R script writes to results/digre_native_scores.csv regardless of fold.
    # To preserve per-fold caches we move the output to a fold-specific name
    # after the R run completes.
    try:
        subprocess.run(
            ["Rscript", rscript, str(fold), digresyn_path],
            cwd=HERE,
            check=True,
            timeout=600,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if not os.path.isfile(NATIVE_CACHE):
        return None
    df = _read_cached_csv(NATIVE_CACHE)
    # Stash a fold-specific copy so future calls skip R entirely.
    try:
        os.makedirs(os.path.dirname(fold_cache), exist_ok=True)
        df.to_csv(fold_cache, index=False)
    except OSError:
        pass
    _R_CACHE[fold] = df
    return df


def native_scores(fold: float = 0.6, **kwargs) -> pd.DataFrame:
    df = _from_r(fold=fold)
    if df is not None:
        return df[["A", "B", "score"]]
    return _from_rank1()
