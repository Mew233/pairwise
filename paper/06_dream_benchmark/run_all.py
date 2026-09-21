#!/usr/bin/env python3
"""
run_all.py — reproduce the DREAM/OCI-LY3 benchmark in one command.

  python run_all.py

What it does:
  1. Validates the official pc-index scorer against archived team submissions.
  2. Runs RACS-style semi-supervised ranking for SynGen, RACS, DrugComboRanker
     at default config.
  3. Scores PAIRWISE semi-sup ranking if predictions/pairwise_semisup_ranking.csv exists,
     else zero-shot pairwise_ranking.csv if present.
"""
import os
import csv

import dream_pcindex as scorer
import semisup_common as sc
import syngen
import racs_labelfree as racs
import drugcomboranker as dcr

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PREDS = os.path.join(HERE, "predictions")


def write_ranking(df, path):
    sc.write_ranking(df, path)


def _default_semisup(mod, method_key, out_name):
    """Quick semi-sup at default config without full grid search."""
    native = mod.native_scores()
    cfg = sc.DEFAULTS[method_key]
    rank_df = sc.semisup_racs_rank(
        native,
        timepoint=str(cfg.get("timepoint", "24")),
        deg_cut=float(cfg.get("deg_cut", 0.5)),
        alpha=float(cfg.get("alpha", 0.85)),
        n_modules=int(cfg.get("n_modules", 8)),
    )
    path = os.path.join(RESULTS, out_name)
    write_ranking(rank_df, path)
    return path


def main():
    import glob
    import re

    print("=" * 64)
    print("STEP 1  Validate official pc-index scorer on archived teams")
    print("=" * 64)
    gold_pairs, eob, err = scorer.read_gold()
    pmat = scorer.probability_matrix(eob, err)
    team_scores = []
    for f in glob.glob(os.path.join(scorer._HERE, "data", "Team Predictions", "Rank*.csv")):
        wci, _ = scorer.score(f, gold_pairs, eob, err, pmat, n_perm=1)
        team_scores.append(wci)
    print(f"  best team pc-index = {max(team_scores):.4f}  (DIGRE winner ~0.6130)")
    print(f"  worst team pc-index = {min(team_scores):.4f}  (~0.42)")

    print("\n" + "=" * 64)
    print("STEP 2  Semi-supervised baselines (default config, RACS-style)")
    print("=" * 64)
    rows = []

    sg_path = _default_semisup(syngen, "syngen", "syngen_semisup_default_ranking.csv")
    pc, pv = scorer.evaluate(sg_path, n_perm=10000)
    rows.append(("SynGen (semi-sup, default cfg)", pc, pv))

    rr_path = _default_semisup(racs, "racs", "racs_semisup_default_ranking.csv")
    pc, pv = scorer.evaluate(rr_path, n_perm=10000)
    rows.append(("RACS (semi-sup, default cfg)", pc, pv))

    dd_path = _default_semisup(dcr, "drugcomboranker", "drugcomboranker_semisup_default_ranking.csv")
    pc, pv = scorer.evaluate(dd_path, n_perm=10000)
    rows.append(("DrugComboRanker (semi-sup, default cfg)", pc, pv))

    digre_path = os.path.join(PREDS, "digre_semisup_ranking.csv")
    if os.path.isfile(digre_path):
        pc, pv = scorer.evaluate(digre_path, n_perm=10000)
        rows.append(("DIGRE (semi-sup, calibrated)", pc, pv))
    else:
        rows.append(("DIGRE (challenge submission)", 0.6130, None))

    print("\n" + "=" * 64)
    print("STEP 3  Score PAIRWISE")
    print("=" * 64)
    for name, fname in [
        ("PAIRWISE (semi-sup, calibrated)", "pairwise_semisup_ranking.csv"),
    ]:
        pw = os.path.join(PREDS, fname)
        if os.path.isfile(pw):
            pc, pv = scorer.evaluate(pw, n_perm=10000)
            rows.append((name, pc, pv))
            print(f"  scored {fname} -> pc={pc:.4f} p={pv:.4f}")
            break
    else:
        print(f"  no PAIRWISE ranking found under {PREDS}; skipping")

    rows.sort(key=lambda r: -r[1])
    print("\n" + "=" * 64)
    print("ASSEMBLED RESULTS (semi-supervised / available rankings)")
    print("=" * 64)
    print(f"{'method':<46}{'pc-index':>9}{'p':>9}")
    for name, pc, pv in rows:
        print(f"{name:<46}{pc:>9.4f}{('' if pv is None else f'{pv:>9.4f}')}")
    print("\nDone.")


if __name__ == "__main__":
    main()
