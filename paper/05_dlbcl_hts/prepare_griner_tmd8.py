#!/usr/bin/env python3
"""Prepare the common Griner TMD8 external-evaluation dataset."""
from __future__ import annotations

import os

from griner_tmd8_common import (
    DATA_DIR,
    add_training_overlap_and_feature_audit,
    build_common_eval,
    build_griner_block_table,
    ensure_dirs,
    load_feature_context,
    write_dataset_summary,
)


def main() -> None:
    ensure_dirs()
    print("[prepare] Building raw Griner block table")
    blocks = build_griner_block_table()
    raw_path = os.path.join(DATA_DIR, "griner_tmd8_blocks_raw.csv")
    blocks.to_csv(raw_path, index=False)
    print(f"[prepare] Raw blocks: {len(blocks)} -> {raw_path}")

    print("[prepare] Loading feature banks for DTI/RWR/GROVER/Morgan/graph audit")
    ctx = load_feature_context()
    blocks, audit = add_training_overlap_and_feature_audit(blocks, ctx)
    blocks_path = os.path.join(DATA_DIR, "griner_tmd8_blocks.csv")
    audit_path = os.path.join(DATA_DIR, "griner_tmd8_feature_audit.csv")
    blocks.to_csv(blocks_path, index=False)
    audit.to_csv(audit_path, index=False)
    print(f"[prepare] Audited blocks -> {blocks_path}")
    print(f"[prepare] Feature audit -> {audit_path}")

    common = build_common_eval(blocks)
    common_path = os.path.join(DATA_DIR, "griner_tmd8_common_eval.csv")
    common.to_csv(common_path, index=False)
    print(f"[prepare] Common evaluation set: {len(common)} rows -> {common_path}")

    summary = write_dataset_summary(blocks, common, audit)
    print("[prepare] Summary")
    for key in [
        "n_blocks_raw",
        "n_blocks_with_synergyfinder_scores",
        "n_blocks_common_eval",
        "common_fraction_of_raw_blocks_pct",
        "common_row_drugs",
        "row_drugs_seen_in_p13_n",
        "row_drugs_seen_in_p13_pct",
        "pairs_any_drug_seen_in_p13_pct",
        "pairs_both_drugs_seen_in_p13_pct",
        "tmd8_cell_id",
        "tmd8_cell_in_p13",
    ]:
        print(f"  {key}: {summary[key]}")
    for th in [0, 5, 10, 20]:
        print(
            f"  Loewe>{th}: positives={summary[f'loewe_threshold_{th}_positives']} "
            f"rate={summary[f'loewe_threshold_{th}_positive_rate']:.3f}"
        )
    for endpoint in ["hsa", "zip", "bliss"]:
        vals = ", ".join(
            f">{th}: {summary[f'{endpoint}_threshold_{th}_positives']}"
            for th in [0, 5, 10, 20]
        )
        print(f"  Secondary {endpoint.upper()} positives: {vals}")


if __name__ == "__main__":
    main()

