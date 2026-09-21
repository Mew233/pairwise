#!/usr/bin/env bash
# Train and evaluate every benchmarked model on p13 (Fig. 2, Table 1).
#
# Run from the repository root so that data/, results/ and weights/ resolve:
#   bash paper/01_benchmark_p13/run_all.sh
#
# Each model reads its feature configuration from pairwise/configs/config_<model>.json,
# so only the model name varies here.
set -euo pipefail

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
GPU="${GPU:-0}"

MODELS=(
  pairwise
  transynergy_liu
  multitaskdnn_kim
  deepsynergy_preuer
  deepdds_wang
  matchmaker_brahim
  TGSynergy
)

for model in "${MODELS[@]}"; do
  echo "=== training ${model} (seed ${SEED}) ==="
  python -m pairwise.main \
    --model "${model}" \
    --synergy_df p13 \
    --train_test_mode train \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --gpu "${GPU}"
done

echo "=== metrics tables ==="
cd "$(dirname "$0")"
python eval_errorbar_intervals.py --synergy_df p13 --write
python build_test_auroc_table.py
python build_manuscript_table1.py
