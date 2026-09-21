# 07 — Ablation and sensitivity

We further ask whether the Loewe threshold drives the result, whether the gene panel is
special, and how the architecture variants compare.

## Modality ablation and architecture variants

Each variant retrains with one input stream removed or replaced:

```bash
# from the repository root
python -m pairwise.main --model pairwise --synergy_df p13 --train_test_mode train \
  --drug_omics smiles_grover            # drop the drug-target/RWR stream
python -m pairwise.main --model pairwise --synergy_df p13 --train_test_mode train \
  --drug_omics drug_target_rwr          # drop the chemical stream
python -m pairwise.main --model pairwise --synergy_df p13 --train_test_mode train \
  --cell_omics mut                      # mutations instead of expression (PAIRWISE-Mut)

# set the full/exp baseline rows from PAIRWISE's own CV folds
python paper/07_ablation_sensitivity/fix_ablation_baseline.py
```

`results/ablation.csv`, all entries 5-fold CV AUROC. The `full` baseline is
PAIRWISE's own CV score, regenerated from
`../01_benchmark_p13/results/cv_folds/pairwise_seed42.csv`.

| Variant | AUROC | Drop vs full |
|---|---|---|
| `full` | 0.8520 | — |
| `exp` (control; expression is the default stream) | 0.8520 | 0.0000 |
| `mut` (mutations instead of expression) | 0.8404 | 0.0116 |
| `ablate_chemical` | 0.8332 | 0.0188 |
| `ablate_dti` | 0.8245 | 0.0276 |
| `ablate_transcriptome` | 0.7796 | 0.0725 |
| `chemical_only` | 0.7780 | 0.0741 |

Swapping mutations for expression costs 0.0116, a few fold-SDs (SD ~0.003), so the
two cell-context representations are close to interchangeable. Dropping the cell
stream entirely costs 0.0725 — the model needs *a* cell-context input more than it
needs the transcriptome specifically.

## Loewe threshold sensitivity

We tend to use differnt  synergy threshold:

```bash
for t in 0 5 10 20; do
  python -m pairwise.main --model pairwise --synergy_df p13 --train_test_mode train \
    --synergy_thres $t --suffix_threshold
done
cd paper/07_ablation_sensitivity
python rebuild_loewe_sensitivity.py     # point estimates from the shipped predictions
python aggregate_loewe_sensitivity.py
Rscript fig_loewe_sensitivity_retrained.R
```

Three tables report the same sweep at increasing detail, all recomputed from
`results/predicts_<model>_p13_original_thres<t>.txt` (60,927 combinations), so
their point estimates are identical by construction:

| File | Adds |
|---|---|
| `results/loewe_sensitivity.csv` | AUROC / AUPRC point estimates |
| `results/loewe_sensitivity_retrained.csv` | + bootstrap CIs |
| `results/loewe_sensitivity_supplementary.csv` | + per-fold AUROCs and spread |

PAIRWISE leads at thresholds 0, 5 and 10 (0.8444 / 0.8985 / 0.9232). Discrimination rises with the threshold for all
models, since the surviving positives are the strongly synergistic pairs.


## Gene panel comparison

Compares the 4,079-gene panel against LINCS landmark genes, the top 2,000 highly
variable genes, and PPI hubs; `results/markerset_paired_bootstrap.csv` holds the paired
bootstrap between panels, which is the right comparison since all panels are scored on
the same combinations. The per-panel predictions are  under
`results/task6_markerset/predicts/` at the repository root.

