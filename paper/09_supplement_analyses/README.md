# 09 — Supplement analyses

We performed additional analyses to test the robustness of model

## Counterfactual transcriptome permutation

*Does the model use the patient's transcriptome, or just the drug pair?* Predictions
are recomputed with each sample's expression profile replaced by that of another sample
— once from the same tissue, once from a different tissue. If the drug pair alone drove
the prediction, neither swap would change anything.

```bash
cd paper/09_supplement_analyses
python run_permutation.py
python bootstrap_permutation_ci.py     # bootstrap CI on the AUROC drop
python fig_permutation.py
```

On the 60,927-row held-out split (17,686 Loewe-positive), original AUROC 0.8164:

| Scheme | Spearman ρ | AUROC | Δ AUROC | 95% CI |
|---|---|---|---|---|
| same tissue | 0.7660 | 0.7396 | −0.0768 | [−0.0801, −0.0735] |
| cross tissue | 0.6837 | 0.7192 | −0.0972 | [−0.1009, −0.0934] |

Spearman ρ is between the original and swapped prediction vectors;
`results/permutation_summary.csv` holds ρ and the point estimates,
`results/permutation_delta_ci.csv` the bootstrap CIs (2000 resamples).


## Scaffold and cell-lineage splits

*Does performance survive when train and test share no chemical scaffold or no cell
lineage?* The random split lets near-duplicate compounds and sibling cell lines do not compete.

```bash
# order matters: the scaffold arrays must exist before the combined split
python scaffold_split_real.py          # Bemis-Murcko scaffold AND ECFP4 Tanimoto >= TAU
python create_all_splits.py            # original / lineage (GroupKFold) / combined
python train_with_custom_splits.py --split scaffold
python train_with_custom_splits.py --split lineage
python extract_metrics.py
```

`scaffold_split_real.py` writes `splits/scaffold_*`; `create_all_splits.py` writes
`splits/original_*` (random reference), `splits/lineage_*` (whole lineages held out by
`GroupShuffleSplit`/`GroupKFold` on the cell-line lineage) and `splits/combined_*` (the
union of the scaffold and lineage test partitions). `splits/` holds the index arrays
actually used, so the splits are reproducible without regenerating them.
`drug_scaffold_analysis.py` and `cell_lineage_analysis.py` quantify the leakage the
random split allowed.

## PR-AUC bootstrap intervals

*Is AUROC the right metric under class imbalance?* Precision-recall AUC with bootstrap
intervals, on the same predictions.

```bash
python bootstrap_pr_auc_ci.py
```

## Additional baselines

`train_baseline_models.py` fits the scikit-learn baselines (random forest, extremely
randomised trees) on the same custom splits, asking whether a simpler
model would do.
