# 08 — Explainability (SHAP)

We sought to compute SHAP attributions over PAIRWISE predictions, at two scopes: the single combination
shown in the manuscript (acalabrutinib + taselisib in DLBCL), and a systematic 
over 30 combinations.

## Run

```bash
# single combination, attributions written alongside the predictions
python -m pairwise.main \
  --model pairwise --synergy_df p13 \
  --train_test_mode test --SHAP_analysis True
```

Writes `results/shap_df_<model>_<expected_value>_<dataset>.txt` and the matching
`feat_df_*` table of feature values. SHAP over the deep model is slow — expect hours
for the 30-combination pass on GPU.

## Output

| File | Content |
|---|---|
| `results/shap_modality_share.csv` | Share of total absolute SHAP per input modality |
| `results/shap_30combos_long.csv` | Per-feature attributions for the 30-combination pass |
| `results/shap_top_features.csv` | Genes ranked by mean absolute SHAP |
| `results/pathway_enrichment.csv` | Enrichment over the top-ranked genes |
| `results/pathway_enrichment_by_synergy.csv` | Enrichment split by predicted label |
| `results/combo_pathway_by_drug.csv` | Pathway attribution per combination |
| `results/combo_pathway_modules.csv` | Pathway modules after clustering |
| `results/combo_pathway_clustermap_order.txt` | Row/column order for the clustermap |
| `results/combo_pathway_clustermap_fdr_order.txt` | Same, FDR-filtered |

The appearing modules are B-cell receptor signalling, PI3K-AKT-mTOR, and NF-κB, which
is what the manuscript reports for BTKi combinations.

## Attribution by modality

`results/shap_modality_share.csv` splits total absolute SHAP across the three input
streams:

| Modality | Share |
|---|---|
| Drug-target (DTI) | 59.2% |
| Transcriptome | 21.7% |
| Chemical structure | 19.1% |

The drug-target stream dominates; the transcriptome contributes about a fifth. This
is consistent with the ablation in `paper/07`, where substituting mutation for
expression leaves performance unchanged. 

