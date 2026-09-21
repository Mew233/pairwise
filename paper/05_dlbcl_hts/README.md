# 05 — External DLBCL high-throughput screen

Applies models trained on p13, to the Griner et al. DLBCL
combination screen and to TMD8. Because the external screen reports continuous synergy
scores, the same predictions are evaluated both as a ranking  and
after thresholding . 

## Pre-processing

The raw screen data come from Griner et al., ("248 clinically
relevant drug combinations screened in DLBCL"; the `241tripod` : ibrutinib +
241 NCATS compounds in TMD8). `241tripod.zip` is the original archive;
`extracted/241tripod/` holds the unpacked metadata/calc/responses tables, the
per-compound SDFs, `tripod_btk.csv` (PubChem→DrugBank mapping) and the
SynergyFinder scores (`241/TripodDrugSynScores.csv`) used to build Loewe/HSA/ZIP/Bliss
labels. 


## Run

```bash
cd paper/05_dlbcl_hts
python prepare_griner_tmd8.py                       # rebuild data/griner_tmd8_common_eval.csv
python run_griner_tmd8_benchmark.py
```



## Model comparison 

From `metrics/griner_tmd8_metrics_loewe.csv`, n = 215 combinations, 139 positive:

| model | AUROC | AUPRC | precision@10 |
|---|---|---|---|
| pairwise | 0.7279 | 0.6764 | 0.8 |
| RF | 0.5429 | 0.6800 | 0.2 |
| transynergy_liu | 0.5254 | 0.6740 | 0.4 |
| ERT | 0.5164 | 0.6740 | 0.2 |
| deepdds_wang | 0.5147 | 0.6694 | 0.3 |
| multitaskdnn_kim | 0.5122 | 0.6515 | 0.3 |
| matchmaker_brahim | 0.4899 | 0.6584 | 0.3 |
| deepsynergy_preuer | 0.4586 | 0.6582 | 0.3 |

PAIRWISE is the only model above chance on this external cohort; the
baselines sit near 0.5. With n = 215 and CIs of roughly ±0.08 (see the
`auroc_ci_low`/`auroc_ci_high` columns).



