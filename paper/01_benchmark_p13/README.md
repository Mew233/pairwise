# 01 — p13 benchmark

The p13 benchmark holds curated drug-combination/cell-line records from 13 public
screens. Models are trained with an 80/20 split and 5-fold cross-validation on the
training portion, then scored on the held-out set.

## 5-fold cross-validation

| Model | AUROC (mean ± SD) | 95% CI |
|---|---|---|
| PAIRWISE | 0.8520 ± 0.0030 | [0.8494, 0.8547] |
| Multitask-DNN | 0.7812 ± 0.0036 | [0.7780, 0.7843] |
| TranSynergy | 0.7804 ± 0.0031 | [0.7777, 0.7831] |
| DeepDDS | 0.7194 ± 0.0039 | [0.7160, 0.7228] |
| DeepSynergy | 0.7144 ± 0.0037 | [0.7111, 0.7177] |
| MatchMaker | 0.6848 ± 0.0035 | [0.6817, 0.6879] |

`build_manuscript_table1.py` builds `results/predictions_auroc.csv` from the
per-fold AUROCs in `results/cv_folds/<model>_seed42.csv`. Those fold files are
from the out-of-fold `metapredicts_*` predictions by `cv_utils.py`
(KFold, seed 42).



## Held-out test

All models scored on the same held-out split: 60,927 combinations, 17,686 Loewe-positive.
Bootstrap CI: 1000 stratified resamples, seed 0.

| Model | AUROC | 95% CI |
|---|---|---|
| PAIRWISE | 0.8444 | [0.8410, 0.8477] |
| TranSynergy | 0.8279 | [0.8245, 0.8316] |
| Multitask-DNN | 0.7849 | [0.7809, 0.7888] |
| DeepSynergy | 0.7243 | [0.7197, 0.7290] |
| DeepDDS | 0.7206 | [0.7158, 0.7252] |
| MatchMaker | 0.7006 | [0.6959, 0.7054] |

`build_test_auroc_table.py` recomputes from
`predictions/predicts_<model>_p13.txt` into `results/test_auroc_bootstrap.csv`.
`run_all.sh` also trains TGSynergy which is variant of a graph model we self built.


## Run

```bash
# train every model, then rebuild both tables (from the repository root)
bash paper/01_benchmark_p13/run_all.sh
```

Or rebuild the tables from the shipped predictions, without retraining:

```bash
cd paper/01_benchmark_p13
python build_test_auroc_table.py     # held-out AUROC + bootstrap CI
python build_manuscript_table1.py    # Table 1, from results/cv_folds/
```

