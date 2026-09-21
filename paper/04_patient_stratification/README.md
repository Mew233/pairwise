# 04 — Patient stratification and survival

Applies PAIRWISE to bulk tumour transcriptomes to split DLBCL patients into predicted
responders and non-responders to BTKi combination therapy, and evaluates their survival
outcomes. Cohort: the Lenz/Staudt DLBCL series, n = 229.

**Note.** This cohort was treated with R-CHOP, not with a BTKi combination. A survival
difference between the predicted groups therefore shows that the predicted-response
signature captures prognostic transcriptional variation. 

## Run

```bash
cd paper/04_patient_stratification

# score each patient and assign BTKiCombo(+/-)
python pairwise_patient_inference.py

# Kaplan-Meier by predicted group
Rscript fig_ls_km_n229.R

# held-out variant, refit per seed
python run_heldout_survival.py
python run_pairwise_direct_survival.py

# multivariable Cox
python fit_ls_cox.py
```

## Multivariable Cox

`fit_ls_cox.py` fits PFS ~ BTKiCombo + age + COO + genetic subtype on all 229
patients and writes `results/ls_cox_n229.csv`:

| Covariate | HR | 95% CI | p |
|---|---|---|---|
| BTKiCombo(+) | 0.641 | [0.424, 0.969] | 0.035 |
| Age (per year) | 1.019 | [1.006, 1.033] | 0.0046 |
| COO: GCB (vs ABC) | 0.366 | [0.196, 0.685] | 0.0017 |
| Genetic subtype: MCD (vs Other) | 2.287 | [1.239, 4.223] | 0.0082 |

The predicted BTKiCombo(+) group retains a lower hazard after
adjusting for age, cell-of-origin and genetic subtype.

`lps_classifier.py` reimplements the LPS subtype classifier used to place cases into
DLBCL subgroups; `tune_ngenes.py` selects the panel size behind it. 
