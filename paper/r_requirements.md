# R environment for figure scripts

The `paper/` figure scripts (11 `.R` files) plus the DREAM comparator scripts in
`baseline/dream_methods/` require R ≥ 4.3 and the packages below. The versions
shown are the ones used to produce the shipped figures (captured with
`sessionInfo()` on R 4.5.1, 2026-08); earlier minor versions are expected to
work as well.

## CRAN

| Package | Version used | Used by |
|---|---|---|
| readr | 2.1.5 | all stage figure scripts |
| dplyr | 1.1.4 | all stage figure scripts |
| tidyr | 1.3.1 | stage 02/05/06/07 figures |
| ggplot2 | 3.5.2 | all stage figure scripts |
| scales | 1.4.0 | stage 02/05/06 figures |
| gridExtra | 2.3 | stage 02/05 figures |
| patchwork | 1.3.1 | stage 06 figures |
| survival | 3.8.3 | stage 05 KM/Cox figures |
| survminer | 0.5.2 | stage 05 KM figures |
| grid | (base R) | stage 05 figures |

Install with:

```r
install.packages(c("readr", "dplyr", "tidyr", "ggplot2", "scales",
                   "gridExtra", "patchwork", "survival", "survminer"))
```

## Bioconductor (DREAM comparators only, `baseline/dream_methods/`)

| Package | Version used | Used by |
|---|---|---|
| preprocessCore | 1.70.0 | `digre_native.R` |
| org.Hs.eg.db | 3.21.0 | `digre_native.R` |
| DIGREsyn | not on CRAN/Bioc — see note | `run_digre.R` |

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("preprocessCore", "org.Hs.eg.db"))
```

**DIGREsyn note.** `DIGREsyn` is the third-party R package accompanying DIGRE
(Yang et al. 2015, doi:10.1038/psp.2015.17). It is not on CRAN/Bioconductor;
obtain it from the authors' distribution (see `baseline/README.md`).
It is only needed to rerun the DIGRE comparator from scratch — all DIGRE result
files used in the benchmark are already shipped under
`paper/06_dream_benchmark/results/`.
