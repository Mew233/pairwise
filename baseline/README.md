# Baselines

## `dream_methods/`

Reimplementations of the published methods PAIRWISE is compared against on the
NCI-DREAM Challenge. 

| File | Method | Reference |
|---|---|---|
| `digre_native.py`, `digre_native.R`, `run_digre.R` | DIGRE | [Yang et al. 2015](https://doi.org/10.1038/psp.2015.17) |

The other comparator modules (SynGen, DrugComboRanker, RACS and the
semi-supervised variants) live next to the benchmark that runs them, in
[`../paper/06_dream_benchmark/`](../paper/06_dream_benchmark), because they import
its shared `semisup_common.py` / `dream_pcindex.py`:

| File | Method | Reference |
|---|---|---|
| `syngen.py` | SynGen | Challenge entry |
| `drugcomboranker.py` | DrugComboRanker | [Huang et al. 2014](https://doi.org/10.1093/bioinformatics/btu278) |
| `racs_labelfree.py` | RACS | [Sun et al. 2015](https://doi.org/10.1038/ncomms9481) |

## Deep-learning baselines

The seven deep models benchmarked on p13 (TranSynergy, DeepDDS, DeepSynergy,
MatchMaker, Multitask-DNN, TGSynergy, GraphSynergy) ship inside the package as
`pairwise/models/`, because they share the feature pipeline and are selected with
`--model`. They are listed with references in the [main README](../README.md#5-models-included).
