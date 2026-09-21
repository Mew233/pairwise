# 06 — NCI-DREAM Challenge benchmark

Independent validation on the NCI-DREAM Drug Synergy Challenge (2014): 14 compounds in
OCI-LY3, scored by the challenge's own probabilistic c-index.

Every method gets the same OCI-LY3 adaptation strategy. LODO excludes the held-out drug's
pairs from that fitting.

Comparators (SynGen, DIGRE, DrugComboRanker, RACS) are in
[`baseline/dream_methods/`](../../baseline/dream_methods) so that the boundary between
this work and previously published methods stays visible.

## Data

The challenge data is under `data/` (gold standard
`NCI-DREAMSubchallenge2_GoldStandard.txt`, the official `Team Predictions/`
submissions, the ACalifano OCI-LY3 response matrices, and the official MATLAB
scoring script for reference). It originates from the NCI-DREAM Drug Synergy
Challenge (synapse.org, registration required for re-download). The comparator
modules are `syngen.py`, `drugcomboranker.py`, `racs_labelfree.py`,
`digre_native.py`, `digre_semisupervised.py` and `racs_semisupervised.py`.

## Run

```bash
cd paper/06_dream_benchmark
python run_all.py       # validates the pc-index scorer, then scores the baselines
```

`dream_pcindex.py` implements the challenge probabilistic c-index used for scoring;
`semisup_common.py` holds the shared semi-supervised ranking machinery the comparator
modules plug into. `run_all.py` writes per-method ranking files into `predictions/`.
