# 03 — Network and pathway analysis



We investigate the network component of the model. First, whether the proteins
targeted by the BTKi combinations PAIRWISE highlights sit closer together in the
interactome than chance allows. Second, whether the random-walk-with-restart
propagation that turns drug targets into features actually contributes, and how
sensitive it is to the restart probability.

## Run

```bash
cd paper/03_network_pathway

# Fig. 5b: size of the largest connected component vs a degree-preserving null
python eval_fig5b_cc_null.py

# RWR: observed statistic against degree-preserving and uniform nulls
python eval_rwr_null.py

# restart-probability sweep metrics (from the shipped per-alpha predictions)
python eval_rwr_sweep.py
```

Both null models resample from the STRING network in
`data/cell_line_data/PPI/protein-protein_network.csv` while preserving each node's
degree.

## Output

| File | Content |
|---|---|
| `results/fig5b_cc_observed.csv` | Fig. 5b: observed largest connected component (20 of 21 seed targets) with both null comparisons |
| `results/fig5b_cc_null_degree.csv` | Degree-preserving null for the component size (mean 16.27, SD 1.99) |
| `results/fig5b_cc_null_uniform.csv` | Uniform-sampling null for the component size (mean 2.35, SD 1.14) |
| `results/rwr_observed.csv` | Observed statistic at the chosen restart probability |
| `results/rwr_null.csv` | Degree-preserving null distribution |
| `results/rwr_null_uniform.csv` | Uniform-sampling null, as a weaker comparison |
| `results/rwr_sweep.csv` | AUROC across restart probabilities (full precision, regenerated) |

The model is robust to the restart
probability.