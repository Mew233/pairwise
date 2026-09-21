# PAIRWISE — Dataset and Model Weights

**PAIRWISE: Deep Learning-based Prediction of Effective Personalized Drug
Combinations in Cancer**

This deposit contains all input data, trained model weights, and per-stage
analysis inputs required to reproduce the manuscript,
and to apply PAIRWISE to new samples.

- Code: <https://github.com/Mew233/pairwise> (`pip install -e .`)
- Interactive prediction: [synergy explorer](https://mew233.shinyapps.io/synergyy_shinyr/) · [BTKi explorer](https://mew233.shinyapps.io/PAIRWISE_Explorer/)



## 1. Deposit layout

```
data/                training & benchmark input data (~8 GB)
weights/             trained model checkpoints (~1.8 GB)
results/             trained-model predictions
example/             minimal runnable example (notebook + sample data)
paper/               external screens, DREAM challenge data, wet-lab data, survival cohorts
```

## 2. Input data inventory (`data/`)

### 2.1 Synergy labels — `data/synergy_data/`

| File | Size | Content | Used for |
|---|---|---|---|
| `p13/p13_trueset.csv` | 504 MB | Gold-standard p13 dataset: 304,632 drug1–drug2–cell-line records with recomputed Loewe/Bliss/ZIP/HSA synergy scores (SynergyFinder v3.0), harmonized from 13 public screens; 1,275 drugs × 163 cell lines | All training and benchmarking  |

### 2.2 Drug features — `data/drug_data/`

| File | Size | Content | Used for |
|---|---|---|---|
| `input_drug_data.npy` | 746 MB | Per-drug feature dictionary: DrugBank target profiles, Morgan fingerprints, GROVER embeddings, SMILES graphs | All models |
| `structures.sdf` | 51 MB | Chemical structures | Feature regeneration |
| `smiles.grover/` | 17 MB | Precomputed GROVER features | Chemical module |
| `all_targets.csv`, `all_enzyme.csv`, `all_carrier.csv`, `all_transporter.csv`, `dtc_db2ncbi.csv`, `hgnc_ncbi.txt` | ~3 MB | DrugTargetCommons v2.0 drug–target interactions and ID mappings (source of the 2,614-drug × 4,098-gene DTI matrix) | Drug-target module |
| `rwr_sweep/` | 1.1 GB | RWR-propagated drug–target features at restart probabilities 0.1–0.9 | RWR for DTI features|
| `markerset_rwr/` | 516 MB | RWR features restricted to alternative gene panels (LINCS, HVG2000, PPI-hub) | gene-panel comparison |

### 2.3 Cell-line / transcriptome features — `data/cell_line_data/`

| File | Size | Content | Used for |
|---|---|---|---|
| `CCLE/CCLE_exp.csv` | 403 MB | CCLE gene expression | Transcriptome module |
| `CCLE/CCLE_cn.csv` | 795 MB | CCLE copy number | Feature banks |
| `CCLE/CCLE_mut.csv` | 246 MB | CCLE mutations (1,083-gene panel) | PAIRWISE-Mut variant |
| `CCLE/input_cellline_data.npy` | 674 MB | Assembled per-cell-line feature dict (expression embedding inputs) | All models |
| `CCLE/cell_feature_std.npy` | — | Standardized cell features | Inference pipeline |
| `CCLE/CCLE_GSVA_scores.csv` | 29 MB | Pathway-level (GSVA) features | PAIRWISE-GSVA variant |
| `PPI/string_network`, `PPI/protein-protein_network.csv`, `PPI/edge_index_PPI_0.95.npy` | 25 MB | STRING protein–protein interaction network | RWR propagation; null models |
| `tcga/tcga_exp.csv` | 662 MB | TCGA pan-cancer expression (autoencoder training) | AE-CCLE pretraining |
| `tcga/tcga_encoder.pth`, `tcga/tcga_encoder_4092genes.pth` | 19 MB each | Pretrained transcriptome autoencoders | Expression embedding (frozen) |
| `Customized/TCGA_PAN_RNA_TPM` | 727 MB | TCGA pan-cancer TPM matrix | patient analyses |
| `Customized/tcga_DLBC_exp.csv`, `crc_exp.csv` | ~36 MB | DLBCL / colon expression subsets | , external cohorts |

### 2.4 Gene sets — `data/gene_sets/`

| File | Content | Used for |
|---|---|---|
| `lincs_landmark_entrez.txt` | LINCS L1000 landmark genes | Stage 07 markerset comparison |

## 3. Model weights (`weights/`)

| File | Model | Notes |
|---|---|---|
| `best_model_pairwise.pth` | **PAIRWISE (main model)** | p13-trained; held-out test AUROC 0.8444 (`example/logs/test_p13.log`) |
| `best_model_transynergy_liu.pth` | TranSynergy | Regeneration checkpoint (see §7 note) |
| `best_model_deepdds_wang.pth` | DeepDDS | Baseline |
| `best_model_deepsynergy_preuer.pth` | DeepSynergy | Baseline |
| `best_model_matchmaker_brahim.pth` | MatchMaker | Baseline |
| `best_model_multitaskdnn_kim.pth` | Multitask-DNN | Baseline |
| `best_model_TGSynergy.pth` | TGSynergy | Baseline |
| `best_model_graphsynergy.pth` | GraphSynergy | Baseline |
| `best_model_LR.pth`, `best_model_XGBOOST.pth` | Logistic regression, XGBoost | Baselines |
| `best_model_RF.pth`, `best_model_ERT.pth` | Random Forest, Extra Trees | Baselines (large: 0.66/1.0 GB) |

## 4. Model training results (`results/`)

Evaluation can be reproduced directly:

| Path | Content |
|---|---|
| `processed_synergydf_p13.csv` | Model-ready p13 table (304,632 rows × drug1/drug2/cell/tissue/score/IC50) |
| `proessed_dpi.csv` | Drug–protein interaction seed table (RWR seeds) |
| `meta_clf/metapredicts_<model>_p13.txt` | Out-of-fold CV predictions per model (source of the 5-fold AUROCs) |
| `predicts_<model>_p13_original_thres{0,5,10,20}.txt` | Held-out predictions of models retrained at each Loewe threshold (stage 07) |
| `test_idx_original_thres*.txt` | Test-fold indices for the threshold runs |
| `rwr/` | RWR sweep predictions + per-alpha checkpoints + null-model distributions |
| `ablation/`, `markerset/` | Ablation / gene-panel training logs and predictions |

## 5. External data (publicly accessible)
The DREAM challenge source files are mirrored in `paper/06_dream_benchmark/data/`
for convenience (original: [syn2785787](https://www.synapse.org/Synapse:syn2785787)).

## 6. Quick start

The full example is `example/example2run.ipynb` (sample data under
`example/data/`, sample model under `example/model/`).

### Install

```bash
conda create -n pairwise python=3.10 && conda activate pairwise
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install -e .          # from the code repository
```

### Predict synergy for new drug–drug–sample triples

```python
import pandas as pd
from pairwise import loadpairwise

# Input: metadata + precomputed features (see example/data/p13_example_predict_*)
meta = pd.read_csv('example/data/p13_example_predict_meta.csv')   # columns: drug1, drug2, cell
# features: the npz companion holding cell/drug feature stacks (see notebook)

model = loadpairwise('weights/best_model_pairwise.pth', map_location='cpu')
probs, df_pred = model.predict(meta, features)
# probs: array of synergy probabilities; df_pred: meta + probability column
```

### Extract learned features (for downstream Cox/logistic models)

```python
df_cell, df_fp, df_e1, df_e2 = model.extract(meta, features)
```

### Train from scratch

```python
from pairwise import Trainer
trainer = Trainer(lr=1e-4, epochs=50, batch_size=256, seed=42)
model, checkpoint, val_results = trainer.train()   # 5-fold CV on p13
```

### Input / output specification

- **Input metadata** (`meta`): one row per query with columns
  `drug1, drug2, cell` (aliases `drug_row, drug_col, DepMap_ID` accepted).
- **Input features**: produced by `pairwise.pipeline.prepare_data` (training) or the notebook's `load_features` helper (inference): per-sample cell
  transcriptome embedding input, per-drug RWR-propagated target profile,
  Morgan fingerprint, and GROVER embedding.
- **Output**: `probs` — synergy probability per triple (float in [0, 1]);
  `df_pred` — metadata + predictions. Binary synergy labels use Loewe > 0.


R figure scripts need the environment listed in `paper/r_requirements.md`.


## 7. License and citation

Code is released under the license in `LICENSE` (repository root). Data files
remain subject to the terms of their original sources (DrugComb and the 13
constituent screens, CCLE/DepMap, TCGA, STRING, DrugTargetCommons, DrugBank,
the NCI-DREAM Challenge, and Griner et al.). If you use this
deposit, please cite the PAIRWISE manuscript and the underlying data sources.
