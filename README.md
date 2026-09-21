# PAIRWISE: Deep Learning-based Prediction of Effective Personalized Drug Combinations in Cancer

PAIRWISE predicts whether a drug pair acts synergistically in a *specific* tumour
sample. It fuses three modalities — molecular graphs of the two compounds, their
drug–target interaction profiles propagated over a protein–protein interaction
network, and the sample transcriptome — through an attention encoder into a single
synergy probability.

This repository contains the model, the seven benchmarked baselines, and the code
that reproduces figure and table in the manuscript.

- Code: <https://github.com/Mew233/pairwise>
- All Data and model weights and code: [Figshare] https://doi.org/10.6084/m9.figshare.33950731
- Interactive predictions: [synergy explorer](https://mew233.shinyapps.io/synergyy_shinyr/) · [BTKi explorer](https://mew233.shinyapps.io/PAIRWISE_Explorer/)

---

## 1. Installation

```bash
conda create -n pairwise python=3.10
conda activate pairwise

# torch and dgl carry CUDA-specific wheels; install them first
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

git clone https://github.com/Mew233/pairwise.git
cd pairwise
pip install -e .
```

Verify:

```bash
python -c "import pairwise; print(pairwise.__version__)"
```

## 2. Data setup

```
data/
├── synergy_data/p13/drugcomb_trueset_NoDup.csv   # ~30K curated combinations
├── cell_line_data/CCLE/                          # expression, CNV, mutation, PPI graphs
├── cell_line_data/tcga/tcga_encoder.pth          # pretrained transcriptome autoencoder
├── drug_data/                                    # DTI matrices, structures.sdf, smiles.grover
└── gene_sets/
```

See [paper/00_data/data_README.md](paper/00_data/data_README.md) for the full schema and
for how the p13 benchmark was curated from 13 public screens.



## 3. Quick start

`example/example2run.ipynb` walks through prediction, feature extraction, fine-tuning, and
training. Full-data training uses `k_fold_trainer_graph_combonet` (Adam `lr=1e-4`, BCE, batch 256,
50 epochs × 2 passes, 5-fold CV) and reaches ~0.85 AUROC (held-out / leave-combo).

**Train a model**

```bash
python -m pairwise.main --model pairwise --synergy_df p13 --train_test_mode train
```

Feature settings (`drug_omics`, `cell_filtered_by`, …) are read from
`pairwise/configs/config_<model>.json` .


**Evaluate a trained checkpoint**

```bash
python -m pairwise.main --model pairwise --synergy_df p13 --train_test_mode test
```

Predictions are written to `results/predicts_{model}_{dataset}.txt` with columns
`index`, `actuals`, `predicts_{model}`; checkpoints go to `weights/best_model_{model}.pth`.

The `pairwise-train` console script is installed as an alias for `python -m pairwise.main`.

GROVER, and DLBCL cooclassifier:

```bash
python scripts/download_grover.py
```


## 4. Models included

`--model` accepts any of the following. Each has a matching `pairwise/configs/config_{model}.json`
declaring which drug and cell features it consumes.

| Model | Cell features | Drug features | Reference |
|---|---|---|---|
| `pairwise` | expression → autoencoder | GROVER embeddings + RWR-propagated DTI | this work |
| `transynergy_liu` | expression → transformer | RWR-propagated DTI | [Liu & Xie 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008653) |
| `multitaskdnn_kim` | expression → DNN | fingerprints + DTI | [Kim et al. 2021](https://academic.oup.com/jamia/article/28/1/42/5920819) |
| `deepsynergy_preuer` | expression → DNN | chemical descriptors | [Preuer et al. 2018](https://academic.oup.com/bioinformatics/article/34/9/1538/4747884) |
| `deepdds_wang` | expression → MLP | SMILES → graph → GCN | [Wang et al. 2021](https://academic.oup.com/bib/article/23/1/bbab390/6375262) |
| `matchmaker_brahim` | expression → DNN | chemical descriptors | [Kuru et al. 2021](https://ieeexplore.ieee.org/document/9447196/) |
| `TGSynergy` | expression → GCN | SMILES → graph → GCN | [Zhu et al. 2022](https://academic.oup.com/bioinformatics/article-abstract/38/2/461/6374919) |
| `graphsynergy` | cell–protein + PPI → GCN | drug–protein + PPI → GCN | [Yang et al. 2021](https://academic.oup.com/jamia/article/28/11/2336/6362567) |
| `LR`, `RF`, `ERT`, `XGBOOST` | expression / CNV / mutation | DTI | scikit-learn baselines |


## 5. License

[MIT](LICENSE)

## 6. Citation

> Chengqi X., et al. PAIRWISE: Deep Learning-based Prediction of Effective Personalized
> Drug Combinations in Cancer.
