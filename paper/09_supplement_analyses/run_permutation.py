"""
run_permutation.py   hold the drug pair FIXED, swap the transcriptome, measure
how much the prediction moves. If the sample branch contributes, predictions
must shift; the more dissimilar the swapped sample, the bigger the shift.

Runnable end-to-end:

    python run_permutation.py \
        --synergy_df p13 --seed 42 \
        --weights <repo>/weights/best_model_pairwise.pth \
        --out_dir results/permutation_rerun

"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(STAGE_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pairwise.get_model import get_model
from pairwise.main import apply_model_config
from pairwise.paths import RESULTS_DIR
from pairwise.pipeline import prepare_data
from pairwise.seeding import get_seed, seed_everything


@torch.no_grad()
def permute_predict(model, model_weights, test_inputs, tissue_labels, seed=42,
                    device="cpu"):
    """
    model         : an instantiated Pairwise model
    model_weights : path to best_model_pairwise.pth
    test_inputs   : dict with keys 'src'(drug target stack tensor list),
                    'fp'(grover fp list), 'cell'(transcriptome tensor list),
                    'y'(labels), 'index'
    tissue_labels : array of tissue/lineage string per test row (for stratified swap)
    device        : 'cpu' (default, historical behavior) or e.g. 'cuda:0'
    Returns a DataFrame with original + permuted predictions for each scheme.
    """
    rng = np.random.RandomState(seed)
    model.eval()
    model.load_state_dict(torch.load(model_weights, map_location="cpu"))
    model = model.to(device)

    cell = list(test_inputs["cell"])
    n = len(cell)
    tissue = np.asarray(tissue_labels)

    # build swap index maps
    def same_tissue_swap():
        order = np.arange(n)
        for t in np.unique(tissue):
            idx = np.where(tissue == t)[0]
            perm = idx.copy(); rng.shuffle(perm)
            # avoid identity where possible
            order[idx] = perm
        return order
    def cross_tissue_swap():
        order = np.arange(n)
        for i in range(n):
            others = np.where(tissue != tissue[i])[0]
            order[i] = rng.choice(others) if len(others) else i
        return order

    schemes = {"original": np.arange(n),
               "same_tissue": same_tissue_swap(),
               "cross_tissue": cross_tissue_swap()}

    out = {"index": test_inputs["index"], "y": test_inputs["y"]}
    for name, order in schemes.items():
        preds = []
        # rebuild a loader-like loop, batch of 256, matching your code
        for start in range(0, n, 256):
            sl = slice(start, min(start+256, n))
            src   = torch.stack([test_inputs["src"][i] for i in range(*sl.indices(n))]).to(device)
            fp    = torch.stack([test_inputs["fp"][i]  for i in range(*sl.indices(n))]).to(device)
            cl_id = [order[i] for i in range(*sl.indices(n))]
            cl    = torch.stack([cell[j] for j in cl_id]).to(device)
            # Pairwise.forward(dts, fps, cell) returns a single sigmoid'd
            # probability tensor of shape (B, 1) — not a tuple.
            res = model(src, fp, cl)
            y_pred = res[0] if isinstance(res, (tuple, list)) else res
            preds.append(y_pred.squeeze(-1).cpu().numpy())
        out[name] = np.concatenate(preds)
    return pd.DataFrame(out)


def summarize(df):
    """Compute ρ(original, permuted) and ΔAUROC for each swap scheme."""
    rows = []
    auc0 = roc_auc_score(df["y"], df["original"])
    for scheme in ["same_tissue", "cross_tissue"]:
        rho = spearmanr(df["original"], df[scheme]).correlation
        auc = roc_auc_score(df["y"], df[scheme])
        rows.append(dict(scheme=scheme, spearman_rho=rho,
                         AUROC=auc, delta_AUROC=auc - auc0))
    s = pd.DataFrame(rows)
    s.attrs["AUROC_original"] = auc0
    return s


def build_model_args(synergy_df, seed):
    """argparse.Namespace equivalent to `python -m pairwise.main --model pairwise`
    with the shipped config_pairwise.json applied (mirrors cv_utils.make_model_args)."""
    args = argparse.Namespace(
        model="pairwise",
        synergy_df=synergy_df,
        synergy_thres=0,
        seed=seed,
        batch_size=256,
        epochs=50,
        train_test_mode="test",
        external_validation=False,
        SHAP_analysis=False,
        ablate="none",
        ri_thres=10,
        cell_df="CCLE",
    )
    return apply_model_config(args)


def main():
    parser = argparse.ArgumentParser(
        description="Transcriptome-swap permutation test for PAIRWISE "
                    "(held-out p13 test set, no retraining).")
    parser.add_argument("--synergy_df", type=str, default="p13")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=str,
                        default=os.path.join(REPO_ROOT, "weights",
                                             "best_model_pairwise.pth"),
                        help="path to best_model_pairwise.pth")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(STAGE_DIR, "results",
                                             "permutation_rerun"),
                        help="where to write rerun CSVs (must NOT be the "
                             "shipped results/ dir of this stage)")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    cli = parser.parse_args()

    # Guard: never clobber the shipped historical artifacts.
    shipped_dir = os.path.join(STAGE_DIR, "results")
    if os.path.abspath(cli.out_dir) == os.path.abspath(shipped_dir):
        parser.error(f"--out_dir must not be the shipped results dir: {shipped_dir}")

    seed_everything(cli.seed)
    args = build_model_args(cli.synergy_df, cli.seed)
    print(f"seed: {get_seed(args)}  device: {cli.device}")
    print(f"model: pairwise  drug_omics: {args.drug_omics}  "
          f"cell_filtered_by: {args.cell_filtered_by}")

    # --- 1. data prep: identical call to the training pipeline -------------
    X_cell, X_drug, Y, _Y_ic1, _Y_ic2 = prepare_data(args)

    processed_path = os.path.join(
        RESULTS_DIR, f"processed_synergydf_{args.synergy_df}.csv")
    synergy_df = pd.read_csv(processed_path, index_col=0)
    dummy = np.array(synergy_df.index)

    # --- 2. pairwise-branch tensor assembly (pipeline.training) ------------
    drug_feat = args.drug_omics[0]  # drug_target_rwr
    X, X2, X3 = [], [], []
    for d1, d2, cell, fp1, fp2 in zip(
            X_drug[f"{drug_feat}_1"], X_drug[f"{drug_feat}_2"], X_cell,
            X_drug["smiles_grover_1"], X_drug["smiles_grover_2"]):
        X.append(torch.from_numpy(np.vstack((d1, d2))).float())
        X2.append(torch.from_numpy(np.vstack((fp1, fp2))).float())
        X3.append(torch.from_numpy(np.array(cell)).float())

    # --- 3. the exact held-out split (test_size=0.2, random_state=seed) -----
    _X_tr, X_test, _Y_tr, Y_test, _d_tr, dummy_test, \
        _X2_tr, X2_test, _X3_tr, X3_test = train_test_split(
            X, Y, dummy, X2, X3, test_size=0.2, random_state=get_seed(args))
    del X, X2, X3, _X_tr, _Y_tr, _d_tr, _X2_tr, _X3_tr, X_drug, X_cell

    print(f"test rows: {len(Y_test)}")
    assert len(Y_test) == 60927, \
        f"expected 60927 test rows, got {len(Y_test)} — split mismatch"

    test_inputs = {
        "src": X_test,       # stacked drug-target RWR tensors
        "fp": X2_test,       # stacked grover fingerprint tensors
        "cell": X3_test,     # cell transcriptome tensors
        "y": np.asarray(Y_test),
        "index": np.asarray(dummy_test),
    }
    tissue_labels = synergy_df.loc[test_inputs["index"], "tissue_name"].values
    print(f"tissues: {len(set(tissue_labels))} types")

    # --- 4. model + 3 inference passes --------------------------------------
    model = get_model("pairwise")
    df_results = permute_predict(
        model=model,
        model_weights=cli.weights,
        test_inputs=test_inputs,
        tissue_labels=tissue_labels,
        seed=cli.seed,
        device=cli.device,
    )
    summary = summarize(df_results)

    # Rename to the schema the downstream scripts (bootstrap_permutation_ci.py,
    # fig_permutation.py) and the shipped CSV use: pred_original, ...
    df_results = df_results.rename(columns={
        "original": "pred_original",
        "same_tissue": "pred_same_tissue",
        "cross_tissue": "pred_cross_tissue",
    })

    print("\n" + "=" * 50)
    print("PERMUTATION TEST RESULTS")
    print("=" * 50)
    print(f"Original AUROC: {summary.attrs['AUROC_original']:.6f}")
    print(summary.to_string(index=False))

    os.makedirs(cli.out_dir, exist_ok=True)
    df_path = os.path.join(cli.out_dir, "permutation_pairwise.csv")
    sum_path = os.path.join(cli.out_dir, "permutation_summary.csv")
    df_results.to_csv(df_path, index=False)
    summary.to_csv(sum_path, index=False)
    print("\nResults saved to:")
    print(f"  - {df_path}")
    print(f"  - {sum_path}")


if __name__ == "__main__":
    main()
