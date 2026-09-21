import os
os.environ["OMP_NUM_THREADS"] = "4"
import argparse

import torch

from . import paths
from .prepare_data import *
from .seeding import get_seed, seed_everything
from .select_features import *
from .pipeline import *



def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--synergy_thres', type=int, default=0,
                        help='synergy threshold (default: loewe score)')
    parser.add_argument('--ri_thres', type=int, default=10,
                        help='percentage inhibition')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate for Adam (default: 1e-4, matching the manuscript)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay for Adam (default: 0.0)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='maximum number of epochs (default: 10)')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='K-fold CV folds (default: 5; use 2 for smoke tests)')
    parser.add_argument('--custom_split', type=str, default=None,
                        help='split tag (e.g. scaffold, lineage); suffixes checkpoints and predictions')
    parser.add_argument('--suffix_threshold', action='store_true',
                        help='also suffix outputs with synergy_thres, for threshold sweeps')
    parser.add_argument('--gpu', type=int, default=0,
                        help='CUDA device index; ignored when no GPU is visible')
    parser.add_argument('--train_test_mode', type=str, default='test',
                        help='train or test or fine_tune')
    parser.add_argument('--SHAP_analysis', type=bool, default=False)
    parser.add_argument('--model', type=str, default='pairwise',
                        help='model name (pairwise, transynergy_liu, ...)')

# --------------- Parse configuration  --------------- #

    parser.add_argument('--synergy_df', type=str, default='p13',
                        help = 'p13 or Customized')
    parser.add_argument('--external_validation', type=bool, default=False,
                        required=False, help = 'True for Customized')
    parser.add_argument('--drug_omics', nargs="+", default=None,
                        required=False, help='overridden by pairwise/configs/config_<model>.json unless set')
    parser.add_argument('--cell_df', type=str, default=None,
                        help='"CCLE","Customized"; default from config JSON')
    parser.add_argument('--cell_omics', nargs="+", default=None,
                        required=False, help='"exp","cn","mut","GNN_cell')
    parser.add_argument('--cell_filtered_by', type=str, default=None,
                        required=False,help='variance / STRING / dti; default from config JSON')
    parser.add_argument('--get_cellfeature_concated', type=bool, default=None,
                        required=False)
    parser.add_argument('--get_drugfeature_concated', type=bool, default=None,
                        required=False, help='if concat, numpy array')
    parser.add_argument('--get_drugs_summed', type=bool, default=None,
                        required=False, help='drug1+drug2 if True, else return dict')

    return parser.parse_args()


def apply_model_config(args):
    """Fill feature settings from the shipped config JSON.

    CLI flags win when explicitly set; otherwise the per-model JSON is the source
    of truth. Without this step, training uses argparse defaults (e.g. drug_target
    instead of drug_target_rwr for PAIRWISE) and KeyErrors on the prepared matrices.
    """
    from .utilitis import configuration_from_json

    cfg = configuration_from_json(args)[args.model]
    for key, val in cfg.items():
        if key == "model_name":
            continue
        current = getattr(args, key, None)
        if current is None:
            setattr(args, key, val)
    return args


def main():
    args = apply_model_config(arg_parse())
    seed_everything(get_seed(args))
    args.device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    print(f"seed: {get_seed(args)}\ndevice: {args.device}")
    print(f"model: {args.model}  drug_omics: {args.drug_omics}  cell_filtered_by: {args.cell_filtered_by}")
    print(f"resolved paths:\n{paths.describe()}")

    write_config(args)
    X_cell, X_drug, Y, Y_ic1, Y_ic2, = prepare_data(args)
    print("data loaded")
    if args.model in ['LR','XGBOOST','RF','ERT']:
        model, scores, test_loader, train_val_dataset = training_baselines(X_cell, X_drug, Y, args)
        print("training finished")
        print(scores)
        val_results = evaluate(model, scores, test_loader, args)
        print("testing started")
        print('ROCAUC: {}, PRAUC: {}'.format(round(val_results['AUC'], 4),round(val_results['AUPR'], 4)))
        print('accuracy: {}, precision: {}, recall: {}, f2: {}'.format(round(val_results['accuracy'], 4),\
            round(val_results['precision'], 4),round(val_results['recall'], 4),round(val_results['f2'], 4)))

    # for deep learning models
    else:
        model, network_weights, test_loader, train_val_dataset = training(X_cell, X_drug, Y, Y_ic1, Y_ic2, args)
        print("training finished")
        val_results = evaluate(model, network_weights, test_loader, train_val_dataset, args)
        print("testing started")
        print('ROCAUC: {}, PRAUC: {}'.format(round(val_results['AUC'], 4),round(val_results['AUPR'], 4)))
        print('accuracy: {}, precision: {}, recall: {}, f1: {}'.format(round(val_results['accuracy'], 4),\
            round(val_results['precision'], 4),round(val_results['recall'], 4),round(val_results['f1'], 4)))

    # save results
    # with open("results/%s.json"%(args.model), "w") as f:
    #     json.dump(val_results, f)


if __name__ == "__main__":
    main() 