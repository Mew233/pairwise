import os
os.environ["OMP_NUM_THREADS"] = "4"
import argparse
from prepare_data import *
from select_features import *
from pipeline import *



def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--synergy_thres', type=int, default=0,
                        help='synergy threshold (default: loewe score)')
    parser.add_argument('--ri_thres', type=int, default=10,
                        help='percentage inhibition IC50')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='maximum number of epochs (default: 10)')
    parser.add_argument('--train_test_mode', type=str, default='train',
                        help='train or test or fine_tune')
    parser.add_argument('--SHAP_analysis', type=bool, default=False)
    parser.add_argument('--model', type=str, default='deepsynergy_preuer',
                        help='import model')
                        #options are 'LR','XGBOOST','RF','ERT','deepsynergy_preuer','multitaskdnn_kim',
                        # 'matchmaker_brahim','deepdds_wang','TGSynergy','transynergy_liu', 'graphsynergy',"pairwise")

# --------------- Parse configuration  --------------- #

    parser.add_argument('--synergy_df', type=str, default='p13',
                        help = 'p13 or Sanger2022 or Customized')
    parser.add_argument('--external_validation', type=bool, default=False,
                        required=False, help = 'True for Sanger2022 or Customized')
    parser.add_argument('--drug_omics', nargs="+", default=["morgan_fingerprint"],
                        required=False, help='drug_target/drug_target_rwr/morgan_fingerprint\
                            /smiles2graph/smiles2graph_TGSynergy/chemical_descriptor/smiles\
                            hetero_graph/ smiles_grover(3285)')    
    parser.add_argument('--cell_df', type=str, default='CCLE',
                        help='"CCLE","Customized"')
    parser.add_argument('--cell_omics', nargs="+", default=['exp'],
                        required=False, help='"exp","cn","mut","GNN_cell')
    parser.add_argument('--cell_filtered_by', type=str, default='all',
                        required=False,help='top genes selected by variance or STRING graph or dti(for Transynergy) or dti')
    parser.add_argument('--get_cellfeature_concated', type=bool, default=True,
                        required=False)
    parser.add_argument('--get_drugfeature_concated', type=bool, default=True,
                        required=False, help='if concat, numpy array')
    parser.add_argument('--get_drugs_summed', type=bool, default=True,
                        required=False, help='drug1+drug2 if True, else return dict')

    return parser.parse_args()


def main():
    args = arg_parse()

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