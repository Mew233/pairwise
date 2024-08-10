"""
    A collection of full training and evaluation pipelines.
"""
from curses.ascii import DC2
from logging import raiseExceptions
from stringprep import in_table_a1
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm
import joblib
from joblib import dump, load
from sklearn.metrics import roc_auc_score,average_precision_score,cohen_kappa_score
from sklearn.metrics import fbeta_score,accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_validate

from prepare_data import *
from select_features import *
from get_model import get_model
from dataloader import dataloader, dataloader_graph, k_fold_trainer, k_fold_trainer_graph, evaluator, evaluator_graph,\
    k_fold_trainer_graph_TGSynergy,evaluator_graph_TGSynergy, k_fold_trainer_graph_trans, evaluator_graph_trans,\
        k_fold_trainer_graph_combonet,evaluator_graph_combonet
from dataloader import SHAP
import torch_geometric.data
from utilitis import smile_to_graph


def prepare_data(args):
    _configs_ = configuration_from_json(args)
    config = _configs_[args.model]

    print("loading synergy dataset ...")
    synergy_df = load_synergy(config['synergy_df'],args)
    print("loading drug features ...")
    drugFeature_dicts = load_drug_features()
    print("loading cell line features ...")
    cellFeatures_dicts = load_cellline_features(config['cell_df'],args)


    # get full drug set and cell line set
    drugset = synergy_df['drug1'].unique().tolist() + synergy_df['drug2'].unique().tolist()
    cellset = synergy_df['cell'].unique().tolist()

    # Cleaning synergy data. Drugs not in drug–target interaction(DTI) are removed
    selected_drugs = get_drug(drugset)
    print("\ndrug features contructed")

    # Cleaning synergy data. Cells not having top variance genes are removed
    ## Cell feats: multi-omics dataset / get_cell select top genes by variance or kegg pathway
    if args.cell_omics[0] in ["exp","cn","mut"]:
        cell_feats, selected_cells, selected_genes = get_cell(cellFeatures_dicts, cellset, config['cell_omics'], \
            config['cell_filtered_by'], config['get_cellfeature_concated'])
        ## Save selected genes for SHAPLEY analysis
        save_path = os.path.join(ROOT_DIR, 'results','selected_genes.txt')
        np.savetxt(save_path, np.array(selected_genes).astype(int), delimiter=',')

    elif args.cell_omics[0] == 'GNN_cell':
        #这里load了更多的cell
        # cell_feats, selected_cells = cellFeatures_dicts, get_GNNCell()
        #为了与其他模型保持一致, 这里选择的cell只有1000个基因map到的
        cell_feats, selected_cells = cellFeatures_dicts, get_GNNCell(cellFeatures_dicts, cellset)

    print("cell line features constructed")
    synergy_df = synergy_df[(synergy_df['drug1'].isin(selected_drugs))\
        &(synergy_df['drug2'].isin(selected_drugs))&(synergy_df['cell'].isin(selected_cells))]
    
    save_path = os.path.join(ROOT_DIR, 'results','processed_synergydf_%s.csv' % args.synergy_df)
    synergy_df.to_csv(save_path, header=True, index=True, sep=",")

    print("\nSynergy triplets are: ")
    print("\t{} drugs:".format(len(selected_drugs)))
    print("\t{} cells:".format(len(selected_cells)))
    print("\t{} rows:".format(synergy_df.shape[0]))

    print("\ngenerating cell line features...")
    if config['get_cellfeature_concated'] == True:
        # in this case, cell_fets stores a dataframe containing features
        X_cell = np.zeros((synergy_df.shape[0], cell_feats.shape[0]))
        for i in tqdm(range(synergy_df.shape[0])):
            row = synergy_df.iloc[i]
            X_cell[i,:] = cell_feats[row['cell']].values

        # 这里对于hetergnn, 只return cell line name
        if config['model_name'] == "hetergnn":
            pesduoX_cell = []
            for i in tqdm(range(synergy_df.shape[0])):
                row = synergy_df.iloc[i]
                # need to get unique value from string of cell
                int_cell = int(re.findall(r'\d+',row['cell'])[0])
                pesduoX_cell.append(int_cell)
            X_cell = np.array(pesduoX_cell)

    else:
        X_cell = {}
        for feat_type in config['cell_omics']:
        ### 需要修改, append到一个list
            if feat_type=='GNN_cell':
                temp_X_cell = []
                for i in tqdm(range(synergy_df.shape[0])):
                    row = synergy_df.iloc[i]
                    ## This is graph. append graph object to list
                    temp_X_cell.append(cell_feats[feat_type][row['cell']])
                X_cell[feat_type] = temp_X_cell
            
            else:
                print(feat_type, cell_feats[feat_type].shape[0])
                temp_cell = np.zeros((synergy_df.shape[0], cell_feats[feat_type].shape[0]))
                for i in tqdm(range(synergy_df.shape[0])):
                    row = synergy_df.iloc[i]
                    temp_cell[i,:] = cell_feats[feat_type][row['cell']].values
                X_cell[feat_type] = temp_cell
    

    if config['get_cellfeature_concated'] == True:
        print("cell features: ", X_cell.shape)
    else:
        print("cell features:", list(X_cell.keys()))


    # generate matrices for drug features
    # first generate individual data matrices for drug1 and drug2 and different feat types
    print("\ngenerating drug features...")
    drug_mat_dict = {}
    for feat_type in config['drug_omics']:
### 需要修改, append到一个list
        if feat_type in ['smiles2graph','smiles2graph_TGSynergy','smiles','smiles_grover']:
            temp_X_drug1, temp_X_drug2 = [], []
            for i in tqdm(range(synergy_df.shape[0])):
                row = synergy_df.iloc[i]
                ## This is graph. append graph object to list
                temp_X_drug1.append(drugFeature_dicts[feat_type][int(row['drug1'])])
                temp_X_drug2.append(drugFeature_dicts[feat_type][int(row['drug2'])])

        elif feat_type =='hetero_graph':

            temp_X_drug1 = np.zeros((synergy_df.shape[0], 1))
            temp_X_drug2 = np.zeros((synergy_df.shape[0], 1))
            for i in tqdm(range(synergy_df.shape[0])):
                row = synergy_df.iloc[i]
                temp_X_drug1[i,:] = int(row['drug1'])
                temp_X_drug2[i,:] = int(row['drug2'])

        #    pass 

        ## this is valid for tabular features
        else:
            dim = drugFeature_dicts[feat_type].shape[0]
            temp_X_drug1 = np.zeros((synergy_df.shape[0], dim))
            temp_X_drug2 = np.zeros((synergy_df.shape[0], dim))
            drugFeature_dicts['drug_target_rwr'].columns = drugFeature_dicts['drug_target_rwr'].columns.values.astype(int)
            for i in tqdm(range(synergy_df.shape[0])):
                row = synergy_df.iloc[i]
                temp_X_drug1[i,:] = drugFeature_dicts[feat_type][int(row['drug1'])]
                temp_X_drug2[i,:] = drugFeature_dicts[feat_type][int(row['drug2'])]

        drug_mat_dict[feat_type+"_1"] = temp_X_drug1
        drug_mat_dict[feat_type+"_2"] = temp_X_drug2

    # now aggregate drug features based on whether they should be summed (drug1+drug2)
    X_drug_temp = {}
    if config['get_drugs_summed'] == True:
        for feat_type in config['drug_omics']:
            temp_X = drug_mat_dict[feat_type+"_1"] + drug_mat_dict[feat_type+"_2"]
            X_drug_temp[feat_type] = temp_X
    else:
        X_drug_temp = drug_mat_dict
    
    # now aggregate drug features based on whether they should be concatenatd
    if config['get_drugfeature_concated'] == False:
        X_drug = X_drug_temp
    else:
        # in this case, drug feature is a numpy array instead of dict of arrays
        # X_drug, X_drug1, X_drug2 = {}, {}, {}
        # for feat_type in config['drug_omics']:
        #     X_drug1[feat_type] = drug_mat_dict[feat_type+"_1"]
        #     X_drug2[feat_type] = drug_mat_dict[feat_type+"_2"]
        # X_drug['drug_1'] = np.concatenate(list(X_drug1.values()), axis=1)
        # X_drug['drug_2'] = np.concatenate(list(X_drug2.values()), axis=1)
        X_drug = np.concatenate(list(X_drug_temp.values()), axis=1)
    

    if config['get_drugs_summed'] == True:
        print("drug features: ", X_drug.shape)
    else:
        print("drug features")
        print(list(X_drug.keys()))
        for key, value in X_drug.items():
            print(key, len(value))

    
    Y_score = (synergy_df['score']>args.synergy_thres).astype(int).values
    try:
        Y_ic1 = (synergy_df['ic_1']>args.ri_thres).astype(int).values
        Y_ic2 = (synergy_df['ic_2']>args.ri_thres).astype(int).values
    except:
        print('Not using RI here')
        Y_ic1 = None
        Y_ic2 = None
        pass

    return X_cell, X_drug, Y_score, Y_ic1, Y_ic2

def training_baselines(X_cell, X_drug, Y, args):
    if args.external_validation:
        test_size = 0.99999 #0.9999
    else:
        test_size = 0.2
    # --------------- baseline  --------------- #
    if args.model in ['LR','XGBOOST','RF','ERT']:
        X = np.concatenate([X_cell,X_drug], axis=1)
        X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
        
        # init model
        model = get_model(args.model)
        # returned for evaluation
        test_loader = {}
        test_loader['X_test'] = X_test
        test_loader['actuals'] = Y_test

        # prepare the cross-validation procedure
        kfold = KFold(n_splits=5, random_state=42, shuffle=True)
        
        # load the best model
        if args.train_test_mode == 'test':
            rfc_fit = 'best_model_%s.pth' % args.model
            scores = 0
        elif args.train_test_mode == 'train':
            # evaluate model
            cv_results = cross_validate(model, X_trainval, Y_trainval, cv=kfold, scoring='roc_auc', return_estimator=True)
            scores = cv_results['test_score']
            rfc_fit = cv_results['estimator']
            # select the best
            rfc_fit = rfc_fit[np.argmax(scores)]
            # save it
            save_path = os.path.join(ROOT_DIR, 'best_model_%s.pth' % args.model)
            joblib.dump(rfc_fit,save_path)
            # should return address at the end
            rfc_fit = 'best_model_%s.pth' % args.model

    return rfc_fit, scores, test_loader


def training(X_cell, X_drug, Y, Y_ic1, Y_ic2, args):
    if args.external_validation:
        test_size = 1-0.1**(len(str(len(X_cell)))-1) #0.9999,0.99999
    else:
        test_size = 0.2
    
    save_path = os.path.join(ROOT_DIR, 'results','processed_synergydf_%s.csv' % args.synergy_df)
    processed_synergydf = pd.read_csv(save_path, index_col=0, sep=",")
    dummy = np.array(processed_synergydf.index)
# --------------- multitask dnn --------------- #
    if args.model == 'multitaskdnn_kim':
        
        # save_path = os.path.join(ROOT_DIR, 'results','processed_synergydf_%s.csv' % args.synergy_df)
        # processed_synergydf = pd.read_csv(save_path, index_col=0, sep=",")
        # dummy = np.array(processed_synergydf.index)

        X_cell_trainval, X_cell_test, \
        X_fp_drug1_trainval, X_fp_drug1_test,\
        X_fp_drug2_trainval, X_fp_drug2_test,\
        X_tg_drug1_trainval, X_tg_drug1_test,\
        X_tg_drug2_trainval, X_tg_drug2_test,\
        Y_trainval, Y_test,  dummy_trainval, dummy_test = train_test_split(X_cell, X_drug['morgan_fingerprint_1'],X_drug['morgan_fingerprint_2'], \
                                X_drug['drug_target_1'],X_drug['drug_target_2'],Y, dummy, \
                                test_size=test_size, random_state=42)

        save_path = os.path.join(ROOT_DIR, 'results','test_idx.txt')
        np.savetxt(save_path,dummy_test.astype(int), delimiter=',')


        cell_channels = X_cell_trainval.shape[1]
        drug_fp_channels = X_fp_drug1_trainval.shape[1]
        drug_tg_channels = X_tg_drug1_trainval.shape[1]

        # init model, order is important 
        model = get_model(args.model,cell_channels,drug_fp_channels,drug_tg_channels)

        # train_val set for k-fold, test set for testing
        # should be compatible with fp_drug, tg_drug, fp_drug2, tg_drug2, cell
        train_val_dataset, test_loader = dataloader(\
            X_fp_drug1_trainval=X_fp_drug1_trainval, X_fp_drug1_test=X_fp_drug1_test,\
            X_tg_drug1_trainval=X_tg_drug1_trainval, X_tg_drug1_test=X_tg_drug1_test,\
            X_fp_drug2_trainval=X_fp_drug2_trainval, X_fp_drug2_test=X_fp_drug2_test,\
            X_tg_drug2_trainval=X_tg_drug2_trainval, X_tg_drug2_test=X_tg_drug2_test,\
            X_cell_trainval=X_cell_trainval, X_cell_test=X_cell_test,\
            Y_trainval=Y_trainval, Y_test=Y_test, dummy_trainval = dummy_trainval, dummy_test=dummy_test
                )

        # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer(train_val_dataset,model,args)
   
# --------------- deep synergy --------------- #
    elif args.model == 'deepsynergy_preuer':
        
        X = np.concatenate([X_cell,X_drug], axis=1)
        #X_{}_trainval, X_{}_test, Y_{}_trainval, Y_{}_test, dummy_train, dummy_test(为了shap analysis)
        # save_path = os.path.join(ROOT_DIR, 'results','processed_synergydf_%s.csv' % args.synergy_df)
        # processed_synergydf = pd.read_csv(save_path, index_col=0, sep=",")
        # dummy = np.array(processed_synergydf.index)

        X_trainval, X_test, Y_trainval, Y_test, dummy_trainval, dummy_test  = train_test_split(X, Y, dummy, test_size=test_size, random_state=42)
        save_path = os.path.join(ROOT_DIR, 'results','test_idx.txt')
        np.savetxt(save_path,dummy_test.astype(int), delimiter=',')

        channels = X_trainval.shape[1]

        # init model
        model = get_model(args.model,channels)
        
        # train_val set for k-fold, test set for testing
        train_val_dataset, test_loader = dataloader(X_trainval=X_trainval, X_test=X_test, \
            Y_trainval=Y_trainval, Y_test=Y_test, dummy_trainval=dummy_trainval, dummy_test=dummy_test)

        # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer(train_val_dataset,model,args)

# --------------- combonet --------------- #
    elif args.model == 'combonet':
        X, X2, X3 = [],[],[]
        for index, (d1, d2, cell, fp1, fp2) in enumerate(zip(X_drug['drug_target_1'], X_drug['drug_target_2'], X_cell,\
            X_drug['smiles_grover_1'],X_drug['smiles_grover_2'])):
            t = torch.from_numpy(np.vstack((d1,d2))).float()
            # t = torch.from_numpy(np.expand_dims(d1, axis=0)).float()
            t2 = torch.from_numpy(np.vstack((fp1,fp2))).float()
            # t3 = torch.from_numpy(np.vstack((d1,d2,cell))).float()
            t3 = torch.from_numpy(np.array(cell)).float()
            X.append(t.float())
            X2.append(t2.float())
            X3.append(t3.float())

        #len(max(smiles_list, key = len)) is 244

        X_trainval, X_test, Y_trainval, Y_test, dummy_trainval, dummy_test, X2_trainval, X2_test,  \
            X3_trainval, X3_test\
            = train_test_split(X, Y, dummy, X2, X3,test_size=test_size, random_state=42)

        save_path = os.path.join(ROOT_DIR, 'results','test_idx.txt')
        np.savetxt(save_path,dummy_test.astype(int), delimiter=',')

        # init model
        model = get_model(args.model)
        # train_val set for k-fold, test set for testing
        train_val_dataset, test_loader = dataloader_graph(X_trainval=X_trainval, X_test=X_test,\
            Y_trainval=Y_trainval, Y_test=Y_test, dummy_trainval = dummy_trainval, dummy_test=dummy_test, \
                 X2_trainval=X2_trainval, X2_test=X2_test,X3_trainval=X3_trainval, X3_test=X3_test)

        # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer_graph_combonet(train_val_dataset,model,args)

# --------------- transynergy --------------- #
    elif args.model == 'transynergy_liu':
        X=[]
        X2=[] #additional features
        X_sm1, X_sm2 = [], []
        
        for index, (d1, d2, cell, fp1, fp2, sm1, sm2) in enumerate(zip(X_drug['drug_target_1'], X_drug['drug_target_2'], X_cell,\
            X_drug['smiles_grover_1'],X_drug['smiles_grover_2'], \
                X_drug['smiles_1'],X_drug['smiles_2'])):
            array_tuple = (d1, d2)
            array = np.vstack(array_tuple)
            t = torch.from_numpy(array).float()

            t2 = torch.from_numpy(np.vstack((fp1,fp2))).float()
            X.append(t.float())
            X2.append(t2.float())

            ##
            # padded_sm1 = np.pad(sm1, pad_width=(0, 244-len(sm1)), mode='constant', constant_values=0)
            # padded_sm2 = np.pad(sm2, pad_width=(0, 244-len(sm2)), mode='constant', constant_values=0)
            X_sm1.append(torch.from_numpy(np.array(cell)).float())
            X_sm2.append(torch.from_numpy(np.array(cell)).float())

        #len(max(smiles_list, key = len)) is 244

        X_trainval, X_test, Y_trainval, Y_test, dummy_trainval, dummy_test, X2_trainval, X2_test,  \
            X_sm1_trainval, X_sm1_test, X_sm2_trainval, X_sm2_test,\
            = train_test_split(X, Y, dummy, X2, X_sm1, X_sm2, test_size=test_size, random_state=42)

        save_path = os.path.join(ROOT_DIR, 'results','test_idx.txt')
        np.savetxt(save_path,dummy_test.astype(int), delimiter=',')

        # init model
        model = get_model(args.model)
        # src = batch.src.transpose(0, 1)
        # trg = batch.trg.transpose(0, 1)
        # trg_input = trg[:, :-1]
        # src_mask, trg_mask = None, None
        # preds = model(src, trg_input, src_mask, trg_mask)
        # ys = trg[:, 1:].contiguous().view(-1)

        # train_val set for k-fold, test set for testing
        train_val_dataset, test_loader = dataloader_graph(X_trainval=X_trainval, X_test=X_test,\
            Y_trainval=Y_trainval, Y_test=Y_test, dummy_trainval = dummy_trainval, dummy_test=dummy_test, \
                 X2_trainval=X2_trainval, X2_test=X2_test,\
                    X_sm1_trainval=X_sm1_trainval, X_sm1_test=X_sm1_test, X_sm2_trainval=X_sm2_trainval, X_sm2_test=X_sm2_test)

        # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer_graph_trans(train_val_dataset,model,args)
        elif args.train_test_mode == 'fine_tune':
            net_weights = k_fold_trainer_graph_trans(test_loader,model,args)

# --------------- matchmaker --------------- #
    elif args.model == 'matchmaker_brahim':
        
        text = args.drug_omics[0]
        if args.get_drugfeature_concated == True:
            text = "drug"

        X_cell_trainval, X_cell_test, \
        X_fp_drug1_trainval, X_fp_drug1_test,\
        X_fp_drug2_trainval, X_fp_drug2_test,\
        Y_trainval, Y_test, dummy_trainval, dummy_test\
        = train_test_split(X_cell, X_drug[text+'_1'],X_drug[text+'_2'], Y, dummy,\
                                test_size=test_size, random_state=42)

        save_path = os.path.join(ROOT_DIR, 'results','test_idx.txt')
        np.savetxt(save_path,dummy_test.astype(int), delimiter=',')

        #morgan_fingerprint_1,morgan_fingerprint_2
        ##chemical_descriptor_1,chemical_descriptor_2

        cell_channels = X_cell_trainval.shape[1]
        drug_channels = X_fp_drug1_trainval.shape[1]

        # init model
        model = get_model(args.model,cell_channels,drug_channels)
    
        # should be compatible with fp_drug, fp_drug2, cell
        train_val_dataset, test_loader = dataloader(\
            X_fp_drug1_trainval=X_fp_drug1_trainval, X_fp_drug1_test=X_fp_drug1_test,\
            X_fp_drug2_trainval=X_fp_drug2_trainval, X_fp_drug2_test=X_fp_drug2_test,\
            X_cell_trainval=X_cell_trainval, X_cell_test=X_cell_test,\
            Y_trainval=Y_trainval, Y_test=Y_test, dummy_trainval=dummy_trainval, dummy_test=dummy_test
                )

        # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer(train_val_dataset,model,args)

# --------------- deepdds --------------- #
    elif args.model == 'deepdds_wang':
        X_cell_trainval, X_cell_test, \
        X_deepdds_sm_drug1_trainval, X_deepdds_sm_drug1_test,\
        X_deepdds_sm_drug2_trainval, X_deepdds_sm_drug2_test,\
        Y_trainval, Y_test, dummy_trainval, dummy_test \
        = train_test_split(X_cell, X_drug['smiles2graph_1'],X_drug['smiles2graph_2'], Y, dummy,\
                                test_size=test_size, random_state=42)


    
        # should be compatible with fp_drug, fp_drug2, cell
        train_val_dataset,test_loader = dataloader_graph(\
            X_deepdds_sm_drug1_trainval=X_deepdds_sm_drug1_trainval, X_deepdds_sm_drug1_test=X_deepdds_sm_drug1_test,\
            X_deepdds_sm_drug2_trainval=X_deepdds_sm_drug2_trainval, X_deepdds_sm_drug2_test=X_deepdds_sm_drug2_test,\
            X_cell_trainval=X_cell_trainval, X_cell_test=X_cell_test,\
            Y_trainval=Y_trainval, Y_test=Y_test, dummy_trainval=dummy_trainval, dummy_test=dummy_test
                )
        
        # init model
        model = get_model(args.model)


        # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer_graph(train_val_dataset,model,args)


# --------------- TGSynergy --------------- #
    elif args.model == 'TGSynergy':
        X_cell_trainval, X_cell_test, \
        X_deepdds_sm_drug1_trainval, X_deepdds_sm_drug1_test,\
        X_deepdds_sm_drug2_trainval, X_deepdds_sm_drug2_test,\
        Y_trainval, Y_test, dummy_trainval, dummy_test \
        = train_test_split(X_cell['GNN_cell'], X_drug['smiles2graph_TGSynergy_1'],X_drug['smiles2graph_TGSynergy_2'], Y, \
                                 dummy, test_size=test_size, random_state=42)


    #     # should be compatible with fp_drug, fp_drug2, cell
        train_val_dataset,test_loader = dataloader_graph(\
            X_deepdds_sm_drug1_trainval=X_deepdds_sm_drug1_trainval, X_deepdds_sm_drug1_test=X_deepdds_sm_drug1_test,\
            X_deepdds_sm_drug2_trainval=X_deepdds_sm_drug2_trainval, X_deepdds_sm_drug2_test=X_deepdds_sm_drug2_test,\
            X_cell_trainval=X_cell_trainval, X_cell_test=X_cell_test,\
            Y_trainval=Y_trainval, Y_test=Y_test, dummy_trainval=dummy_trainval, dummy_test=dummy_test
                )
        
        save_path = os.path.join(ROOT_DIR, 'results','test_idx.txt')
        np.savetxt(save_path,dummy_test.astype(int), delimiter=',')
        
    # init model
        save_path = os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE','cluster_predefine_PPI_0.95.npy')
        cluster_predefine = np.load(save_path, allow_pickle=True).item()
        cluster_predefine = {i: j for i, j in cluster_predefine.items()}
        model = get_model(args.model,cluster_predefine)

    # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer_graph_TGSynergy(train_val_dataset,model,args)


# --------------- hetergnn --------------- #
    elif args.model == 'hetergnn':
        X_cell_trainval, X_cell_test, \
        X_hetero_graph_drug1_trainval, X_hetero_graph_drug1_test,\
        X_hetero_graph_drug2_trainval, X_hetero_graph_drug2_test,\
        Y_trainval, Y_test, dummy_trainval, dummy_test \
        = train_test_split(X_cell, X_drug['hetero_graph_1'],X_drug['hetero_graph_2'], Y, \
                                 dummy, test_size=test_size, random_state=42)

        train_val_dataset, test_loader = dataloader(\
            X_hetero_graph_drug1_trainval=X_hetero_graph_drug1_trainval, X_hetero_graph_drug1_test=X_hetero_graph_drug1_test,\
            X_hetero_graph_drug2_trainval=X_hetero_graph_drug2_trainval, X_hetero_graph_drug2_test=X_hetero_graph_drug2_test,\
            X_cell_trainval=X_cell_trainval, X_cell_test=X_cell_test,\
            Y_trainval=Y_trainval, Y_test=Y_test, dummy_trainval=dummy_trainval, dummy_test=dummy_test
                )

        save_path = os.path.join(ROOT_DIR, 'results','test_idx.txt')
        np.savetxt(save_path,dummy_test.astype(int), delimiter=',')

        # init model
        save_path = os.path.join(ROOT_DIR, 'data', 'drug_data','input_drug_data.npy')
        drugFeature_dicts = np.load(save_path, allow_pickle=True).item()
        

        graph = drugFeature_dicts["hetero_graph"][0]
        dpi_dict = drugFeature_dicts["hetero_graph"][1]
        cpi_dict = drugFeature_dicts["hetero_graph"][2]

        # save_path = os.path.join(ROOT_DIR, 'results')
        # selected_genes = list(np.loadtxt(os.path.join(save_path,'selected_genes.txt'), delimiter=',').astype(int))

        model = get_model(args.model,graph,dpi_dict,cpi_dict)

        # load the best model
        if args.train_test_mode == 'test':
            net_weights = 'best_model_%s.pth' % args.model
        elif args.train_test_mode == 'train':
            net_weights = k_fold_trainer(train_val_dataset,model,args)


    return model, net_weights, test_loader, train_val_dataset


def evaluate(model, model_weights, test_loader, train_val_dataset, args):

    if args.model in ['LR','XGBOOST','RF','ERT']:
    
        model = load(model)
        actuals, predictions = test_loader['actuals'], model.predict_proba(test_loader['X_test'])[:,1]

    elif args.model in ['deepdds_wang']:

        actuals, predictions, shap_df, features_df, expected_value = evaluator_graph(model, model_weights,test_loader, args)
        #actuals, predictions = evaluator_graph(model, model_weights,test_loader)

    elif args.model in ['TGSynergy']:
        
        actuals, predictions, shap_df, features_df, expected_value = evaluator_graph_TGSynergy(model, model_weights,train_val_dataset, test_loader, args)
    
    elif args.model in ['transynergy_liu']:
        
        actuals, predictions, shap_df, features_df, expected_value = evaluator_graph_trans(model, model_weights,train_val_dataset, test_loader, args)
    
    elif args.model in ['combonet']:

        actuals, predictions, shap_df, features_df, expected_value = evaluator_graph_combonet(model, model_weights,train_val_dataset, test_loader, args)

    else:
        actuals, predictions, shap_df, features_df, expected_value  = evaluator(model, model_weights,train_val_dataset, test_loader, args)

    # save actuals/predictions
    test_idx = list(np.loadtxt(os.path.join(ROOT_DIR, 'results','test_idx.txt')).astype(int))
    predict_df = pd.DataFrame(np.column_stack([test_idx,actuals,predictions]),\
                columns=['index','actuals','predicts_%s' % args.model])
    save_path = os.path.join(ROOT_DIR, 'results','predicts_%s_%s.txt' % (args.model, args.synergy_df))
    predict_df.to_csv(save_path, header=True, index=True, sep=",")

    # save the shap_df and features_df
    save_path_shap = os.path.join(ROOT_DIR, 'results','shap_df_%s_%s_%s.txt' % (args.model, str(expected_value), args.synergy_df))
    save_path_feat = os.path.join(ROOT_DIR, 'results','feat_df_%s_%s_%s.txt' % (args.model, str(expected_value), args.synergy_df))
    try:
        shap_df.to_csv(save_path_shap, header=True, index=True, sep=",")
        features_df.to_csv(save_path_feat, header=True, index=True, sep=",")
    except:
        print('Not calculate Shapley value here!')
        pass

    auc = roc_auc_score(y_true=actuals, y_score=predictions)
    ap = average_precision_score(y_true=actuals, y_score=predictions)

    #Cohens kapa or Matthews correlation coefficien 
    #cohen = cohen_kappa_score(y1=actuals, y2=np.round(predictions),sample_weight=actuals)
    accuracy = accuracy_score(y_true=actuals, y_pred=np.round(predictions))
    precision = precision_score(y_true=actuals, y_pred=np.round(predictions))
    recall = recall_score(y_true=actuals, y_pred=np.round(predictions))
    f1 = fbeta_score(y_true=actuals, y_pred=np.round(predictions), average='binary', beta=1)
 
    val_results = {'AUC':auc, 'AUPR':ap,'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1}

    return val_results
