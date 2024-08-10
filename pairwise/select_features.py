import numpy as np 
import pandas as pd 
import os
from utilitis import *

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

def get_drug(original_list):
    targets = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'drug_data','all_targets.csv'))
    enzymes = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'drug_data','all_enzyme.csv'))
    carrier = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'drug_data','all_carrier.csv'))
    transporter = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'drug_data','all_transporter.csv'))
    all = pd.concat([targets,enzymes,carrier,transporter])
    drug_targets = explode_dpi(all)
    
    dtc = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'drug_data','dtc_db2ncbi.csv'))
    drug_targets_L = pd.concat([drug_targets,dtc])
    drug_targets_L = drug_targets_L.drop_duplicates(subset=['Drug IDs','NCBI_ID'])

    drug_list_with_targets = drug_targets_L['Drug IDs'].unique().tolist()
    selected_drugs = list(set(original_list) & set(drug_list_with_targets))

    return selected_drugs

def get_GNNCell(cellFeatures_dicts, cellset):
    # targets = np.load(os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE','cell_feature_cn_std.npy'),\
    #     allow_pickle=True).item()
    # selected_cells = list(targets.keys())

    temp = cellFeatures_dicts['exp']
    var_df = temp.var(axis=1)
    selected_genes = list(var_df.sort_values(ascending=False).iloc[:1000].index)

    selected_cells = list(set(cellset) & set(list(temp.columns)))

    return selected_cells


def get_cell(cellFeature_dicts, synergy_cellset, cell_omics, cell_filtered_by, matrix=False):
    
    def filter_by_variance():
        if len(cell_omics) > 1:
            # if mut/cnv/exp, use exp
            temp = cellFeature_dicts['exp']
        else:
            temp = cellFeature_dicts[cell_omics[0]]
        var_df = temp.var(axis=1)
        selected_genes = list(var_df.sort_values(ascending=False).iloc[:1000].index)
        
        return selected_genes

    def filter_by_2000_genes():
        # following is copied from prepare_data
        data_dicts = np.load(os.path.join(ROOT_DIR, 'data', 'drug_data','input_drug_data.npy'),allow_pickle=True).item()
        selected_genes = data_dicts['drug_target_rwr'].index
        return selected_genes
    
    def filter_by_706_genes():
        # following is copied from prepare_data
        temp = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'cell_line_data', 'CCLE','CCLE_exp.csv'), index_col=0)
        temp.columns = ['Entrez gene id']+[split_it_cell(_) for _ in list(temp.columns)[1:]]
        df_transpose = temp.T
        df_transpose.columns = df_transpose.iloc[0]
        processed_data = df_transpose.drop(df_transpose.index[0])

        var_df = processed_data.var(axis=1)
        gene_list = list(var_df.sort_values(ascending=False).iloc[:1000].index)

        #
        ppi_data = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'cell_line_data','PPI','protein-protein_network.csv'))
        ppi_data_genes = set(list(ppi_data['protein_a']) + list(ppi_data['protein_b']))
        selected_genes = list(set(gene_list) & set(ppi_data_genes))
        return selected_genes

    def ALL():
        # use before batch corrected CCLE most vairance genes
        # index is drug, so we need clean up the df
        temp = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'cell_line_data','CCLE','CCLE_exp.csv'),sep=',')
        temp.columns = ['Entrez gene id']+[split_it_cell(_) for _ in list(temp.columns)[1:]]
        df_transpose = temp.T
        df_transpose.columns = df_transpose.iloc[0]
        processed_data = df_transpose.drop(df_transpose.index[0])

        var_df = processed_data.var(axis=1)
        selected_genes = list(var_df.sort_values(ascending=False).iloc[:1000].index)
        data_dicts = np.load(os.path.join(ROOT_DIR, 'data', 'drug_data','input_drug_data.npy'),allow_pickle=True).item()
        drug_target = list(data_dicts['drug_target'].index)

        return list(set(selected_genes+drug_target))

    # select genes based on criterion (variance or STRING)
    function_mapping = {'variance':'filter_by_variance', 'STRING':'filter_by_706_genes', 'dti':'filter_by_2000_genes',\
        'all':'ALL'}
    selected_genes = locals()[function_mapping[cell_filtered_by]]()


    CCLE_dicts = {}
    # Iterate over different CCLE type, for example exp or cn or mu
    for ccle_type in cell_omics:
        type_df = cellFeature_dicts[ccle_type]
        ## use cells in both synergyt_df and (CCLE_*)
        selected_cols = list(set(synergy_cellset) & set(list(type_df.columns)))
        ## use selected genes in both CCLE_exp and (CCLE_*)
        selected_rows = list(set(selected_genes) & set(list(type_df.index)))

        trimmed_type_df = type_df.loc[selected_rows, selected_cols]
        #trimmed_type_df.dropna(axis=0, how='any',inplace=True)
        trimmed_type_df = trimmed_type_df.fillna(0)
        trimmed_type_df = trimmed_type_df[~trimmed_type_df.index.duplicated(keep='first')]

        CCLE_dicts[ccle_type] = trimmed_type_df
                
        # if integrate is True, then the return value is a dataframe
        # otherwise, the return value is a dictionary of dataframe
        if matrix == True:
            feats = pd.concat(list(CCLE_dicts.values()))
        else:
            feats = CCLE_dicts

    return feats, selected_cols, selected_rows



def get_drug_feats_dim(drug_data_dicts, drug_feats):
    if len(drug_feats) == 1:
        dims = len(list(drug_data_dicts[drug_feats[0]].values())[0])

    else:
        dims = 0
        for feat_type in drug_feats:
            dims += len(list(drug_data_dicts[feat_type].values())[0])
    
    return dims
