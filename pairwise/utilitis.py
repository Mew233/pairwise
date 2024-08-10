import re
import os
import pandas as pd
import argparse
import networkx as nx
import time
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from rdkit import Chem
import numpy as np
import pandas as pd
import torch
import torch_geometric
import torch_geometric.data
from dgllife.utils import *

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

# split DB -- unique identifier
def split_it(compound):
    return int(re.split('\d*\D+',compound)[1])

# split PubChemID
def split_it_cell(compound):
    return int(re.search(r'\((.*?)\)', compound).group(1))

# split Cell line name like ACH-001113
def split_it_cellName(compound):
    return int(re.findall(r'\d+',compound)[0])

def one_hot(mu,clean_cells_ALL,clean_genes_ALL):
    onehot_rows = list()
    for cell in clean_cells_ALL:
        onehot_row = list()
        temp_cell = mu[mu['DepMap_ID']==cell]
        temp_gene_list = list(temp_cell['Entrez_Gene_Id'])
        onehot_row.append(cell)
        for gene in clean_genes_ALL:
            if gene in temp_gene_list:
                onehot_row.append(1)
            else:
                onehot_row.append(0)
        onehot_rows.append(onehot_row)
    return onehot_rows


# Explode Drug-Target Interaction
def explode_dpi(targets):
    targets = targets[targets['Species'] == 'Humans']
    targets['Drug IDs'] = targets['Drug IDs'].str.split('; ').fillna(targets['Drug IDs'])
    targets = targets.explode('Drug IDs')

    targets = targets[['HGNC ID','Name','Gene Name','Drug IDs']]
    ## convert HGNC ID to NCBI ID
    entrez_IDs_df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'drug_data','hgnc_ncbi.txt'), sep="\t", index_col=False)
    entrez_to_genename = dict(entrez_IDs_df[['HGNC ID', 'NCBI Gene ID']].values)
    targets = targets.dropna(subset=['HGNC ID'])
    targets['NCBI_ID'] = targets['HGNC ID'].apply(lambda x: entrez_to_genename[x])
    ## remove NCBI_ID where is nan
    targets = targets.dropna(subset=['NCBI_ID'])
    targets['NCBI_ID'] = targets['NCBI_ID'].apply(lambda x: int(x))
    targets['Drug IDs'] = targets['Drug IDs'].apply(lambda x: split_it(x))
    drug_targets = targets

    return drug_targets

# read the method_config file
import json
def configuration_from_json(args):
    with open(os.path.join(ROOT_DIR,'configs','%s%s%s' % ('config_',args.model,'.json')), "r") as jsonfile:
        config = json.load(jsonfile)
    return config


## write json from argparse
def write_config(args):
    with open(os.path.join(ROOT_DIR,'configs','%s%s%s' % ('config_',args.model,'.json')), "w") as f:
        json.dump(
            {
                args.model:{
                "synergy_df": args.synergy_df,
                "drug_omics": args.drug_omics,
                "cell_df": args.cell_df,
                "cell_omics": args.cell_omics,
                "cell_filtered_by": args.cell_filtered_by,
                "model_name": args.model,
                "get_cellfeature_concated": args.get_cellfeature_concated,
                "get_drugfeature_concated": args.get_drugfeature_concated,
                "get_drugs_summed": args.get_drugs_summed,
                }
             },
                f
            )
        

## SMILES2Graph for prepare_data for TGSynergy

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    8 features are canonical, 2 features are from OGB
    """
    featurizer_funcs = ConcatFeaturizer([atom_type_one_hot,
                                         atom_degree_one_hot,
                                         atom_implicit_valence_one_hot,
                                         atom_formal_charge,
                                         atom_num_radical_electrons,
                                         atom_hybridization_one_hot,
                                         atom_is_aromatic,
                                         atom_total_num_H_one_hot,
                                         atom_is_in_ring,
                                         atom_chirality_type_one_hot,
                                         ])
    atom_feature = featurizer_funcs(atom)
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    featurizer_funcs = ConcatFeaturizer([bond_type_one_hot,
                                         # bond_is_conjugated,
                                         # bond_is_in_ring,
                                         # bond_stereo_one_hot,
                                         ])
    bond_feature = featurizer_funcs(bond)

    return bond_feature


def smiles2graph(mol):
    """
    Converts SMILES string or rdkit's mol object to graph Data object without remove salt
    :input: SMILES string (str)
    :return: graph object
    """

    if isinstance(mol, Chem.rdchem.Mol):
        pass
    else:
        mol = Chem.MolFromSmiles(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = torch_geometric.data.Data(x=torch.tensor(x, dtype=torch.float),
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr), dtype=torch.float)

    return graph


# -------------------------------------------
## SMILES2Graph for prepare_data for DeepDDS

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    
    graph = None
    try:
        graph = torch_geometric.data.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0))
        graph.__setitem__('c_size', torch.LongTensor([c_size]))

    except Exception:
        pass
    
    return graph


# -------------------------------------------
## pyNBS for prepare_data for transynergy

def normalize_network(network, symmetric_norm=False):
    adj_mat = nx.adjacency_matrix(network)
    adj_array = np.array(adj_mat.todense())
    if symmetric_norm:
        D = np.diag(1/np.sqrt(sum(adj_array)))
        adj_array_norm = np.dot(np.dot(D, adj_array), D)
    else:
        degree = sum(adj_array)
        adj_array_norm = (adj_array*1.0/degree).T
    return adj_array_norm
# Closed form random-walk propagation (as seen in HotNet2) for each subgraph: Ft = (1-alpha)*Fo * (I-alpha*norm_adj_mat)^-1
# Concatenate to previous set of subgraphs
def fast_random_walk(alpha, binary_mat, subgraph_norm, prop_data_prev):
    term1=(1-alpha)*binary_mat
    term2=np.identity(binary_mat.shape[1])-alpha*subgraph_norm
    term2_inv = np.linalg.inv(term2)
    subgraph_prop = np.dot(term1, term2_inv)
    prop_data_add = np.concatenate((prop_data_prev, subgraph_prop), axis=1)
    return prop_data_add

# Wrapper for random walk propagation of full network by subgraphs
# Implementation is based on the closed form of the random walk model over networks presented by the HotNet2 paper
def network_propagation(network, binary_matrix, alpha=0.7, symmetric_norm=False, verbose=True, **save_args):                        
    # Parameter error check
    alpha = float(alpha)
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError('Alpha must be a value between 0 and 1')
    # Begin network propagation
    starttime=time.time()
    if verbose:
        print('Performing network propagation with alpha:', alpha)
    # Separate network into connected components and calculate propagation values of each sub-sample on each connected component
    #subgraphs = list(nx.connected_component_subgraphs(network))
    A = (network.subgraph(c) for c in nx.connected_components(network))
    subgraphs = list(A)

    # Initialize propagation results by propagating first subgraph
    subgraph = subgraphs[0]
    subgraph_nodes = list(subgraph.nodes)
    prop_data_node_order = list(subgraph_nodes)
    #binary_matrix_filt = np.array(binary_matrix.T.loc[subgraph_nodes].fillna(0).T)
    binary_matrix_filt = np.array(binary_matrix.T.reindex(columns=subgraph_nodes).fillna(0).T)
    #binary_matrix_filt = np.array(binary_matrix.T.fillna(0).T)
    subgraph_norm = normalize_network(subgraph, symmetric_norm=symmetric_norm)
    prop_data_empty = np.zeros((binary_matrix_filt.shape[0], 1))
    prop_data = fast_random_walk(alpha, binary_matrix_filt, subgraph_norm, prop_data_empty)
    # Get propagated results for remaining subgraphs
    for subgraph in subgraphs[1:]:
        subgraph_nodes = list(subgraph.nodes)
        prop_data_node_order = prop_data_node_order + subgraph_nodes
        #binary_matrix_filt = np.array(binary_matrix.T.loc[subgraph_nodes].fillna(0).T)
        binary_matrix_filt = np.array(binary_matrix.T.reindex(columns=subgraph_nodes).fillna(0).T)
        subgraph_norm = normalize_network(subgraph, symmetric_norm=symmetric_norm)
        prop_data = fast_random_walk(alpha, binary_matrix_filt, subgraph_norm, prop_data)
    # Return propagated result as dataframe
    prop_data_df = pd.DataFrame(data=prop_data[:,1:], index = binary_matrix.index, columns=prop_data_node_order)
    if verbose:
        print('Network Propagation Complete:', time.time()-starttime, 'seconds')              
    return prop_data_df

# Wrapper for propagating binary mutation matrix over network by subgraph given network propagation kernel
# The network propagation kernel can be pre-computed using the network_propagation function and a identity matrix data frame of the network
# Pre-calculating the kernel for many runs of NBS saves a significant amount of time
def network_kernel_propagation(network, network_kernel, binary_matrix, verbose=False, **save_args):
    if verbose:
        print('Performing network propagation with network kernel')
    # Separate network into connected components and calculate propagation values of each sub-sample on each connected component
    subgraph_nodelists = list(nx.connected_components(network))
    # Initialize propagation results by propagating first subgraph
    prop_nodelist = list(subgraph_nodelists[0])
    prop_data = np.dot(binary_matrix.T.loc[prop_nodelist].fillna(0).T, 
                       network_kernel.loc[prop_nodelist][prop_nodelist])
    # Get propagated results for remaining subgraphs
    for nodelist in subgraph_nodelists[1:]:
        subgraph_nodes = list(nodelist)
        prop_nodelist = prop_nodelist + subgraph_nodes
        subgraph_prop_data = np.dot(binary_matrix.T.loc[subgraph_nodes].fillna(0).T, 
                                    network_kernel.loc[subgraph_nodes][subgraph_nodes])
        prop_data = np.concatenate((prop_data, subgraph_prop_data), axis=1)
    # Return propagated result as dataframe
    prop_data_df = pd.DataFrame(data=prop_data, index = binary_matrix.index, columns=prop_nodelist)
    return prop_data_df

import scipy.stats as stats
def standarize_dataframe(data, with_mean = True):

    # scaler = StandardScaler(with_mean=with_mean)
    # scaler.fit(df.values.reshape(-1,1))
    # for col in df.columns:
    #     df.loc[:, col] = np.tanh(scaler.transform(df.loc[:, col].values.reshape(-1,1)))
    
    df = data.T
    df_out = df.copy(deep=True)
    dic = {}
    # Sort each gene's propagation value for each patient
    for col in df:
        dic.update({col:sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    # Rank averages for each gene across samples
    ranked_avgs = sorted_df.mean(axis = 1).tolist()
    # Construct quantile normalized Pandas DataFrame by assigning ranked averages to ranks of each gene for each sample
    for col in df_out:
        t = stats.rankdata(df[col]).astype(int)
        df_out[col] = [ranked_avgs[i-1] for i in t]
    qnorm_data = df_out.T

    return qnorm_data

#=====================transynergy=========================
import torch.nn.functional as F
import math

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        #self.input_linear = nn.Linear(d_input, d_model)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_input, d_model, heads, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(d_input, d_model)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask=None, trg_mask=None):
        x = F.relu(self.input_linear(x))
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x




def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class OutputFeedForward(nn.Module):

    def __init__(self, H, W, d_layers = None, dropout=0.2):

        super().__init__()

        self.d_layers = [512, 1] if d_layers is None else d_layers
        # self.linear_1 = nn.Linear(H*W, self.d_layers[0])
        self.linear_1 = nn.Linear(H, self.d_layers[0])
        self.n_layers = len(self.d_layers)
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(1, self.n_layers))
        self.layers = nn.ModuleList(nn.Linear(d_layers[i-1], d_layers[i]) for i in range(1, self.n_layers))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.linear_1(x)
        for i in range(self.n_layers-1):
            x = self.dropouts[i](F.relu(x))
            x = self.layers[i](x)
        
        x = self.sigmoid(x)
        return x

def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

class Mapping():
    
    def __init__(self, items):        
        self.item2idx={}
        self.idx2item=[]
        
        for idx, item in enumerate(items):
            self.item2idx[item]=idx
            self.idx2item.append(item)
            
    def add(self,item):
        if item not in self.idx2item:
            self.idx2item.append(item)
            self.item2idx[item]=len(self.idx2item)-1

if __name__ == "__main__":
    # configuration_from_json()
    test = smile_to_graph('CC[C@H](C)[C@H](NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@H](CC(O)=O)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(N)=N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](N)CC1=CC=CC=C1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CCC(O)=O)C(=O)N[C@@H](CC1=CC=C(O)C=C1)C(=O)N[C@@H](CC(C)C)C(O)=O')

    #print(test)