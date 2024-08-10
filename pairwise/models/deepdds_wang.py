"""
    [Wang et al., 2021a] Jinxian Wang, Xuejun Liu, Siyuan Shen, Lei Deng, and Hui Liu. DeepDDS: Deep Graph Neural Network 
    with Attention Mechanism to Predict Synergistic Drug Combinations.  Briefings in Bioinformatics, 09 2021.
"""
import torch
import torch.nn as nn
from torchdrug.layers import MLP, MaxReadout
from torchdrug.models import GraphConvolutionalNetwork
from torch.nn.functional import normalize
from torch_geometric.nn import GCNConv, global_max_pool
import numpy as np

class DeepDDS_Wang(nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=1000, output_dim=128, dropout=0.2):

        super(DeepDDS_Wang, self).__init__()

        self.relu = nn.ReLU()
        self.prelu = nn.PReLU(num_parameters=1, init=0.75)
        self.dropout = nn.Dropout(dropout)
        # SMILES1 graph branch
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.drug1_conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*4, num_features_xd*2)
        self.drug1_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)

        # SMILES2 graph branch
        self.drug2_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug2_conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.drug2_conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 4, num_features_xd*2)
        self.drug2_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)


        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # combined layers
        self.fc1 = nn.Linear(3*output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)
        self.out2 = nn.Sigmoid()

    def forward(self,x1, edge_index1, x2, edge_index2, cell, batch1, batch2):
        x1, edge_index1, x2, edge_index2, cell, batch1, batch2 = x1, edge_index1, x2, edge_index2, cell, batch1, batch2
        
        # Run the MLP forward for the cell line features
        #x, edge_index = data.x, data.edge_index
        #x = self.conv1(x, edge_index)

        #cell_out = self.cell_mlp(normalize(torch.FloatTensor(cell), p=2, dim=1))

            # deal drug1
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = global_max_pool(x1, batch1)       # global max pooling

        # flatten
        x1 = self.relu(self.drug1_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.drug1_fc_g2(x1)
        x1 = self.dropout(x1)
        # print('x1.shape', x1.shape)
        # print('x1', x1[0])


        # deal drug2
        x2 = self.drug1_conv1(x2, edge_index2)
        x2 = self.relu(x2)

        x2 = self.drug1_conv2(x2, edge_index2)
        x2 = self.relu(x2)

        x2 = self.drug1_conv3(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = global_max_pool(x2, batch2)  # global max pooling

        # flatten
        x2 = self.relu(self.drug1_fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.drug1_fc_g2(x2)
        x2 = self.dropout(x2)
        # print('x2.shape', x2.shape)
        # print('x', x2[0])

        # deal cell
        cell = np.stack(cell, axis=1)
        cell = torch.FloatTensor(cell)
        cell_vector = self.reduction(cell)

        # concat
        xc = torch.cat((x1, x2, cell_vector), 1)
        xc = normalize(xc, 2, 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.prelu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.prelu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.out2(out)
        return out

    