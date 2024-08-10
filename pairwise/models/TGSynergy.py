import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, global_max_pool
from torch_geometric.nn import GATConv, max_pool

class GNN_drug(torch.nn.Module):
    def __init__(self, layer_drug=3, dim_drug=128):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()

        for i in range(self.layer_drug):
            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(77, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, drug):
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x_drug_list = []
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            x_drug_list.append(x)

        node_representation = self.JK(x_drug_list)
        x_drug = global_max_pool(node_representation, batch)
        return x_drug
    

class GNN_cell(torch.nn.Module):
    def __init__(self, num_feature=1, layer_cell=3, dim_cell=8, cluster_predefine=None):
        super().__init__()
        self.num_feature = num_feature
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.cluster_predefine = cluster_predefine
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        # self.activations = torch.nn.ModuleList()

        for i in range(self.layer_cell):
            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)
            else:
                conv = GATConv(self.num_feature, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # True or False
            # activation = nn.PReLU(self.dim_cell)

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)
            # self.activations.append(activation)

    def forward(self, cell):
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            num_node = int(cell.x.size(0) / cell.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)
        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)

        return node_representation

    def grad_cam(self, cell):
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            if i == 0:
                cell_node = cell.x
                cell_node.retain_grad()
            num_node = int(cell.x.size(0) / cell.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)

        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)

        return cell_node, node_representation


class TGSynergy(nn.Module):
    def __init__(self, cluster_predefine):
        super().__init__()
        self.layer_drug = 3
        self.dim_drug = 128
        self.num_feature = 1
        self.layer_cell = 3
        self.dim_cell = 8
        self.dropout_ratio = 0.2
        

        # drug graph branch
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug)

        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )

        # cell graph branch
        self.GNN_cell = GNN_cell(self.num_feature, self.layer_cell, self.dim_cell, cluster_predefine)

        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.GNN_cell.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )

        self.regression = nn.Sequential(
            nn.Linear(768, 512),
            #nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 1)
        )

        self.regression_classify = nn.Sequential(
            nn.Linear(768, 512),
            # nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),  
            nn.Linear(512, 1),
            nn.Sigmoid()
            
        )
        
        self.dsn = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 256)
        )

    def forward(self, drug, drug2, cell):

        #drug, drug2, cell = inputs[0], inputs[1], inputs[2]
        # forward drug
        x_drug = self.GNN_drug(drug)
        x_drug = self.drug_emb(x_drug)

        x_drug2 = self.GNN_drug(drug2)
        x_drug2 = self.drug_emb(x_drug2)

        # forward cell
        x_cell = self.GNN_cell(cell)
        x_cell = self.cell_emb(x_cell)

        x = torch.cat([x_drug, x_drug2,x_cell], -1)
        x = self.regression_classify(x)

        
        return x