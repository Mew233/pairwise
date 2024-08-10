"""
    [Yang et al., 2021] Yang, Jiannan, Zhongzhi Xu, William Ka Kei Wu, Qian Chu, and Qingpeng Zhang. "GraphSynergy: a network-inspired deep learning 
    model for anticancer drug combination prediction." Journal of the American Medical Informatics Association 28, no. 11 (2021): 2336-2345.
"""
import numpy as np
import torch
import torch.nn as nn
import collections
import torch.nn.functional as F

def get_neighbor_set(items, item_target_dict, graph):
    # print('constructing neighbor set ...')
    n_memory = 128

    neighbor_set = collections.defaultdict(list)
    neighbor_set_virtual = collections.defaultdict(list)

    nodes_dict = dict(zip(graph.nodes(),range(len(graph.nodes()))))

    for item in items:
        for hop in range(2):
            # use the target directly
            if hop == 0:
                replace = len(item_target_dict[item]) < n_memory
                _target_list = list(np.random.choice(item_target_dict[item], size=n_memory, replace=replace))
            else:
                # use the last one to find k+1 hop neighbors
                origin_nodes = neighbor_set[item][-1]
                neighbors = []
                for node in origin_nodes:
                    neighbors += graph.neighbors(node)
                # sample
                replace = len(neighbors) < n_memory
                _target_list = list(np.random.choice(neighbors, size=n_memory, replace=replace))
            
            #map protein ID to a virtual index
            neighbor_set[item].append(_target_list)
            target_list = [nodes_dict[k] for k in _target_list]
            
            neighbor_set_virtual[item].append(target_list)

    return neighbor_set_virtual
    # return neighbor_set
    


class Graphsynergy(nn.Module):
    def __init__(self, graph, dpi_dict, cpi_dict):
        super().__init__()
        
        self.graph =  graph
        self.drug_protein_dict = dpi_dict
        self.cell_protein_dict = cpi_dict

        self.protein_num = graph.number_of_nodes()
        self.emb_dim = 64
        self.n_hop = 2

        self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)
        # self.protein_embedding = nn.Embedding(int(max(graph.nodes())), self.emb_dim)
        self.drug_embedding = nn.Embedding(max(dpi_dict.keys()), self.emb_dim)
        self.cell_embedding = nn.Embedding(int(max(cpi_dict.keys())), self.emb_dim)
        self.protein_embedding.weight.requires_grad = True
        self.drug_embedding.weight.requires_grad = True
        self.cell_embedding.weight.requires_grad = True

        self.combine_function = nn.Linear(self.emb_dim*2, self.emb_dim, bias=False)
        # self.combine_function = nn.Linear(in_features=2, out_features=1, bias=False)

        self.aggregation_function = nn.Linear(self.emb_dim*self.n_hop, self.emb_dim)
        self.out = nn.Sigmoid()
        
        self.dropout_ratio = 0.2
        self.regression_classify = nn.Sequential(
            nn.Linear(192, 64),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),  
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def _get_neighbor_emb(self, items, neighbors):
        

        all_list = []
        neighbors_emb_list_hp1 = []
        neighbors_emb_list_hp2 = []
        for item in items:
            _neigh = neighbors[item]
            for hop in range(self.n_hop):
                if hop == 0:
                    neighbors_emb_list_hp1.append(self.protein_embedding(torch.LongTensor(_neigh[hop])))
                else:
                    neighbors_emb_list_hp2.append(self.protein_embedding(torch.LongTensor(_neigh[hop])))
            
        all_list.append(torch.stack(neighbors_emb_list_hp1))
        all_list.append(torch.stack(neighbors_emb_list_hp2))

         # [hop, batch_size, n_memory, dim]
        return all_list

    def _interaction_aggregation(self, item_embeddings, neighbors_emb_list):
        interact_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim]
            neighbor_emb = neighbors_emb_list[hop]
            # [batch_size, dim, 1]
            item_embeddings_expanded = torch.unsqueeze(item_embeddings, dim=2)
            # [batch_size, n_memory]
            contributions = torch.squeeze(torch.matmul(neighbor_emb,
                                                        item_embeddings_expanded))
            # [batch_size, n_memory]
            contributions_normalized = F.softmax(contributions, dim=1)
            # [batch_size, n_memory, 1]
            contributions_expaned = torch.unsqueeze(contributions_normalized, dim=2)
            # [batch_size, dim]
            i = (neighbor_emb * contributions_expaned).sum(dim=1)
            # update item_embeddings
            item_embeddings = i
            interact_list.append(i)
        return interact_list



    def _aggregation(self, item_i_list):
        # [batch_size, n_hop+1, emb_dim]
        item_i_concat = torch.cat(item_i_list, 1)
        # [batch_size, emb_dim]
        self.aggregation_function = nn.Linear(self.emb_dim*self.n_hop, self.emb_dim)
        item_embeddings = self.aggregation_function(item_i_concat)
        return item_embeddings
            
    def _therapy(self, drug1_embeddings, drug2_embeddings, cell_embeddings):
        
        combined_durg = self.combine_function(torch.cat([drug1_embeddings, drug2_embeddings], dim=1))
        therapy_score = (combined_durg * cell_embeddings).sum(dim=1)
        # drug1_score = torch.unsqueeze((drug1_embeddings * cell_embeddings).sum(dim=1), dim=1)
        # drug2_score = torch.unsqueeze((drug2_embeddings * cell_embeddings).sum(dim=1), dim=1)
        # therapy_score = torch.squeeze(self.combine_function(torch.cat([drug1_score, drug2_score], dim=1)))
        return therapy_score

    def _toxic(self, drug1_embeddings, drug2_embeddings):
        return (drug1_embeddings * drug2_embeddings).sum(dim=1)

    def _emb_loss(self, cell_embeddings, drug1_embeddings, drug2_embeddings, 
                  cell_neighbors_emb_list, drug1_neighbors_emb_list, drug2_neighbors_emb_list):
        item_regularizer = (torch.norm(cell_embeddings) ** 2
                          + torch.norm(drug1_embeddings) ** 2
                          + torch.norm(drug2_embeddings) ** 2) / 2
        node_regularizer = 0
        for hop in range(self.n_hop):
            node_regularizer += (torch.norm(cell_neighbors_emb_list[hop]) ** 2
                              +  torch.norm(drug1_neighbors_emb_list[hop]) ** 2
                              +  torch.norm(drug2_neighbors_emb_list[hop]) ** 2) / 2
        
        emb_loss = 1e-6 * (item_regularizer + node_regularizer) / cell_embeddings.shape[0]

        return emb_loss
    

    def forward(self, drug1, drug2, cell):

        drug1_items = drug1.int().view(-1).tolist()
        drug2_items = drug2.int().view(-1).tolist()

        drug1_neighbors = get_neighbor_set(items=drug1_items, item_target_dict=self.drug_protein_dict, graph=self.graph)
        drug2_neighbors = get_neighbor_set(items=drug2_items, item_target_dict=self.drug_protein_dict, graph=self.graph)

        drug1_neighbors_emb_list = self._get_neighbor_emb(drug1_items, drug1_neighbors)
        drug2_neighbors_emb_list = self._get_neighbor_emb(drug2_items, drug2_neighbors)

        drug1_embeddings = self.drug_embedding(torch.LongTensor(np.array(drug1_items)))
        drug2_embeddings = self.drug_embedding(torch.LongTensor(np.array(drug2_items)))

        drug1_i_list = self._interaction_aggregation(drug1_embeddings, drug1_neighbors_emb_list)
        drug2_i_list = self._interaction_aggregation(drug2_embeddings, drug2_neighbors_emb_list)

        drug1_embeddings = self._aggregation(drug1_i_list)
        drug2_embeddings = self._aggregation(drug2_i_list)

        ## cell embedding
        cell_items = cell.int().view(-1).tolist()
        cell_embeddings = self.cell_embedding(torch.LongTensor(np.array(cell_items)))

        cell_neighbors = get_neighbor_set(items=cell_items, item_target_dict=self.cell_protein_dict, graph=self.graph)
        cell_neighbors_emb_list = self._get_neighbor_emb(cell_items, cell_neighbors)
        cell_i_list = self._interaction_aggregation(cell_embeddings, cell_neighbors_emb_list)
        cell_embeddings = self._aggregation(cell_i_list)


        score = self._therapy(drug1_embeddings, drug2_embeddings, cell_embeddings) - \
        self._toxic(drug1_embeddings, drug2_embeddings)
        x = self.out(score.unsqueeze(1))
        

                # embedding loss
        emb_loss = self._emb_loss(cell_embeddings, drug1_embeddings, 
                                  drug2_embeddings, cell_neighbors_emb_list,
                                  drug1_neighbors_emb_list, drug2_neighbors_emb_list)

        # x = torch.cat([drug1_embeddings, drug2_embeddings,cell_embeddings], -1)
        # x = self.regression_classify(x)

        return x, emb_loss