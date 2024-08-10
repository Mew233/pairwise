"""
[Liu Q, Xie L (2021)]TranSynergy: mechanism-driven interpretable deep neural network for the 
synergistic prediction and pathway deconvolution of drug combinations. PLoS Comput Biol 17(2):653
"""
import torch.nn as nn
from utilitis import EncoderLayer, DecoderLayer, Norm, OutputFeedForward, dice
import copy
from torch import flatten
import torch.nn.functional as F
from torch import cat, stack
import numpy as np
from models.TGSynergy import GNN_drug
import torch 
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..','..'))
save_path = os.path.join(ROOT_DIR, 'data','cell_line_data','tcga','tcga_encoder.pth')

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class AE(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = None, \
        dop: float = 0.1, noise_flag: bool = False, **kwargs) -> None:
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        self.noise_flag = noise_flag
        self.dop = dop

        if hidden_dims is None:
            hidden_dims = [512]

        # build encoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                #nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.dop)
                )
            )
        modules.append(nn.Dropout(self.dop))
        modules.append(nn.Linear(hidden_dims[-1], latent_dim, bias=True))

        self.encoder = nn.Sequential(*modules)

        # build decoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[-1], bias=True),
                #nn.BatchNorm1d(hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.dop)
                )
            )
            
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1], bias=True),
            #nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(self.dop),
            nn.Linear(hidden_dims[-1], input_dim)
        )
    
    def forward(self, input):
        encoded_input = self.encoder(input)
        encoded_input = nn.functional.normalize(encoded_input, p=2, dim=1)
        output = self.final_layer(self.decoder(encoded_input))

        return output

    def encode(self, input):
        return self.encoder(input)

    def decode(self, z):
        return self.decoder(z)

class Transynergy_Liu(nn.Module):
    def __init__(self, d_input, d_model, n_feature_type, N, heads, dropout):
        super().__init__()
        self.ae = AE(4298,d_model)#2734
        self.ae.load_state_dict(torch.load(save_path))

        self.reduction = nn.Linear(d_input, d_model, bias=True)
        self.reduction2 = nn.Linear(3285, d_model, bias=True)

        self.encoder = Encoder(d_model, N, heads, dropout)
        # self.decoder = Decoder(d_input, d_model, N, heads, dropout)


        input_length = 1280 #1280 #768+244*2 #256/
        self.out = OutputFeedForward(input_length, n_feature_type, d_layers=[64, 32, 1])


    def forward(self, src, fp=None, sm1=None, sm2=None, \
        trg=None, src_mask=None, trg_mask=None):
        

        # # 1) Default
        # e_outputs = self.encoder(src, src_mask)
        # flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))
        # output = self.out(flat_e_output)
        
        ## 2) 5 layers
        _src = self.reduction(src)
        _fp = self.reduction2(fp)
        _cell = self.ae.encode(sm1)
        _cell = torch.unsqueeze(_cell, dim=1)
        cat_input = cat((_src,_fp,_cell), dim=1)
        
        e_outputs = self.encoder(cat_input, src_mask)
        flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))
        output = self.out(flat_e_output)

        return output

