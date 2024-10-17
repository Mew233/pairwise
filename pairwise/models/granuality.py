import torch
import torch.nn as nn
from utilitis import Layer, DecoderLayer, OutputFeedForward, Norm
import copy
import numpy as np
from torch.nn import functional as F
import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
save_path = os.path.join(ROOT_DIR, 'data', 'cell_line_data', 'tcga', 'tcga_encoder.pth')

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(Layer(d_model, heads, dropout), N)
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
                # nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
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
                # nn.BatchNorm1d(hidden_dims[-1]),
                nn.ReLU(),
                nn.Dropout(self.dop)
            )
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(self.dop)
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1], bias=True),
            # nn.BatchNorm1d(hidden_dims[-1]),
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


class Vt(nn.Module):
    def __init__(self, d_input, d_model, n_feature_type, N, heads, dropout):
        super().__init__()
        self.ae = AE(3460, d_model)
        # self.ae.load_state_dict(torch.load(save_path))
        # self.reduction = torch.nn.Linear(d_input,d_model,bias = True)
        self.reduction2 = nn.Linear(3285, d_model, bias=True)
        self.reduction3 = nn.Linear(d_input, d_model, bias=True)

        self.encoder = Encoder(d_model=3, N=1, heads=1, dropout=0.2)
        self.encoder2 = Encoder(d_model=d_input, N=1, heads=4, dropout=0.2)

        input_length = n_feature_type * (d_model + d_model)
        self.out = OutputFeedForward(input_length, n_feature_type, d_layers=[512, 1])

    def forward(self, dts, fps, cell, src_mask=None):
        # coarse granuality
        # _src = self.reduction(dts)
        _fp = self.reduction2(fps)
        _cell = self.ae.encode(cell)
        _cell = torch.unsqueeze(_cell, dim=1)
        cat_input = torch.cat((_fp, _cell), dim=1)

        e_outputs = self.encoder(cat_input.transpose(-2, -1), src_mask)
        # e_outputs = cat_input
        flat_e_output = e_outputs.reshape(-1, e_outputs.size(-2) * e_outputs.size(-1))

        # fine granuality
        gene_input = torch.cat((dts, torch.unsqueeze(cell, dim=1)), dim=1)
        # gene_input = self.reduction3(gene_input)

        gene_outputs = self.encoder2(gene_input, src_mask)
        # gene_outputs = self.reduction3(gene_outputs)
        flat_gene_output = gene_outputs.reshape(-1, gene_outputs.size(-2) * gene_outputs.size(-1))

        output = self.out(torch.cat((flat_e_output, flat_gene_output), dim=1))
        # output = self.out(flat_gene_output)

        return output
