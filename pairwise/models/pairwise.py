import torch.nn as nn
from utilitis import Layer, Norm, OutputFeedForward, dice
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

class DTI_CNN(nn.Module):
    def __init__(self):
        super(DTI_CNN, self).__init__()
        # Defining convolutional layers to condense 3,645 input features into a smaller number of features
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)

        # Calculate the output size after convolutions to correctly size the fully connected layer
        self.flattened_size = 128 * 114 * 2  # Adjusted based on the actual number of output features after the three convolutions

        # Fully connected layer to get output of size [batch_size, 2, 256]
        self.fc1 = nn.Linear(self.flattened_size, 512)  # Reduce feature size
        self.fc2 = nn.Linear(512, 2 * 256)  # Produce the final output in the correct format

    def forward(self, x):
        # Apply convolutional layers with activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layers to get the final output size [batch_size, 2, 256]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 2, 256)
        return x


class Pairwise(nn.Module):
    def __init__(self, d_input, d_model, n_feature_type, N, heads, dropout):
        super().__init__()
        self.ae = AE(4298,d_model)#2734
        self.ae.load_state_dict(torch.load(save_path))

        self.cnn = DTI_CNN()
        self.reduction2 = nn.Linear(3285, d_model, bias=True)
        self.atte = Encoder(d_model, N, heads, dropout)

        input_length = 1280 #1280 #768+244*2 #256/
        self.out = OutputFeedForward(input_length, n_feature_type, d_layers=[64, 32, 1])


    def forward(self, src, fp, cell):

        _src = self.cnn(src)
        _fp = self.reduction2(fp)
        _cell = self.ae.encode(cell)
        _cell = torch.unsqueeze(_cell, dim=1)
        cat_input = cat((_src,_fp,_cell), dim=1)
        
        e_outputs = self.atte(cat_input)
        flat_e_output = e_outputs.view(-1, e_outputs.size(-2)*e_outputs.size(-1))
        output = self.out(flat_e_output)

        return output

