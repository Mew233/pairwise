"""
    [Brahim et al., 2021] Kuru Halil Brahim, Oznur Tastan, and Ercument Cicek. MatchMaker: A Deep Learning Framework 
    for Drug Synergy Prediction. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 2021.
"""
import torch
import torch.nn as nn


class MatchMaker_Brahim(nn.Module):
    def __init__(self,
        cell_channels: int,
        drug_channels: int,
        dropout_rate = 0.5     
    ):

        super(MatchMaker_Brahim, self).__init__()

        #: Applied to the left+context and right+context separately
        self.drug_context_layer = nn.Sequential(
            nn.Linear(drug_channels + cell_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 32),
        )
        # Applied to the concatenated left/right tensors
        self.final = nn.Sequential(
            nn.Linear(2 * 32, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            #nn.Sigmoid(),
        )

        self.classify = nn.Sigmoid()

    def forward(self, fp_drug, fp_drug2, cell):
        # fp_drug, fp_drug2, cell = inputs[0], inputs[1], inputs[2]
        
        # The left drug
        hidden_left = torch.cat([cell, fp_drug], dim=1)
        hidden_left = self.drug_context_layer(hidden_left)

        # The right drug
        hidden_right = torch.cat([cell, fp_drug2], dim=1)
        hidden_right = self.drug_context_layer(hidden_right)

        hidden = torch.cat([hidden_left, hidden_right], dim=1)
        hidden_rev = torch.cat([hidden_right, hidden_left], dim=1)

        hidden = self.final(hidden)
        hidden_rev = self.final(hidden_rev)

        return self.classify((hidden+hidden_rev)/2)