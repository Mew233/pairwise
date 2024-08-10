"""
    [Kim et al., 2021] Yejin Kim, Shuyu Zheng, Jing Tang, Wenjin Jim Zheng, Zhao Li, and Xiaoqian Jiang. Anticancer Drug Synergy
    Prediction in Understudied Tissues Using Transfer Learning. Journal of the American Medical Informatics Association, 28(1):42–51, 2021.
"""
import torch
import torch.nn as nn

class Multitaskdnn_Kim(nn.Module):
    def __init__(self,
        cell_channels: int,
        drug_fp_channels: int,
        drug_tg_channels: int,
        dropout_rate = 0.5   
    ):
        super(Multitaskdnn_Kim, self).__init__()
        

        # cell  branch
        self.cell_emb = nn.Sequential(
            nn.Linear(cell_channels, 1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        # drug fingerprints branch
        self.drug_fp_emb = nn.Sequential(
            nn.Linear(drug_fp_channels, 128),
            nn.ReLU(),
        )
        # drug target branch
        self.drug_target_emb = nn.Sequential(
            nn.Linear(drug_tg_channels, 128),
            nn.ReLU(),
        )
        # combined drug
        self.combined = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # combined drug and cell
        self.combined2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, fp_drug, tg_drug, fp_drug2, tg_drug2, cell):
        #fp_drug, tg_drug, fp_drug2, tg_drug2, cell \
        # = inputs[0] (256), inputs[1](2832), inputs[2](256), inputs[3](2832), inputs[4](1000)
        # forward drug
        x_fp_drug = self.drug_fp_emb(fp_drug)
        x_tg_drug = self.drug_target_emb(tg_drug)

        x_fp_drug2 = self.drug_fp_emb(fp_drug2)
        x_tg_drug2 = self.drug_target_emb(tg_drug2)


        # forward cell
        x_cell = self.cell_emb(cell)

        # combine drug1 features
        x_1 = torch.cat([x_fp_drug, x_tg_drug], -1)
        x_1 = self.combined(x_1)

        # combine drug2 features
        x_2 = torch.cat([x_fp_drug2, x_tg_drug2], -1)
        x_2 = self.combined(x_2)

        # combine drugs and cell features
        x_3 = torch.cat([x_1, x_cell, x_2], -1)
        x_3 = self.combined2(x_3)

        return x_3
