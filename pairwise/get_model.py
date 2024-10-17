from models.baselines import *
from models.deepsynergy_preuer import *
from models.multitaskdnn_kim import *
from models.matchmaker_brahim import *
from models.deepdds_wang import *
from models.TGSynergy import *
from models.transynergy_liu import *
from models.graphsynergy import *
from models.granuality import *
from models.pairwise import *

import argparse

def get_model(model_name,*args):

    base_model = baseline()
    if model_name in ['LR','XGBOOST','RF','ERT']:

        return getattr(base_model, model_name)()

    if model_name == "deepsynergy_preuer":

        return Deepsynergy_Preuer(channels=args[0],dropout_rate = 0.5)
    
    if model_name == "transynergy_liu":

        return Transynergy_Liu(d_input=3645, d_model=256, n_feature_type=1, N=1, heads=4, dropout=0.2)

    if model_name == "pairwise":

        return Pairwise(d_input=4098, d_model=256, n_feature_type=3, N=1, heads=4, dropout=0.2)
    
    if model_name == "multitaskdnn_kim":

        return Multitaskdnn_Kim(cell_channels=args[0],\
            drug_fp_channels=args[1],drug_tg_channels=args[2],dropout_rate = 0.5)
    
    if model_name == "matchmaker_brahim":

        return MatchMaker_Brahim(cell_channels=args[0],drug_channels=args[1],dropout_rate = 0.5)
    
    if model_name == "deepdds_wang":
        return DeepDDS_Wang()

    if model_name == "TGSynergy":

        return TGSynergy(cluster_predefine=args[0])

    if model_name == "graphsynergy":
        return Graphsynergy(graph=args[0], dpi_dict=args[1], cpi_dict=args[2])


if __name__ == "__main__":
    get_model('deepsynergy_preuer')