"""Construct a model by name.

DeepDDS needs ``torchdrug``;
TGSynergy / GraphSynergy need geometric extras) do not block loading PAIRWISE.
"""


def get_model(model_name, *args, cell_input_dim=4298):
    if model_name in ["LR", "XGBOOST", "RF", "ERT"]:
        from .models.baselines import baseline
        return getattr(baseline(), model_name)()

    if model_name == "deepsynergy_preuer":
        from .models.deepsynergy_preuer import Deepsynergy_Preuer
        return Deepsynergy_Preuer(channels=args[0], dropout_rate=0.5)

    if model_name == "transynergy_liu":
        from .models.transynergy_liu import Transynergy_Liu
        # d_input = number of drug-target genes (from drug_target matrix)
        # cell_input_dim = number of cell-line genes (from cell_filtered_by config)
        return Transynergy_Liu(
            d_input=3645, d_model=256, n_feature_type=3, N=1, heads=4, dropout=0.2,
            cell_input_dim=cell_input_dim
        )

    if model_name == "pairwise":
        from .models.pairwise import Pairwise
        return Pairwise(
            d_input=3460, d_model=256, n_feature_type=3, N=1, heads=4, dropout=0.2
        )

    if model_name == "multitaskdnn_kim":
        from .models.multitaskdnn_kim import Multitaskdnn_Kim
        return Multitaskdnn_Kim(
            cell_channels=args[0],
            drug_fp_channels=args[1],
            drug_tg_channels=args[2],
            dropout_rate=0.5,
        )

    if model_name == "matchmaker_brahim":
        from .models.matchmaker_brahim import MatchMaker_Brahim
        return MatchMaker_Brahim(
            cell_channels=args[0], drug_channels=args[1], dropout_rate=0.5
        )

    if model_name == "deepdds_wang":
        from .models.deepdds_wang import DeepDDS_Wang
        return DeepDDS_Wang()

    if model_name == "TGSynergy":
        from .models.TGSynergy import TGSynergy
        return TGSynergy(cluster_predefine=args[0])

    if model_name == "graphsynergy":
        from .models.graphsynergy import Graphsynergy
        return Graphsynergy(graph=args[0], dpi_dict=args[1], cpi_dict=args[2])

    raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    get_model("deepsynergy_preuer", 1000)
