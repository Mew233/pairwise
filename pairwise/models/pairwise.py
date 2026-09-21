"""
Our implementation of PAIRWISE is based on multi-modal transformers, with a dual-encoder architecture that processes drug and cell features separately before combining them for synergy prediction. The model is designed to handle various types of input features, including drug-target interactions, chemical fingerprints, and cell line transcriptomic profiles.
"""
import torch
import torch.nn as nn
from ..utilitis import EncoderLayer, DecoderLayer, OutputFeedForward, Norm
import copy
import numpy as np
from torch.nn import functional as F


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


class Pairwise(nn.Module):
    def __init__(self, d_input, d_model, n_feature_type, N, heads, dropout):
        super().__init__()
        self.ae = AE(d_input, d_model)
        self.reduction2 = nn.Linear(3285, d_model, bias=True)
        self.reduction3 = nn.Linear(d_input, d_model, bias=True)

        self.encoder = Encoder(d_model=d_model, N=N, heads=heads, dropout=dropout)
        self.encoder2 = Encoder(d_model=d_model, N=N, heads=heads, dropout=dropout)

        input_length = n_feature_type * (d_model + d_model)
        self.out = OutputFeedForward(input_length, n_feature_type, d_layers=[512, 1])

    def forward(self, dts, fps, cell, src_mask=None):
        if dts.dim() == 2:
            dts = dts.unsqueeze(0)
        if fps.dim() == 2:
            fps = fps.unsqueeze(0)
        if cell.dim() == 1:
            cell = cell.unsqueeze(0)

        fp_tokens = torch.stack([self.reduction2(fps[:, i, :]) for i in range(fps.size(1))], dim=1)
        cell_token = self.ae.encode(cell).unsqueeze(1)
        coarse = torch.cat([fp_tokens, cell_token], dim=1)
        e1 = self.encoder(coarse)
        flat1 = e1.reshape(e1.size(0), -1)

        gene = torch.stack([
            self.reduction3(dts[:, 0, :]),
            self.reduction3(dts[:, 1, :]),
            self.reduction3(cell),
        ], dim=1)
        e2 = self.encoder2(gene)
        flat2 = e2.reshape(e2.size(0), -1)

        return self.out(torch.cat([flat1, flat2], dim=1))


# Backward-compatible alias.
Paiwise = Pairwise


# ---- High-level convenience methods (added for COMPASS-style API) ----

import pandas as pd
from torch.utils.data import DataLoader as TorchDataLoader


@torch.no_grad()
def _predict_from_tensors(model, dts_list, fps_list, cell_list, device="cpu", batch_size=256):
    """Batched inference over pre-assembled feature lists.

    Returns (probabilities_array, indices_array).
    """
    model = model.to(device)
    model.eval()
    predictions = []
    n = len(dts_list)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_dts = torch.stack(dts_list[start:end]).to(device)
        batch_fps = torch.stack(fps_list[start:end]).to(device)
        batch_cell = torch.stack(cell_list[start:end]).to(device)
        out = model(batch_dts, batch_fps, batch_cell)
        out = out.squeeze(-1).cpu().numpy()
        predictions.extend(out.tolist())
    return np.array(predictions)


def predict(self, meta_df, features, device="cpu", batch_size=256):
    """Predict synergy probabilities for a set of drug–drug–cell triples.

    Mirrors ``compass.loadcompass().predict(df_tpm)``.  Accepts the feature
    format produced by ``load_features`` in the example notebook:

        meta, T = load_features('predict.npz', 'predict_meta.csv')
        model = loadpairwise('best_model_pairwise.pth')
        probs, df_pred = model.predict(meta, T)

    Parameters
    ----------
    meta_df : pandas.DataFrame
        Metadata with columns ``drug1``, ``drug2``, ``cell`` (or
        ``drug_row``, ``drug_col``, ``DepMap_ID``).
    features : tuple or dict
        Either a ``(dts_array, fps_array, cell_array)`` tuple of numpy arrays,
        or the ``T`` object returned by ``load_features``.
    device : str or torch.device
        Compute device.
    batch_size : int
        Inference batch size.

    Returns
    -------
    probs : numpy.ndarray
        Synergy probability for each row.
    df_pred : pandas.DataFrame
        Same rows as ``meta_df`` with an added ``prediction`` column.
    """
    # Unpack features
    if isinstance(features, dict):
        dts = features["dts"]
        fps = features["fps"]
        cell = features["cell"]
    elif isinstance(features, (tuple, list)) and len(features) == 3:
        dts, fps, cell = features
    else:
        raise ValueError(
            "features must be a (dts, fps, cell) tuple or a dict with those keys"
        )

    # Convert to tensor lists
    dts_t = [torch.from_numpy(np.array(d)).float() for d in dts]
    fps_t = [torch.from_numpy(np.array(f)).float() for f in fps]
    cell_t = [torch.from_numpy(np.array(c)).float() for c in cell]

    probs = _predict_from_tensors(self, dts_t, fps_t, cell_t, device, batch_size)

    df_pred = meta_df.copy()
    df_pred["prediction"] = probs
    return probs, df_pred


@torch.no_grad()
def extract(self, meta_df, features, device="cpu", batch_size=256):
    """Extract intermediate representations from a trained PAIRWISE model.

    Mirrors ``compass.loadcompass().extract(df_tpm)``.  Returns four
    feature matrices:

    - ``df_cell``: transcriptome autoencoder embeddings
    - ``df_fp``: chemical token representations
    - ``df_e1``: coarse-grained dual-encoder features
    - ``df_e2``: gene-level encoder features

    These can be used for downstream tasks such as building a Cox model
    for survival prediction or a logistic regression for response.
    """
    self = self.to(device)
    self.eval()

    if isinstance(features, dict):
        dts = features["dts"]
        fps = features["fps"]
        cell = features["cell"]
    elif isinstance(features, (tuple, list)) and len(features) == 3:
        dts, fps, cell = features
    else:
        raise ValueError("features must be a (dts, fps, cell) tuple or dict")

    cell_emb, fp_emb, e1_emb, e2_emb = [], [], [], []
    n = len(dts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_dts = torch.stack([
            torch.from_numpy(np.array(d)).float() for d in dts[start:end]
        ]).to(device)
        batch_fps = torch.stack([
            torch.from_numpy(np.array(f)).float() for f in fps[start:end]
        ]).to(device)
        batch_cell = torch.stack([
            torch.from_numpy(np.array(c)).float() for c in cell[start:end]
        ]).to(device)

        # Autoencoder embedding
        cell_encoded = self.ae.encode(batch_cell)
        cell_emb.append(cell_encoded.cpu().numpy())

        # Chemical tokens (after reduction)
        fp_tokens = torch.stack(
            [self.reduction2(batch_fps[:, i, :]) for i in range(batch_fps.size(1))],
            dim=1,
        )
        fp_emb.append(fp_tokens.cpu().numpy())

        # Coarse encoder (encoder1)
        coarse = torch.cat([fp_tokens, cell_encoded.unsqueeze(1)], dim=1)
        e1 = self.encoder(coarse)
        e1_flat = e1.reshape(e1.size(0), -1)
        e1_emb.append(e1_flat.cpu().numpy())

        # Gene encoder (encoder2)
        gene = torch.stack(
            [self.reduction3(batch_dts[:, 0, :]),
             self.reduction3(batch_dts[:, 1, :]),
             self.reduction3(batch_cell)],
            dim=1,
        )
        e2 = self.encoder2(gene)
        e2_flat = e2.reshape(e2.size(0), -1)
        e2_emb.append(e2_flat.cpu().numpy())

    df_cell = pd.DataFrame(np.concatenate(cell_emb), index=meta_df.index)
    fp_concat = np.concatenate(fp_emb, axis=0)
    fp_2d = fp_concat.reshape(fp_concat.shape[0], -1)
    df_fp = pd.DataFrame(fp_2d, index=meta_df.index)
    df_e1 = pd.DataFrame(np.concatenate(e1_emb), index=meta_df.index)
    df_e2 = pd.DataFrame(np.concatenate(e2_emb), index=meta_df.index)
    return df_cell, df_fp, df_e1, df_e2


# Bind methods to the Pairwise class
Pairwise.predict = predict
Pairwise.extract = extract
