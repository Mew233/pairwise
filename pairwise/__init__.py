"""PAIRWISE: prediction of effective personalized drug combinations in cancer.

This package provides the PAIRWISE synergy-prediction model, seven benchmarked
baselines, and the feature pipelines that feed them.  The public API mirrors
the COMPASS (Shen et al., Nature Medicine) pattern:

    from pairwise import loadpairwise, Trainer

    # Inference
    model = loadpairwise('best_model_pairwise.pth', map_location='cpu')
    probs, df_pred = model.predict(meta_df, features)

    # Training from scratch
    trainer = Trainer(lr=1e-4, epochs=50)
    model, ckpt, results = trainer.train()
"""

__version__ = "1.1.0"

from . import paths
from .paths import CONFIG_DIR, DATA_DIR, RESULTS_DIR, WEIGHTS_DIR
from .seeding import DEFAULT_SEED, get_seed, seed_everything
from .api import loadpairwise, Trainer

__all__ = [
    "__version__",
    "paths",
    "CONFIG_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "WEIGHTS_DIR",
    "DEFAULT_SEED",
    "get_seed",
    "seed_everything",
    "loadpairwise",
    "Trainer",
]
