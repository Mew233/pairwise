"""Single source of truth for the random seed.

``--seed`` used to be accepted on the command line but ignored, because every
split and weight initialisation hard-coded 42. Everything now routes through
``get_seed(args)`` so a run is reproducible from its recorded seed alone.

"""
import os
import random

import numpy as np
import torch

DEFAULT_SEED = int(os.environ.get("PAIRWISE_SEED", 42))


def get_seed(args=None):
    if args is None:
        return DEFAULT_SEED
    seed = getattr(args, "seed", None)
    return DEFAULT_SEED if seed is None else int(seed)


def seed_everything(seed):
    """Seed all RNG sources and force deterministic cuDNN.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
