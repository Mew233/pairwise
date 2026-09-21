"""Filesystem roots for data, results, weights and configs.

Every location is resolved at import time from an environment variable and
falls back to a directory under the current working directory, so an installed
copy of the package never reads from inside ``site-packages``.

    PAIRWISE_HOME         working root that holds data/, results/, weights/
    PAIRWISE_DATA_ROOT    input data       (default $PAIRWISE_HOME/data)
    PAIRWISE_RESULTS_DIR  written outputs  (default $PAIRWISE_HOME/results)
    PAIRWISE_WEIGHTS_DIR  checkpoints      (default $PAIRWISE_HOME/weights)
    PAIRWISE_CONFIG_DIR   model configs    (default the configs/ shipped here)
"""
import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(env_var, default):
    return os.path.abspath(os.environ.get(env_var) or default)


ROOT_DIR = _resolve("PAIRWISE_HOME", os.getcwd())
DATA_DIR = _resolve("PAIRWISE_DATA_ROOT", os.path.join(ROOT_DIR, "data"))
RESULTS_DIR = _resolve("PAIRWISE_RESULTS_DIR", os.path.join(ROOT_DIR, "results"))
WEIGHTS_DIR = _resolve("PAIRWISE_WEIGHTS_DIR", os.path.join(ROOT_DIR, "weights"))
CONFIG_DIR = _resolve("PAIRWISE_CONFIG_DIR", os.path.join(_PKG_DIR, "configs"))


def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)


def results_path(*parts):
    """Path under the results directory, creating parent folders on demand."""
    path = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(os.path.dirname(path) or RESULTS_DIR, exist_ok=True)
    return path


def weights_path(*parts):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    return os.path.join(WEIGHTS_DIR, *parts)


def config_path(model_name):
    return os.path.join(CONFIG_DIR, f"config_{model_name}.json")


def describe():
    """Human-readable dump of the resolved roots, for run logs."""
    return "\n".join(
        f"  {name:<12} {value}"
        for name, value in [
            ("home", ROOT_DIR),
            ("data", DATA_DIR),
            ("results", RESULTS_DIR),
            ("weights", WEIGHTS_DIR),
            ("configs", CONFIG_DIR),
        ]
    )
