"""Reproducibility utilities — fixes random seeds across every library used."""

import random

import numpy as np


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix random seeds across numpy, random, and torch (if installed).

    Call this once at the very top of every entrypoint script
    (eda.py, train.py, evaluate.py, predict.py) before any data
    loading or model construction happens.

    Args:
        seed: The seed value to apply everywhere. Defaults to the
            module-level SEED constant so all scripts stay in sync.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN behavior — slightly slower, fully reproducible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass