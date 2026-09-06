"""Runtime settings shared by the command-line tools."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

def configure_determinism(seed: int = 0) -> None:
    """Configure process-level RNGs and deterministic Torch kernels."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
