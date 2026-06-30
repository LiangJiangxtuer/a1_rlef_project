from __future__ import annotations

import os, platform, random, subprocess, json
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def snapshot_env() -> dict:
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu.append(torch.cuda.get_device_name(i))
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda': torch.version.cuda,
        'gpus': gpu,
    }
