import importlib
import numpy as np
import torch
import os
import torchvision.transforms as transforms

from src import nets
from src.config import NetConfig, CriterionConfig, OptimizerConfig, SchedulerConfig


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
