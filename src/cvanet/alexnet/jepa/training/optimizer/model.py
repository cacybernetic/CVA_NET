import logging
from typing import Dict, Tuple
from dataclasses import dataclass
from torch.optim import *  # noqa

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    optimizer: str = 'AdamW'
    momentum: float = 0.005
    dampening: float = 0.000001
    lr0: float = 1e-4
    weight_decay: float = 0.0005
    eps: float = 1e-8
    betas: Tuple[float, float] = (0.9, 0.999)
    layers_config: Dict[str, float] = None
