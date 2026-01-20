from dataclasses import dataclass
from torch.optim.lr_scheduler import *  # noqa


@dataclass
class Config:
    scheduleur: str = 'CosineAnnealingLR'
    T_max: int = 10
