from typing import Tuple
from cvanet.alexnet.jepa.training.optimizer.model import Optimizer
from .model import CosineAnnealingLR, LRScheduler, Config


def lr_scheduler(optimizer: Optimizer, config: Config=None, **kwargs) -> Tuple[LRScheduler, Config]:
    if config is None:
        config = Config()
    config.__dict__.update(kwargs)
    instance = None
    if config.scheduleur == 'CosineAnnealingLR':
        instance = CosineAnnealingLR(optimizer, T_max=config.T_max)
    return instance, config
