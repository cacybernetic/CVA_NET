import logging
import typing as t
from dataclasses import dataclass

import yaml
import torch
from torch import optim

# Set up logging:
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s \t %(message)s',
    handlers=[
        logging.FileHandler("classification_optimizer.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)


###############################################################################
# OPTIMIZER
###############################################################################


@dataclass
class OptimizerConfig:
    optimizer: str = 'AdamW'
    params: t.List[t.Dict[str,  t.Any]]
    lr0: float=1e-4
    weight_decay: float=0
    eps: float = 1e-8
    betas: t.Tuple[float, float] = (0.9, 0.999)


def _adamw_optim_factory_fn(config: OptimizerConfig):
    instance = optim.AdamW(
        params=config.params, lr=config.lr0, weight_decay=config.weight_decay,
        eps=config.eps, betas=config.betas,
    )
    return instance


IMPLEMENTED_OPTIMIZERS = {
    'AdamW': _adamw_optim_factory_fn,
}


class OptimizerFactory:

    @staticmethod
    def build(config: OptimizerConfig=None, **kwargs: t.Dict[str, t.Any]):
        if config is None:
            config = OptimizerConfig()
        config.__dict__.update(kwargs)
        if config.optimizer not in IMPLEMENTED_OPTIMIZERS:
            raise NotImplementedError(
                "The optimizer named '%s' is not implemented yed."
                % (config.optimizer,)
            )
        optim_factory_fn = IMPLEMENTED_OPTIMIZERS[config.optimizer]
        instance = optim_factory_fn(config)
        return instance


###############################################################################
# OPTIMIZATION SCHEDULER
###############################################################################

from torch.optim import lr_scheduler

@dataclass
class SchedulerConfig:
    scheduler: str = 'ReduceLROnPlateau'
    mode: str = 'min'
    factor: float=0.5
    patience: int = 10


def _reducelronplateau_factory_fn(optimizer, config: SchedulerConfig):
    instance = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode=config.min, factor=config.factor,
        patience=config.patience,
    )
    return instance


IMPLEMENTED_SCHEDULERS = {
    'ReduceLROnPlateau': _reducelronplateau_factory_fn,
}


class SchedulerFactory:

    @staticmethod
    def build(config: SchedulerConfig=None, **kwargs: t.Dict[str, t.Any]):
        if config is None:
            config = SchedulerConfig()
        config.__dict__.update(kwargs)
        if config.scheduler not in IMPLEMENTED_SCHEDULERS:
            raise NotImplementedError(
                "The scheduler named '%s' is not implemented yed."
                % (config.scheduler,)
            )
        optim_factory_fn = IMPLEMENTED_SCHEDULERS[config.scheduler]
        instance = optim_factory_fn(config)
        return instance
