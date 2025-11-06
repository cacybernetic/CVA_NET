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

@dataclass
class OptimizerConfig:
    optimizer: str = 'AdamW'
    params: t.List[t.Dict[str,  t.Any]]
    lr0: float=1e-4
    weight_decay: float=0
    eps: float = 1e-8
    betas: t.Tuple[float, float] = (0.9, 0.999)


def adamw_optim_factory(config: OptimizerConfig):
    instance = optim.AdamW(
        params=config.params, lr=config.lr0, weight_decay=config.weight_decay,
        eps=config.eps, betas=config.betas,
    )
    return instance


IMPLEMENTED_OPTIMIZERS = {
    'AdamW': adamw_optim_factory,
}


class OptimizerFactory:

    @staticmethod
    def build(config: OptimizerConfig=None, **kwargs: t.Dict[str, t.Any]):
        if config is None:
            config = OptimizerConfig()
        ...
