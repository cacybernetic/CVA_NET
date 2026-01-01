from typing import Tuple
from cva_net.alexnet.jepa.model import JEPA
from cva_net.alexnet.jepa.factory import jepa
from .model import JEPATrainer, Config as JEPATrainerConfig
from .optimizer.model import Optimizer
from .optimizer.lr_scheduler.model import LRScheduler


def jepa_trainer(
    model: JEPA=None,
    optimizer: Optimizer=None,
    scheduler: LRScheduler=None,
    config: JEPATrainerConfig=None,
    **kwargs
) -> Tuple[JEPATrainer, JEPATrainerConfig]:
    if config is None:
        config = JEPATrainerConfig()
    config.__dict__.update(kwargs)
    if model is None:
        model, _ = jepa(config.model)
    instance = JEPATrainer(model=model, optimizer=optimizer, scheduler=scheduler, config=config)
    return instance, config
