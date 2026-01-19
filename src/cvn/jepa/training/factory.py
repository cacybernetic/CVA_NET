from typing import Tuple
from .model import JEPATrainer, Config as JEPATrainerConfig


def jepa_trainer(config: JEPATrainerConfig=None, **kwargs) -> Tuple[JEPATrainer, JEPATrainerConfig]:
    if config is None:
        config = JEPATrainerConfig()
    config.__dict__.update(kwargs)
    instance = JEPATrainer(config=config)
    return instance, config
