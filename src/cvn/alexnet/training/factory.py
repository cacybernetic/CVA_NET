from typing import Tuple
from .model import Trainer, Config


def model_trainer(config: Config=None, **kwargs) -> Tuple[Trainer, Config]:
    if config is None:
        config = Config()
    config.__dict__.update(kwargs)
    instance = Trainer(config=config)
    return instance, config
