from typing import Tuple
from .model import YOLOTrainer, Config


def yolo_trainer(config: Config=None, **kwargs) -> Tuple[YOLOTrainer, Config]:
    if config is None:
        config = Config()
    config.__dict__.update(kwargs)
    instance = YOLOTrainer(config=config)
    return instance, config
