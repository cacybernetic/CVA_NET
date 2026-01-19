from typing import Tuple
from .model import AlexNetBackbone, Config


def alexnet_backbone(config: Config=None, **kwargs) -> Tuple[AlexNetBackbone, Config]:
    if not config:
        config = Config()
    config.__dict__.update(kwargs)
    model = AlexNetBackbone(num_channels=config.num_channels, latent_dim=config.latent_dim)
    return model, config
