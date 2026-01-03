from typing import Tuple
from .model import JEPA, Config
from cvanet.alexnet.backbone.factory import alexnet_backbone


def jepa(config: Config=None, **kwargs) -> Tuple[JEPA, Config]:
    if not config:
        config = Config()
    config.__dict__.update(kwargs)
    backbone, _ = alexnet_backbone(config.backbone)
    model = JEPA(
        backbone=backbone, latent_dim=config.latent_dim, ema_tau_min=config.ema_tau_min, ema_tau_max=config.ema_tau_max,
        ema_total_steps=config.ema_total_steps)
    return model, config
