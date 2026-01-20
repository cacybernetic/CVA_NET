import re
from typing import Tuple, List  # noqa
from torch import nn
from .model import *


def _find_params(pattern: str, model: nn.Module) -> Tuple[List[str], List[nn.Parameter]]:
    found_sublayers = []
    found_params = []
    for name, param in model.named_parameters():
        if re.match(pattern=pattern, string=name):
            found_sublayers.append(name)
            found_params.append(param)
    return found_sublayers, found_params


def optimizer(model: nn.Module, config: Config=None, **kwargs) -> Tuple[Optimizer, Config]:
    if config is None:
        config = Config()
    config.__dict__.update(kwargs)
    params_list = []
    if config.layers_config is not None:
        LOGGER.info("Layers configuration:")
        for layer_name in sorted(config.layers_config.keys()):
            lr = config.layers_config[layer_name]
            LOGGER.info("For learning rate: " + str(lr))
            found_sb_layers, found_params = _find_params(layer_name, model)
            for p in found_sb_layers:
                LOGGER.info("\t* " + str(p))
            params_list.append({'params': found_params, 'lr': lr})
    else:
        params_list = list(model.parameters())
        LOGGER.info("All model parameters are registered for optimization.")
    assert params_list, "The model parameters list provided is empty."
    instance = None
    if config.optimizer == 'Adam':
        instance = Adam(
            params=params_list, lr=config.lr0, weight_decay=config.weight_decay, eps=config.eps, betas=config.betas)
    elif config.optimizer == 'SGD':
        instance = SGD(
            params=params_list, lr=config.lr0, momentum=config.momentum, dampening=config.dampening,
            weight_decay=config.weight_decay)
    elif not config.optimizer or config.optimizer == 'AdamW':
        instance = AdamW(
            params=params_list, lr=config.lr0, weight_decay=config.weight_decay, eps=config.eps, betas=config.betas)
    return instance, config
