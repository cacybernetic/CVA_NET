import os
import json
from typing import Tuple, Dict

import torch
from torch import nn
from .model import Optimizer, Config
from .factory import optimizer


def save(
    model: Optimizer,
    config: Config,
    dir_path: str,
    device_type: str=None,
    encoding: str='utf-8',
) -> Dict[str, str]:
    assert not dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if model is None or config is None:
        raise ValueError("The instance of the model or its config is none (NoneType). ")
    model_file = os.path.join(dir_path, 'weights.pth')
    config_file = os.path.join(dir_path, 'config.json')
    # Move weight into CPU if it not is in CPU;
    weights = model.state_dict()
    if not device_type or device_type != 'cpu':
        cpu_weights = {}
        for name, weight in weights.items():
            weight = weight.cpu()
            cpu_weights[name] = weight
        weights = cpu_weights
    # Save weights model into file;
    torch.save(weights, model_file)
    # Save config model into file;
    with open(config_file, mode='w', encoding=encoding) as f:
        config_json_data = json.dumps(config.__dict__, indent=2)
        f.write(config_json_data)
    return {
      'model_file': model_file,
      'config_file': config_file,
    }


def load(dir_path: str, model: nn.Module, encoding: str='utf-8') -> Tuple[Optimizer, Config]:
    assert not dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such model directory at: %s" % (dir_path,))
    config_file = os.path.join(dir_path, 'config.json')
    model_file = os.path.join(dir_path, 'weights.pth')
    # Load config data of model;
    if not os.path.isfile(config_file):
        raise FileNotFoundError("No such config file at: %s" % (config_file,))
    config = Config()
    with open(config_file, mode='r', encoding=encoding) as f:
        config_data_dict = json.load(f)
        config.__dict__.update(config_data_dict)
    # Load model weights;
    model, _ = optimizer(model, config)
    if not os.path.isfile(model_file):
        raise FileNotFoundError("No such model file at: %s" % (model_file,))
    weights = torch.load(model_file, weights_only=True, map_location='cpu')
    model.load_state_dict(weights)
    return model, config
