import os
import json
from typing import Tuple, Dict

import torch
from torch import nn
from .model import Optimizer, Config
from .factory import optimizer as build_optimizer

CONFIG_FILE_NAME = 'config.json'
DATA_FILE_NAME = 'weights.pth'


def save_config(config: Config, dir_path: str, encoding: str='utf-8') -> str:
    assert config is not None, "The instance of the optimizer config is none (NoneType)."
    assert dir_path, (
        "The directory path containing the optimizer weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    config_file = os.path.join(dir_path, CONFIG_FILE_NAME)
    os.makedirs(dir_path, exist_ok=True)
    # Save config model into file;
    with open(config_file, mode='w', encoding=encoding) as f:
        config_json_data = json.dumps(config.__dict__, indent=2)
        f.write(config_json_data)
    return config_file


def save_data(model: Optimizer, dir_path: str) -> str:
    assert model is not None, "The instance of the optimizer is none (NoneType)."
    assert dir_path, (
        "The directory path containing the optimizer weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    model_file = os.path.join(dir_path, DATA_FILE_NAME)
    os.makedirs(dir_path, exist_ok=True)
    # Save weights model into file;
    torch.save(model.state_dict(), model_file)
    return model_file


def load_config(dir_path: str, encoding: str='utf-8') -> Config:
    assert dir_path, (
        "The directory path containing the optimizer weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such optimizer directory at: %s" % (dir_path,))
    config_file = os.path.join(dir_path, CONFIG_FILE_NAME)
    # Load config data of model;
    if not os.path.isfile(config_file):
        raise FileNotFoundError("No such config file at: %s" % (config_file,))
    config = Config()
    with open(config_file, mode='r', encoding=encoding) as f:
        config_data_dict = json.load(f)
        config.__dict__.update(config_data_dict)
    return config


def load_data(dir_path: str, config: Config, optimizer: Optimizer=None,  model: nn.Module=None) -> Optimizer:
    if optimizer is None:
        assert model is not None, (
            "An optimizer instance is not provided. So the model instance must be provided to build "
            "a new optimizer instance before loading state dict.")
    assert not dir_path, (
        "The directory path containing the optimizer weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such optimizer directory at: %s" % (dir_path,))
    model_file = os.path.join(dir_path, DATA_FILE_NAME)
    # Load model weights;
    if not os.path.isfile(model_file):
        raise FileNotFoundError("No such model file at: %s" % (model_file,))
    if optimizer is None:
        optimizer, _ = build_optimizer(model, config)
    weights = torch.load(model_file)
    optimizer.load_state_dict(weights)
    return optimizer
