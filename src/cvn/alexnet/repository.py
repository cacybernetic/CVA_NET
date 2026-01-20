import os
import json
from typing import Dict, Any
import torch
from .model import AlexNet, Config
from .factory import alexnet

CONFIG_FILE_NAME = 'config.json'
DATA_FILE_NAME = 'weights.pth'


def _write_json_file(data_dict: Dict[str, Any], file_path: str, encoding: str='utf-8', indent: int=2) -> None:
    with open(file_path, mode='w', encoding=encoding) as f:
        config_json_data = json.dumps(data_dict, indent=indent)
        f.write(config_json_data)


def _read_json_file(file_path: str, encoding: str='utf-8') -> Dict[str, Any]:
    with open(file_path, mode='r', encoding=encoding) as f:
        config_data_dict = json.load(f)
        return config_data_dict


def save_config(config: Config, dir_path: str, encoding: str='utf-8') -> str:
    assert config is not None, "The instance of the model config is none (NoneType). "
    assert dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    config_file = os.path.join(dir_path, CONFIG_FILE_NAME)
    # Load state dict of model config;
    config_data = config.__dict__
    # Save state dict of model config into file;
    os.makedirs(dir_path, exist_ok=True)
    _write_json_file(config_data, config_file, encoding=encoding)
    return config_file


def save_data(model: AlexNet, dir_path: str, device_type: str=None) -> str:
    assert model is not None, "The instance of the model is none (NoneType). "
    assert dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    model_file = os.path.join(dir_path, DATA_FILE_NAME)
    weights = model.state_dict()
    cpu_weights = {}
    # Move weight into CPU if it not is in CPU;
    if not device_type or device_type != 'cpu':
        for name, weight in weights.items():
            weight = weight.cpu()
            cpu_weights[name] = weight
        weights = cpu_weights
    # Save weights model into file;
    os.makedirs(dir_path, exist_ok=True)
    torch.save(weights, model_file)
    return model_file


def load_config(dir_path: str, encoding: str='utf-8') -> Config:
    assert dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such model directory at: %s" % (dir_path,))
    config_file = os.path.join(dir_path, CONFIG_FILE_NAME)
    config = Config()
    # Load model config;
    if not os.path.isfile(config_file):
        raise FileNotFoundError("No such backbone config file at: %s" % (config_file,))
    config_data = _read_json_file(config_file, encoding)
    config.__dict__.update(config_data)
    return config


def load_data(dir_path: str, config: Config=None, model: AlexNet=None) -> AlexNet:
    assert dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such model directory at: %s" % (dir_path,))
    model_file = os.path.join(dir_path, DATA_FILE_NAME)
    # Load Alexnet model weights;
    if model is None:
        assert config is not None, (
            "The existing model instance is not provided, so you have to provide an instance of config.")
        model, _ = alexnet(config)
    if not os.path.isfile(model_file):
        raise FileNotFoundError("No such model file at: %s" % (model_file,))
    weights = torch.load(model_file, weights_only=True, map_location='cpu')
    model.load_state_dict(weights)
    return model
