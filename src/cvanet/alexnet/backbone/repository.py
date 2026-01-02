import os
import json
import torch
from .model import AlexNetBackbone, Config
from .factory import alexnet_backbone

CONFIG_FILE_NAME = 'config.json'
DATA_FILE_NAME = 'weights.pth'


def save_config(config: Config, dir_path: str, encoding: str='utf-8') -> None:
    assert config is not None, "The instance of the model config is none (NoneType)."
    assert dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    # Save config model into file;
    config_file = os.path.join(dir_path, CONFIG_FILE_NAME)
    with open(config_file, mode='w', encoding=encoding) as f:
        config_json_data = json.dumps(config.__dict__, indent='2')
        f.write(config_json_data)


def save_data(model: AlexNetBackbone, dir_path: str, device_type: str=None) -> None:
    assert model is not None, "The instance of the model is none (NoneType)."
    # Move weight into CPU if it not is in CPU;
    weights = model.state_dict()
    if not device_type or device_type != 'cpu':
        cpu_weights = {}
        for name, weight in weights.items():
            weight = weight.cpu()
            cpu_weights[name] = weight
        weights = cpu_weights
    # Save weights model into file;
    model_file = os.path.join(dir_path, DATA_FILE_NAME)
    torch.save(weights, model_file)


def load_config(dir_path: str, encoding: str='utf-8') -> Config:
    assert dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such model directory at: %s" % (dir_path,))
    # Load config data of model;
    config_file = os.path.join(dir_path, CONFIG_FILE_NAME)
    if not os.path.isfile(config_file):
        raise FileNotFoundError("No such config file at: %s" % (config_file,))
    config = Config()
    with open(config_file, mode='r', encoding=encoding) as f:
        config_data_dict = json.load(f)
        config.__dict__.update(config_data_dict)
    return config


def load_data(dir_path: str, config: Config,  model: AlexNetBackbone=None) -> AlexNetBackbone:
    assert not dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such model directory at: %s" % (dir_path,))
    # Load model weights;
    if model is None:
        model, _ = alexnet_backbone(config)
    model_file = os.path.join(dir_path, DATA_FILE_NAME)
    if not os.path.isfile(model_file):
        raise FileNotFoundError("No such model file at: %s" % (model_file,))
    weights = torch.load(model_file, weights_only=True, map_location='cpu')
    model.load_state_dict(weights)
    return model
