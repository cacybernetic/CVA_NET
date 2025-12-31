import os
import json
from typing import Tuple, Dict, Any
import torch
from cva_net.alexnet.backbone.factory import alexnet_backbone
from cva_net.alexnet.backbone.model import Config as BackboneConfig
from .model import JEPA, Config
from .factory import jepa


def _write_json_file(data_dict: Dict[str, Any], file_path: str, encoding: str='utf-8', indent: int=2) -> None:
    with open(file_path, mode='w', encoding=encoding) as f:
        config_json_data = json.dumps(data_dict, indent=indent)
        f.write(config_json_data)


def _read_json_file(file_path: str, encoding: str='utf-8') -> Dict[str, Any]:
    with open(file_path, mode='r', encoding=encoding) as f:
        config_data_dict = json.load(f)
        return config_data_dict


def save(
    backbone_config: BackboneConfig,
    model: JEPA,
    config: Config,
    dir_path: str,
    device_type: str=None,
    encoding: str='utf-8'
) -> Dict[str, str]:
    assert not dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if model is None or config is None:
        raise ValueError("The instance of the model or its config is none (NoneType). ")
    backbone_config_file = os.path.join(dir_path, 'backbone.json')
    config_file = os.path.join(dir_path, 'config.json')
    model_file = os.path.join(dir_path, 'weights.pth')
    weights = model.state_dict()
    cpu_weights = {}
    # Move weight into CPU if it not is in CPU.
    if not device_type or device_type != 'cpu':
        for name, weight in weights.items():
            weight = weight.cpu()
            cpu_weights[name] = weight
        weights = cpu_weights
    # Save weights model into file;
    torch.save(weights, model_file)
    # Save model config into file;
    _write_json_file(config.__dict__, config_file, encoding=encoding)
    # Save backbone config into file.
    _write_json_file(backbone_config.__dict__, backbone_config_file, encoding=encoding)
    return {
        'backbone_config_file': backbone_config_file,
        'config_file': config_file,
        'model_file': model_file,
    }


def load(dir_path: str, encoding: str='utf-8') -> Tuple[JEPA, Config]:
    assert not dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such model directory at: %s" % (dir_path,))
    backbone_config_file = os.path.join(dir_path, 'backbone.json')
    config_file = os.path.join(dir_path, 'config.json')
    model_file = os.path.join(dir_path, 'weights.pth')
    backbone_config = BackboneConfig()
    config = Config()
    # Load backbone config data;
    if not os.path.isfile(backbone_config_file):
        raise FileNotFoundError("No such backbone config file at: %s" % (backbone_config_file,))
    backbone_config_data = _read_json_file(backbone_config_file, encoding)
    backbone_config.__dict__.update(backbone_config_data)
    # Load model config;
    if not os.path.isfile(config_file):
        raise FileNotFoundError("No such backbone config file at: %s" % (config_file,))
    config_data = _read_json_file(config_file, encoding)
    config.__dict__.update(config_data)
    # Load JEPA model weights;
    backbone, _ = alexnet_backbone(backbone_config)
    model, _ = jepa(backbone, config)
    if not os.path.isfile(model_file):
        raise FileNotFoundError("No such model file at: %s" % (model_file,))
    weights = torch.load(model_file, weights_only=True, map_location='cpu')
    model.load_state_dict(weights)
    return model, config
