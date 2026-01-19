import os
import json
import torch
from yolo.v3.training.optimizer.model import Optimizer
from .model import LRScheduler, Config
from .factory import lr_scheduler

CONFIG_FILE_NAME = 'config.json'
DATA_FILE_NAME = 'weights.pth'


def save_config(config: Config, dir_path: str, encoding: str='utf-8') -> str:
    assert config is not None, "The instance of the scheduler config is none (NoneType)."
    assert dir_path, (
        "The directory path containing the scheduler weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    config_file = os.path.join(dir_path, CONFIG_FILE_NAME)
    os.makedirs(dir_path, exist_ok=True)
    # Save config model into file;
    with open(config_file, mode='w', encoding=encoding) as f:
        config_json_data = json.dumps(config.__dict__, indent=2)
        f.write(config_json_data)
    return config_file


def save_data(scheduler: LRScheduler, dir_path: str) -> str:
    assert scheduler is not None, "The instance of the scheduler is none (NoneType)."
    assert dir_path, (
        "The directory path containing the scheduler weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    model_file = os.path.join(dir_path, DATA_FILE_NAME)
    os.makedirs(dir_path, exist_ok=True)
    # Save weights model into file;
    torch.save(scheduler.state_dict(), model_file)
    return model_file


def load_config(dir_path: str, encoding: str='utf-8') -> Config:
    assert dir_path, (
        "The directory path containing the scheduler weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such scheduler config directory at: %s" % (dir_path,))
    config_file = os.path.join(dir_path, CONFIG_FILE_NAME)
    # Load config data of model;
    if not os.path.isfile(config_file):
        raise FileNotFoundError("No such scheduler config file at: %s" % (config_file,))
    config = Config()
    with open(config_file, mode='r', encoding=encoding) as f:
        config_data_dict = json.load(f)
        config.__dict__.update(config_data_dict)
    return config


def load_data(dir_path: str, config: Config, scheduler: LRScheduler=None, optimizer: Optimizer=None) -> LRScheduler:
    if scheduler is None:
        assert optimizer is not None, (
            "An scheduler instance is not provided. So the optimizer instance must be provided to build "
            "a new scheduler instance before loading state dict.")
    assert dir_path, (
        "The directory path containing the scheduler weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such scheduler weights directory at: %s" % (dir_path,))
    model_file = os.path.join(dir_path, DATA_FILE_NAME)
    # Load scheduler weights;
    if scheduler is None:
        scheduler, _ = lr_scheduler(optimizer, config)
    if not os.path.isfile(model_file):
        raise FileNotFoundError("No such scheduler file at: %s" % (model_file,))
    weights = torch.load(model_file)
    scheduler.load_state_dict(weights)
    return scheduler
