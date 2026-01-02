import os
import json
from typing import Dict, Any
from cvanet.alexnet.jepa import repository as jepa_repos
from cvanet.alexnet.jepa.training.factory import jepa_trainer
from cvanet.alexnet.jepa.training.optimizer import repository as optimizer_repos
from cvanet.alexnet.jepa.training.optimizer.lr_scheduler import repository as scheduler_repos
from .model import JEPATrainer, Config

CONFIG_FILE_NAME = 'config.json'
DATA_FILE_NAME = 'training.json'


def _write_json_file(data_dict: Dict[str, Any], file_path: str, encoding: str='utf-8', indent: int=2) -> None:
    with open(file_path, mode='w', encoding=encoding) as f:
        config_json_data = json.dumps(data_dict, indent=indent)
        f.write(config_json_data)


def _read_json_file(file_path: str, encoding: str='utf-8') -> Dict[str, Any]:
    with open(file_path, mode='r', encoding=encoding) as f:
        config_data_dict = json.load(f)
        return config_data_dict


def save_config(config: Config, dir_path: str, encoding: str='utf-8') -> Dict[str, Any]:
    assert config is not None, "The instance of the model or its config is none (NoneType)."
    assert dir_path, (
        "The directory path containing the training state and its configs is not provided. "
        "NoneType/blank string provided instead.")
    # Compose folder path;
    config_file = os.path.join(dir_path, CONFIG_FILE_NAME)
    jepa_model_dir = os.path.join(dir_path, 'jepa')
    optimizer_dir = os.path.join(dir_path, 'optimizer')
    scheduler_dir = os.path.join(dir_path, 'scheduler')
    results: Dict[str, Any] = {"trainer_config": config_file}
    os.makedirs(dir_path, exist_ok=True)
    # Save  model config;
    results['model_config'] = jepa_repos.save_config(config.model, jepa_model_dir, encoding)  # noqa
    # Save optimizer;
    results['optimizer_config'] = optimizer_repos.save_config(config.optimizer, optimizer_dir, encoding)  # noqa
    # Save scheduler;
    results['scheduler_config'] = scheduler_repos.save_config(config.scheduler, scheduler_dir, encoding)  # noqa
    # Save the config of training model;
    config_data = {attr:val for attr, val in config.__dict__.items() if attr not in ('model', 'optimizer', 'scheduler')}
    _write_json_file(config_data, config_file, encoding)
    return results


def save_data(trainer: JEPATrainer, dir_path: str, device_type: str=None, encoding: str='utf-8') -> Dict[str, Any]:
    assert trainer is not None, "The instance of the model or its config is none (NoneType)."
    assert dir_path, (
        "The directory path containing the training state and its configs is not provided. "
        "NoneType/blank string provided instead.")
    # Compose folder path;
    model_file = os.path.join(dir_path, DATA_FILE_NAME)
    jepa_model_dir = os.path.join(dir_path, 'jepa')
    optimizer_dir = os.path.join(dir_path, 'optimizer')
    scheduler_dir = os.path.join(dir_path, 'scheduler')
    results: Dict[str, Any] = {"trainer": model_file}
    # Retreive instances;
    model = trainer.model
    optimizer = trainer.optimizer
    scheduler = trainer.scheduler
    os.makedirs(dir_path, exist_ok=True)
    # Save JEPA model;
    results['model'] = jepa_repos.save_data(model, jepa_model_dir, device_type=device_type)  # noqa
    # Save optimizer;
    results['optimizer'] = optimizer_repos.save_data(optimizer, optimizer_dir)  # noqa
    # Save scheduler;
    results['scheduler'] = scheduler_repos.save_data(scheduler, scheduler_dir)  # noqa
    # Save the training model;
    train_state_dict = trainer.state_dict()
    _write_json_file(train_state_dict, model_file, encoding)
    return results


def load_config(dir_path: str, encoding: str='utf-8') -> Config:
    assert dir_path, (
        "The directory path containing the training configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such training model directory at: %s" % (dir_path,))
    # Compose folder path;
    config_file = os.path.join(dir_path, CONFIG_FILE_NAME)
    jepa_model_dir = os.path.join(dir_path, 'jepa')
    optimizer_dir = os.path.join(dir_path, 'optimizer')
    scheduler_dir = os.path.join(dir_path, 'scheduler')
    # Load JEPA model config;
    model_config = jepa_repos.load_config(jepa_model_dir, encoding)
    # Load Optimizer model config;
    optimizer_config = optimizer_repos.load_config(optimizer_dir, encoding)
    # Load scheduler model config;
    scheduler_config = scheduler_repos.load_config(scheduler_dir, encoding)
    # Load training config;
    if not os.path.isfile(config_file):
        raise FileNotFoundError("No such training config file at: %s" % (config_file,))
    training_config_data = _read_json_file(config_file, encoding)
    trainer_config = Config()
    trainer_config.__dict__.update(training_config_data)
    trainer_config.model = model_config
    trainer_config.optimizer = optimizer_config
    trainer_config.scheduler = scheduler_config
    return trainer_config


def load_data(dir_path: str, config: Config, encoding: str='utf-8', trainer: JEPATrainer=None) -> JEPATrainer:
    assert dir_path, (
        "The directory path containing the training state is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such training model directory at: %s" % (dir_path,))
    # Compose folder path;
    model_file = os.path.join(dir_path, DATA_FILE_NAME)
    jepa_model_dir = os.path.join(dir_path, 'jepa')
    optimizer_dir = os.path.join(dir_path, 'optimizer')
    scheduler_dir = os.path.join(dir_path, 'scheduler')
    # Load training state model;
    if trainer is None:
        trainer, _ = jepa_trainer(config)
    if not os.path.isfile(model_file):
        raise FileNotFoundError("No such training state file at: %s" % (model_file,))
    state_dict = _read_json_file(model_file, encoding)
    trainer.load_state_dict(state_dict)
    # Load JEPA model;
    trainer.model = jepa_repos.load_data(jepa_model_dir, config.model, trainer.model)
    # Load Optimizer model;
    trainer.optimizer = optimizer_repos.load_data(optimizer_dir, config.optimizer,  trainer.optimizer, trainer.model)
    # Load scheduler model;
    trainer.scheduler = scheduler_repos.load_data(scheduler_dir, config.scheduler, trainer.scheduler, trainer.optimizer)
    return trainer
