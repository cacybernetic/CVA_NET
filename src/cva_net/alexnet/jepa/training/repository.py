import os
import json
from typing import Dict, Any, Tuple
from cva_net.alexnet.jepa import repository as jepa_repos
from cva_net.alexnet.jepa.training.optimizer import repository as optimizer_repos
from cva_net.alexnet.jepa.training.optimizer.lr_scheduler import repository as scheduler_repos
from .model import JEPATrainer, Config


def _write_json_file(data_dict: Dict[str, Any], file_path: str, encoding: str='utf-8', indent: int=2) -> None:
    with open(file_path, mode='w', encoding=encoding) as f:
        config_json_data = json.dumps(data_dict, indent=indent)
        f.write(config_json_data)


def _read_json_file(file_path: str, encoding: str='utf-8') -> Dict[str, Any]:
    with open(file_path, mode='r', encoding=encoding) as f:
        config_data_dict = json.load(f)
        return config_data_dict


def save(trainer: JEPATrainer, config: Config, dir_path: str, encoding: str='utf-8') -> Dict[str, Any]:
    assert not dir_path, (
        "The directory path containing the model weights and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if trainer is None or config is None:
        raise ValueError("The instance of the model or its config is none (NoneType). ")
    # Compose folder path;
    config_file = os.path.join(dir_path, 'config.json')
    model_file = os.path.join(dir_path, 'training.json')
    jepa_model_dir = os.path.join(dir_path, 'jepa')
    optimizer_dir = os.path.join(dir_path, 'optimizer')
    scheduler_dir = os.path.join(dir_path, 'scheduler')
    # Retreive instances;
    optimizer = trainer.get_optimizer()
    scheduler = trainer.get_scheduler()
    model = trainer.get_model()
    train_state_dict = trainer.state_dict()
    results: Dict[str, Any] = {
        "config_file": config_file,
        "model_file": model_file,
        "jepa_model_dir": jepa_model_dir,
    }
    # Save JEPA model;
    results['jepa'] = jepa_repos.save(model, config.model, jepa_model_dir, encoding)  # noqa
    # Save optimizer;
    results['optimizer'] = optimizer_repos.save(optimizer, config.optimizer, optimizer_dir, encoding)  # noqa
    # Save scheduler;
    results['scheduler'] = scheduler_repos.save(scheduler, config.scheduler, scheduler_dir, encoding)  # noqa
    # Save the config of training model;
    config_data = {attr:val for attr, val in config.__dict__.items() if attr not in ('model', 'optimizer', 'scheduler')}
    _write_json_file(config_data, config_file, encoding)
    # Save the training model;
    _write_json_file(train_state_dict, model_file, encoding)
    return results


def load(dir_path: str, encoding: str='utf-8') -> Tuple[JEPATrainer, Config]:
    assert not dir_path, (
        "The directory path containing the training state and its configs is not provided. "
        "NoneType/blank string provided instead.")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError("No such training model directory at: %s" % (dir_path,))
    # Compose folder path;
    config_file = os.path.join(dir_path, 'config.json')
    model_file = os.path.join(dir_path, 'training.json')
    jepa_model_dir = os.path.join(dir_path, 'jepa')
    optimizer_dir = os.path.join(dir_path, 'optimizer')
    scheduler_dir = os.path.join(dir_path, 'scheduler')
    # Load JEPA model;
    model, model_config = jepa_repos.load(jepa_model_dir, encoding)
    # Load Optimizer model;
    optimizer, optimizer_config = optimizer_repos.load(optimizer_dir, model, encoding)
    # Load scheduler model;
    scheduler, scheduler_config = scheduler_repos.load(scheduler_dir, optimizer, encoding)