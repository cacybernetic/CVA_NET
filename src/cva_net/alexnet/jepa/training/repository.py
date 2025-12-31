import os
import json
from typing import Dict, Any, Tuple
from torch.utils.data import Dataset
from cva_net.alexnet.jepa import repository as jepa_repos
from cva_net.alexnet.jepa.training.factory import jepa_trainer
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
        "The directory path containing the training state and its configs is not provided. "
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


def load(
    dir_path: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    encoding: str='utf-8'
) -> Tuple[JEPATrainer, Config]:
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
    # Load training config;
    if not os.path.isfile(config_file):
        raise FileNotFoundError("No such training config file at: %s" % (config_file,))
    training_config_data = _read_json_file(config_file, encoding)
    trainer_config = Config()
    trainer_config.__dict__.update(training_config_data)
    trainer_config.model = model_config
    trainer_config.optimizer = optimizer_config
    trainer_config.scheduler = scheduler_config
    # Load training state model;
    trainer, _ = jepa_trainer(train_dataset, val_dataset, model, optimizer, scheduler, trainer_config)
    if not os.path.isfile(model_file):
        raise FileNotFoundError("No such training state file at: %s" % (model_file,))
    state_dict = _read_json_file(model_file, encoding)
    trainer.load_state_dict(state_dict)
    return trainer, trainer_config
