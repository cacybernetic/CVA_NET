import json
from typing import Dict, Any
import torch
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


def save(model: JEPATrainer, config: Config, dir_path: str, encoding: str='utf-8') -> Dict[str, str]:
    pass

