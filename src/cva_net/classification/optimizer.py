import logging
import typing as t
from dataclasses import dataclass

import yaml
import torch
from torch import nn
from torch import optim

# Set up logging:
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s \t %(message)s',
    handlers=[
        logging.FileHandler("classification_optimizer.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)


###############################################################################
# OPTIMIZER
###############################################################################


@dataclass
class OptimizerConfig:
    optimizer: str = 'AdamW'
    params: t.List[t.Dict[str,  t.Union[t.List[str], float]]] = None
    lr0: float=1e-4
    weight_decay: float=0
    eps: float = 1e-8
    betas: t.Tuple[float, float] = (0.9, 0.999)


###############################################################################
# OPTIMIZER REPOSITORY
###############################################################################
from pathlib import Path
import yaml


class OptimizerRepository:
    optimizer_FN: str = 'optimizer'
    CONFIG_FN: str = 'configs.yaml'

    def __init__(
        self,
        optimizer_folder: str,
        optimizer_fn: str = optimizer_FN,
        config_fn: str = CONFIG_FN,
    ) -> None:
        self.optimizer_fn = optimizer_fn
        self.config_fn = config_fn

        self._folder: Path = None
        self._optimizer_fp: Path = None
        self._conf_fp: Path = None
        self.set_optimizer_folder(optimizer_folder)

    def set_optimizer_folder(self, folder: str) -> None:
        """
        Set the optimizer storage folder and update all file paths.

        :param folder: Root directory path for optimizer storage.
        """
        self._folder = Path(folder)
        self._optimizer_fp = self._folder / (self.optimizer_fn + '.pth')
        self._conf_fp = self._folder / self.config_fn

    @staticmethod
    def save_weights(m: optim.Optimizer, file_path: Path) -> None:
        """
        Save optimizer weights to specified file path.

        :param m: PyTorch optimizer module to save.
        :param file_path: Destination file path for weights.
        """
        file_path.parent.mkdir(exist_ok=True)
        weights = m.state_dict()
        torch.save(obj=weights, f=str(file_path.absolute()))

    @staticmethod
    def load_weights(
        file_path: t.Union[Path, str],
        map_location: torch.device=None
    ) -> dict:
        """
        Load optimizer weights from specified file path.

        :param file_path: Source file path for weights.
        :param map_location: Device to load weights onto, defaults to CPU.
        :return: Dictionary containing optimizer state.
        :raises FileNotFoundError: If weights file does not exist.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(
                "No such optimizer file at \"%s\"." % (str(file_path),)
            )
        if map_location is None:
            map_location = torch.device('cpu')
        weights = torch.load(
            f=str(file_path.absolute()), weights_only=False,
            map_location=map_location
        )
        return weights

    @staticmethod
    def save_config(c: t.Dict[str, t.Any], file_path: Path) -> None:
        """
        Save configuration dictionary to YAML file.

        :param c: Configuration dictionary to save.
        :param file_path: Destination file path for configuration.
        """
        file_path.parent.mkdir(exist_ok=True)
        with open(file=file_path, mode='w', encoding='utf-8') as file:
            yaml.safe_dump(c, file)

    @staticmethod
    def load_config(file_path: t.Union[Path, str]) -> t.Dict[str, t.Any]:
        """
        Load configuration dictionary from YAML file.

        :param file_path: Source file path for configuration.
        :return: Dictionary containing configuration data.
        :raises FileNotFoundError: If configuration file does not exist.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(
                "No such optimizer file at \"%s\"." % (str(file_path),)
            )
        with open(file=file_path, mode='r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config

    def save(self, opt: optim.Optimizer, config: OptimizerConfig) -> None:
        self.save_weights(opt, self._optimizer_fp)
        self.save_config(config.__dict__, self._conf_fp)

    def load_optimizer(self, opt: optim.Optimizer) -> optim.Optimizer:
        weights = self.load_weights(self._optimizer_fp)
        opt.load_state_dict(weights)
        return opt

    def load_optimizer_config(self, config: OptimizerConfig) -> OptimizerConfig:
        config_dict = self.load_config(self._conf_fp)
        config.__dict__.update(config_dict)
        return config


###############################################################################
# OPTIMIZER FACTORY
###############################################################################
import re


def _find_param(pattern: str, model: nn.Module) -> nn.Parameter:
    found_params = []
    for name, param in model.named_parameters():
        if re.match(pattern=pattern, string=name):
            found_params.append(param)
    return found_params


def _adamw_optim_factory_fn(model: nn.Module, config: OptimizerConfig):
    params_list = []
    if config.params is not None:
        for param_conf in config.params:
            names = param_conf['params']
            found_params = []
            for name in names:
                found = _find_param(name, model)
                found_params.extend(found)
            params_list.append(
                {'params': found_params, 'lr': param_conf['lr']}
            )
    else:
        param_conf = list(model.parameters())

    assert params_list, "The model parameters list provided is empty."
    instance = optim.AdamW(
        params=params_list, lr=config.lr0, weight_decay=config.weight_decay,
        eps=config.eps, betas=config.betas,
    )
    return instance


IMPLEMENTED_OPTIMIZERS = {
    'AdamW': _adamw_optim_factory_fn,
}


class OptimizerFactory:

    @staticmethod
    def build(
        model: nn.Module,
        config: OptimizerConfig=None,
        **kwargs: t.Dict[str, t.Any]
    ) -> optim.Optimizer:
        if config is None:
            config = OptimizerConfig()
        config.__dict__.update(kwargs)
        if config.optimizer not in IMPLEMENTED_OPTIMIZERS:
            raise NotImplementedError(
                "The optimizer named '%s' is not implemented yed."
                % (config.optimizer,)
            )
        optim_factory_fn = IMPLEMENTED_OPTIMIZERS[config.optimizer]
        instance = optim_factory_fn(model, config)
        return instance
    
    @staticmethod
    def load(
        repository: OptimizerRepository
    ) -> t.Tuple[optim.Optimizer, OptimizerConfig]:
        model_config = OptimizerConfig()
        model_config = repository.load_model_config(model_config)
        model = OptimizerFactory.build(model_config)
        loaded_model = repository.load_model(model)
        return loaded_model, model_config


###############################################################################
# OPTIMIZATION SCHEDULER
###############################################################################

from torch.optim import lr_scheduler

@dataclass
class SchedulerConfig:
    scheduler: str = 'ReduceLROnPlateau'
    mode: str = 'min'
    factor: float=0.5
    patience: int = 10


###############################################################################
# SCHEDULER REPOSITORY
###############################################################################
from pathlib import Path
import yaml


class SchedulerRepository:
    scheduler_FN: str = 'scheduler'
    CONFIG_FN: str = 'configs.yaml'

    def __init__(
        self,
        scheduler_folder: str,
        scheduler_fn: str = scheduler_FN,
        config_fn: str = CONFIG_FN,
    ) -> None:
        self.scheduler_fn = scheduler_fn
        self.config_fn = config_fn

        self._folder: Path = None
        self._scheduler_fp: Path = None
        self._conf_fp: Path = None
        self.set_optimizer_folder(scheduler_folder)

    def set_optimizer_folder(self, folder: str) -> None:
        """
        Set the optimizer storage folder and update all file paths.

        :param folder: Root directory path for scheduler storage.
        """
        self._folder = Path(folder)
        self._scheduler_fp = self._folder / (self.scheduler_fn + '.pth')
        self._conf_fp = self._folder / self.config_fn

    @staticmethod
    def save_weights(m: optim.Optimizer, file_path: Path) -> None:
        """
        Save scheduler weights to specified file path.

        :param m: PyTorch scheduler module to save.
        :param file_path: Destination file path for weights.
        """
        file_path.parent.mkdir(exist_ok=True)
        weights = m.state_dict()
        torch.save(obj=weights, f=str(file_path.absolute()))

    @staticmethod
    def load_weights(
        file_path: t.Union[Path, str],
        map_location: torch.device=None
    ) -> dict:
        """
        Load optimizer weights from specified file path.

        :param file_path: Source file path for weights.
        :param map_location: Device to load weights onto, defaults to CPU.
        :return: Dictionary containing scheduler state.
        :raises FileNotFoundError: If weights file does not exist.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(
                "No such optimizer file at \"%s\"." % (str(file_path),)
            )
        if map_location is None:
            map_location = torch.device('cpu')
        weights = torch.load(
            f=str(file_path.absolute()), weights_only=False,
            map_location=map_location
        )
        return weights

    @staticmethod
    def save_config(c: t.Dict[str, t.Any], file_path: Path) -> None:
        """
        Save configuration dictionary to YAML file.

        :param c: Configuration dictionary to save.
        :param file_path: Destination file path for configuration.
        """
        file_path.parent.mkdir(exist_ok=True)
        with open(file=file_path, mode='w', encoding='utf-8') as file:
            yaml.safe_dump(c, file)

    @staticmethod
    def load_config(file_path: t.Union[Path, str]) -> t.Dict[str, t.Any]:
        """
        Load configuration dictionary from YAML file.

        :param file_path: Source file path for configuration.
        :return: Dictionary containing configuration data.
        :raises FileNotFoundError: If configuration file does not exist.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(
                "No such scheduler file at \"%s\"." % (str(file_path),)
            )
        with open(file=file_path, mode='r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config

    def save(self, opt: optim.Optimizer, config: OptimizerConfig) -> None:
        self.save_weights(opt, self._scheduler_fp)
        self.save_config(config.__dict__, self._conf_fp)

    def load_optimizer(self, opt: optim.Optimizer) -> optim.Optimizer:
        weights = self.load_weights(self._scheduler_fp)
        opt.load_state_dict(weights)
        return opt

    def load_optimizer_config(self, config: OptimizerConfig) -> OptimizerConfig:
        config_dict = self.load_config(self._conf_fp)
        config.__dict__.update(config_dict)
        return config



###############################################################################
# SCHEDULER FACTORY
###############################################################################


def _reducelronplateau_factory_fn(
    optimizer: optim.Optimizer, config: SchedulerConfig
):
    instance = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode=config.min, factor=config.factor,
        patience=config.patience,
    )
    return instance


IMPLEMENTED_SCHEDULERS = {
    'ReduceLROnPlateau': _reducelronplateau_factory_fn,
}


class SchedulerFactory:

    @staticmethod
    def build(
        optimizer: optim.Optimizer,
        config: SchedulerConfig=None,
        **kwargs: t.Dict[str, t.Any]
    ):
        if config is None:
            config = SchedulerConfig()
        config.__dict__.update(kwargs)
        if config.scheduler not in IMPLEMENTED_SCHEDULERS:
            raise NotImplementedError(
                "The scheduler named '%s' is not implemented yed."
                % (config.scheduler,)
            )
        optim_factory_fn = IMPLEMENTED_SCHEDULERS[config.scheduler]
        instance = optim_factory_fn(optimizer, config)
        return instance


###############################################################################
# MAIN IMPLEMENTATION
###############################################################################

def _get_arguments():
    import argparse
    ...


def main() -> None:
    ...


if __name__ == '__main__':
    main()
