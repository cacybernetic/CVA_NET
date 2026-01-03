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
        logging.FileHandler("classification_scheduler.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)



###############################################################################
# OPTIMIZATION SCHEDULER
###############################################################################

from torch.optim import lr_scheduler

@dataclass
class SchedulerConfig:
    scheduler: str = 'ReduceLROnPlateau'
    mode: str = 'min'
    factor: float=0.1
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

    def save(self, scheduler, config: SchedulerConfig) -> None:
        self.save_weights(scheduler, self._scheduler_fp)
        self.save_config(config.__dict__, self._conf_fp)

    def load_scheduler(self, scheduler):
        weights = self.load_weights(self._scheduler_fp)
        scheduler.load_state_dict(weights)
        return scheduler

    def load_optimizer_config(self, config: SchedulerConfig) -> SchedulerConfig:
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

    @staticmethod
    def load(
        repository: SchedulerRepository
    ):
        scheduler_config = SchedulerConfig()
        scheduler_config = repository.load_scheduler_config(scheduler_config)
        scheduler = SchedulerFactory.build(scheduler_config)
        loaded_scheduler = repository.load_scheduler(scheduler)
        return loaded_scheduler, scheduler_config


###############################################################################
# MAIN IMPLEMENTATION
###############################################################################

def _get_arguments():
    import argparse

    parser = argparse.ArgumentParser()

    # scheduler: str = 'ReduceLROnPlateau'
    # mode: str = 'min'
    # factor: float=0.5
    # patience: int = 10
    scheduler_choices = list(IMPLEMENTED_SCHEDULERS.keys())
    parser.add_argument(
        '--scheduler-name', type=str, choices=scheduler_choices,
        default='ReduceLROnPlateau',
        help=(
            "The name of the scheduler selected or the path to file "
            "of the saved scheduler. By default `ReduceLROnPlateau` "
            "is selected."
        )
    )
    parser.add_argument(
        '--mode', type=str, choices=["min", "max"], default="min",
        help=(
            "One of min, max. In min mode, `lr` will be reduced "
            "when the quantity monitored has stopped decreasing; in max mode "
            "it will be reduced when the quantity monitored has stopped "
            "increasing. Default: `'min'`."
        )
    )
    parser.add_argument(
        '--factor', type=float, default=0.1,
        help=(
            "Factor by which the learning rate will be reduced. "
            "`new_lr = lr * factor`. Default: `0.1`."
        )
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help=(
            "The number of allowed epochs with no improvement "
            "after which the learning rate will be reduced. "
            "For example, consider the case of having no patience "
            "(patience = 0). In the first epoch, a baseline is established "
            "and is always considered good as there's no previous baseline. "
            "In the second epoch, if the performance is worse "
            "than the baseline, we have what is considered an intolerable "
            "epoch. Since the count of intolerable epochs (1) "
            "is greater than the patience level (0), the learning rate "
            "is reduced at the end of this epoch. From the third epoch "
            "onwards, the learning rate continues to be reduced at the end "
            "of each epoch if the performance is worse than the baseline. "
            "If the performance improves or remains the same, "
            "the learning rate is not adjusted. Default: 10."
        )
    )

    parser.add_argument(
        '--optimizer', type=str, default=None, 
        help="The path to the saved optimizer what we want to schedule."
    )
    parser.add_argument(
        '-o', '--output', type=str, default=None,
        help=(
            "The path to the output directory where we want to save "
            "the optimizer and its scheduler."
        )
    )
    return parser.parse_args()


def _build_scheduler() -> None:
    ...


def _load_scheduler() -> None:
    ...


def main() -> None:
    import os
    import sys
    from pathlib import Path

    args = _get_arguments()
    output_dir = args.output if args.output is not None else ''
    saved_scheduler = None

    if os.path.isdir(args.scheduler):
        saved_scheduler = Path(args.scheduler)

    sys.exit(0)

if __name__ == '__main__':
    main()
