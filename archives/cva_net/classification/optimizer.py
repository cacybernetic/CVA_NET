import logging
import typing as t
from dataclasses import dataclass

import yaml
import torch
from torch import nn
from torch import optim

LOGGER = logging.getLogger(__name__)


###############################################################################
# OPTIMIZER
###############################################################################


@dataclass
class OptimizerConfig:
    optimizer: str = 'AdamW'
    layers_config: t.Dict[str, float] = None
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
            f=str(file_path.absolute()), weights_only=True,
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

        # DEBUG: Print detailed information
        LOGGER.debug("=== DEBUGGING OPTIMIZER LOAD ===")
        LOGGER.debug(
            "Number of param groups in saved state: %d."
            % (len(weights['param_groups']),)
        )
        LOGGER.debug(
            "Number of param groups in new optimizer: %d."
            % (len(opt.param_groups))
        )
        
        for i, (saved_group, current_group) in enumerate(
            zip(weights['param_groups'], opt.param_groups)
        ):
            saved_param_count = len(saved_group['params'])
            current_param_count = len(current_group['params'])
            LOGGER.debug(f"Group {i}:")
            LOGGER.debug(f"  Saved params count: {saved_param_count}")
            LOGGER.debug(f"  Current params count: {current_param_count}")
            LOGGER.debug(f"  Saved LR: {saved_group['lr']}")
            LOGGER.debug(f"  Current LR: {current_group['lr']}")
            
            if saved_param_count != current_param_count:
                LOGGER.error(f"  ❌ MISMATCH in group {i}!")

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


def _find_params(pattern: str, model: nn.Module) -> nn.Parameter:
    found_sublayers = []
    found_params = []
    for name, param in model.named_parameters():
        if re.match(pattern=pattern, string=name):
            found_sublayers.append(name)
            found_params.append(param)
    return found_sublayers, found_params


def _adamw_optim_factory_fn(model: nn.Module, config: OptimizerConfig):
    params_list = []
    if config.layers_config is not None:
        LOGGER.info("Layers configuration:")
        for layer_name in sorted(config.layers_config.keys()):
            lr = config.layers_config[layer_name]
            LOGGER.info("For learning rate: " + str(lr))
            found_sb_layers, found_params = _find_params(layer_name, model)
            for p in found_sb_layers:
                LOGGER.info("\t* " + str(p))
            params_list.append({'params': found_params, 'lr': lr})
    else:
        params_list = list(model.parameters())
        LOGGER.info("All model parameters are registered for optimization.")

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
    ) -> t.Tuple[optim.Optimizer, OptimizerConfig]:
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
        return instance, config

    @staticmethod
    def load(
        model: nn.Module,
        repository: OptimizerRepository
    ) -> t.Tuple[optim.Optimizer, OptimizerConfig]:
        optimizer_config = OptimizerConfig()
        optimizer_config = repository.load_optimizer_config(optimizer_config)
        optimizer, _ = OptimizerFactory.build(model, optimizer_config)
        loaded_optimizer = repository.load_optimizer(optimizer)
        return loaded_optimizer, optimizer_config


###############################################################################
# MAIN IMPLEMENTATION
###############################################################################

def _get_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    # optimizer: str = 'AdamW'
    # params: t.List[t.Dict[str,  t.Union[t.List[str], float]]] = None
    # lr0: float=1e-4
    # weight_decay: float=0
    # eps: float = 1e-8
    # betas: t.Tuple[float, float] = (0.9, 0.999)
    parser.add_argument(
        '--optimizer', type=str, default='AdamW',
        help=(
            "The name of the optimizer selected or the path to file "
            "of the saved optimizer. By default `AdamW` is selected."
        )
    )
    parser.add_argument(
        '--layers-config', type=str, default=None,
        help="The config file of the leyer parameter optimizations."
    )
    parser.add_argument(
        '--lr0', type=float, default=1e-4,
        help="The learning rate of the optimization step."
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0,
        help="The weight decay value of the optimizer."
    )
    parser.add_argument(
        '--eps', type=float, default=1e-8,
        help="The epsilon value of the optimizer."
    )
    parser.add_argument(
        '--betas', nargs=2, type=float, default=(0.9, 0.999),
        help="The beta values of the optimizer."
    )

    parser.add_argument(
        '-m', '--model', type=str,
        help="The path to the saved model weights."
    )
    parser.add_argument(
        '-o', '--output', type=str, default=None,
        help=(
            "The path to the output directory where we want to save "
            "the optimizer and its scheduler."
        )
    )
    return parser.parse_args()


def _load_model_from_file(model_dir):
    import sys

    ## Model loading from file.
    model = None
    try:
        ### Determination the model architecture.
        model_config_file = model_dir / 'configs.yaml'
        with open(model_config_file, mode='r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
            model_arch = model_config['arch']
            LOGGER.info('Model architecture determinated: ' + str(model_arch))
        ### Import the right implementation of the model component.
        if model_arch == 'AlexNet':
            from cva_net.alexnet import ModelRepository, ModelFactory
        elif model_arch == 'ResNet18':
            from cva_net.resnet18 import ModelRepository, ModelFactory
        elif model_arch == 'ResNet50':
            from cva_net.resnet50 import ModelRepository, ModelFactory
        else:
            raise NotImplementedError(
                "The model architecture named `%s` is not implemented yet."
                % (model_arch,)
            )
        ### Model weight loaded according the implementation found.
        model_repository = ModelRepository(str(model_dir))
        model, _ = ModelFactory.load(model_repository)
        LOGGER.info("Model weights loaded successfully.")
        return model

    except ImportError as e:
        LOGGER.error("Error: " + str(e))
        LOGGER.info("This model architecture is unknown or not implemented.")
        sys.exit(1)


def _build_optimizer(args) -> None:
    from pathlib import Path
    
    output_dir = Path(args.output if args.output is not None else './')
    layers_config = Path(args.layers_config) if args.layers_config is not None \
        else None
    model_dir = Path(args.model)

    ## Openning and loading the config file.
    param_config = None
    if layers_config is not None and layers_config.is_file():
        with open(layers_config, mode='r', encoding='utf-8') as f:
            param_config = yaml.safe_load(f)
            LOGGER.info(
                "Parametter config is loaded from " + str(layers_config)
            )

    ## Loading of model from file.
    model = _load_model_from_file(model_dir)

    ## Building of optimizer in question.
    config = OptimizerConfig()
    config.optimizer = args.optimizer
    config.layers_config = param_config
    config.lr0 = args.lr0
    config.weight_decay = args.weight_decay
    config.eps = args.eps
    config.betas = tuple(args.betas)
    optimizer, _ = OptimizerFactory.build(model, config=config)
    repos_folder = output_dir / 'saved_optimizer'
    repository = OptimizerRepository(repos_folder)
    repository.save(opt=optimizer, config=config)
    LOGGER.info("Optimizer built and saved at: " + str(repos_folder))
    LOGGER.info("Optimizer config: " + repr(config))
    LOGGER.info("Optimizer instance: " + str(optimizer))


def _load_optimizer(args) -> None:
    repository = OptimizerRepository(args.optimizer)
    ## Loading of model from file.
    model_dir = Path(args.model)
    model = _load_model_from_file(model_dir)
    ## Loading of optimizer using model instance.
    optimizer, config = OptimizerFactory.load(model, repository=repository)
    LOGGER.info("Optimizer config: " + repr(config))
    LOGGER.info("Optimizer instance: " + str(optimizer))


def main() -> None:
    import os
    import sys

    # Set up logging:
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s \t %(message)s',
        handlers=[
            logging.FileHandler("classification_optimizer.log"),
            logging.StreamHandler()
        ]
    )

    args = _get_arguments()
    if not os.path.isdir(args.optimizer):
        _build_optimizer(args)
    else:
        _load_optimizer(args)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
