import logging
import typing as t
from dataclasses import dataclass

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class AlexNet(nn.Module):
    def __init__(
        self,
        image_size: t.Tuple[int, int],
        num_channels: int=3,
        num_classes: int=1000,
        dropout: float=0.5,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                num_channels, 96, kernel_size=11, stride=4, padding=0
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
    
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        features_map_size = self._evaluate_features_map_shape(
            num_channels, image_size
        )
        # feature_map_dim = 1
        # for dim in features_map_size[1:]:
        #     feature_map_dim *= dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(features_map_size[1], 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def _evaluate_features_map_shape(
        self,
        num_channels: int,
        image_size: t.Tuple[int, int]
    ) -> t.Tuple[int, int, int, int]:
        x = torch.zeros((1, num_channels, *image_size))
        y = self.features(x)
        y = y.reshape(y.shape[0], -1)
        LOGGER.info("Feature map shape: " + str(y.shape))
        return y.shape

    def forward(self, x):
        # LOGGER.info(str(x.shape))
        # exit(0)
        x = self.features(x)
        # LOGGER.info(str(x.shape))
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out


@dataclass
class ModelConfig:
    arch: str = 'AlexNet'
    img_size: t.Tuple[int, int] = (224, 224)
    num_channels: int = 3
    dropout: float = 0.5
    num_classes: int = 1000
    class_names: t.List[str] = None
    freeze_feature_layers: bool = False


def initialize_weights(m: AlexNet) -> None:
    """Function of model weights initialization."""
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        LOGGER.debug("CONV2D weights:\n" + str(m.weight[0]))
        LOGGER.debug("CONV2D bias:\n" + str(m.bias[0]))
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.constant_(m.bias, 0)
        LOGGER.debug("LINEAR weights:\n" + str(m.weight[0]))
        LOGGER.debug("LINEAR bias:\n" + str(m.bias[0]))


###############################################################################
# MODEL ARCHITECTURE PRINTING
###############################################################################

def print_model_summary(
    model: AlexNet,
    config: ModelConfig,
    batch_size: int=1,
    depth: int=8,
    device: t.Union[str, torch.device]='cpu'
):
    """
    This function to make summary for the model instance received
    by arguments.
    """
    named_parameters = model.state_dict()
    print("=" * 80)
    print(model.__class__.__name__ + " model parameters:")
    for name, param in named_parameters.items():
        print("\t" + name + " \t " + str(param.shape))

    import time as tm
    from torchinfo import summary

    input_shape = (batch_size, config.num_channels, *config.img_size)
    input_data = torch.randn(input_shape)

    start = tm.time()
    state = summary(
        model=model, input_data=input_data, device=device, depth=depth,
        col_names=(
            "input_size", "output_size", "num_params", "params_percent",
            "trainable",
        )
    )
    end = tm.time()
    inference_time = (end - start)
    return state, inference_time


###############################################################################
# MODEL REPOSITORY
###############################################################################
from pathlib import Path
import yaml


class ModelRepository:
    MODEL_FN: str = 'model'
    CONFIG_FN: str = 'configs.yaml'

    def __init__(
        self,
        model_folder: str,
        model_fn: str = MODEL_FN,
        config_fn: str = CONFIG_FN,
        formatting: t.Literal['pt', 'onnx']='pt',
    ) -> None:
        self.model_fn = model_fn
        self.config_fn = config_fn
        self.formatting = formatting

        self._folder: Path = None
        self._model_fp: Path = None
        self._conf_fp: Path = None
        self.set_model_folder(model_folder)

    def set_model_folder(self, folder: str) -> None:
        """
        Set the model storage folder and update all file paths.

        :param folder: Root directory path for model storage.
        """
        self._folder = Path(folder)
        self._model_fp = self._folder / (self.model_fn + '.' + self.formatting)
        self._conf_fp = self._folder / self.config_fn

    @staticmethod
    def save_weights(m: nn.Module, file_path: Path) -> None:
        """
        Save model weights to specified file path.

        :param m: PyTorch model module to save.
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
        Load model weights from specified file path.

        :param file_path: Source file path for weights.
        :param map_location: Device to load weights onto, defaults to CPU.
        :return: Dictionary containing model state.
        :raises FileNotFoundError: If weights file does not exist.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(
                "No such model file at \"%s\"." % (str(file_path),)
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
                "No such model file at \"%s\"." % (str(file_path),)
            )
        with open(file=file_path, mode='r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config

    def save(self, model: AlexNet, config: ModelConfig) -> None:
        self.save_weights(model, self._model_fp)
        self.save_config(config.__dict__, self._conf_fp)

    def load_model(self, model: AlexNet) -> AlexNet:
        weights = self.load_weights(self._model_fp)
        model.load_state_dict(weights)
        return model

    def load_model_config(self, config: ModelConfig) -> ModelConfig:
        config_dict = self.load_config(self._conf_fp)
        config.__dict__.update(config_dict)
        return config


###############################################################################
# MODEL FACTORY
###############################################################################

class ModelFactory:
    
    @staticmethod
    def build(
        config: ModelConfig = None, 
        **kwargs: t.Dict[str, t.Any]
    ) -> t.Tuple[AlexNet, ModelConfig]:
        if config is None:
            config = ModelConfig()
        config.__dict__.update(kwargs)
        model = AlexNet(
            image_size=config.img_size, num_channels=config.num_channels,
            num_classes=config.num_classes, dropout=config.dropout
        )
        model.apply(initialize_weights)
        return model, config

    @staticmethod
    def load(repository: ModelRepository) -> t.Tuple[AlexNet, ModelConfig]:
        model_config = ModelConfig()
        model_config = repository.load_model_config(model_config)
        model, _ = ModelFactory.build(model_config)
        loaded_model = repository.load_model(model)
        return loaded_model, model_config


###############################################################################
# MAIN IMPLEMENTATION
###############################################################################


def _get_argument():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['build', 'print'])
    parser.add_argument(
        '-b', '--batch-size', type=int, default=1, help="Batch size."
    )
    parser.add_argument('--image-size', nargs=2, type=int, default=(224, 224))
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--class-names', nargs='+', type=str, default=[])
    parser.add_argument(
        '-nc', '--num-classes', type=int, default=1000,
        help="The number of the classes."
    )
    parser.add_argument(
        '-m', '--model', type=str, default='outputs/saved_model',
        help="The path to model directory."
    )
    # parser.add_argument(
    #     '-output', type=str, help="The path to model directory."
    # )
    return parser.parse_args()


def main() -> None:
    import sys

    # Set up logging:
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s \t %(message)s',
        handlers=[
            logging.FileHandler("alexnet_model.log"),
            logging.StreamHandler()
        ]
    )
    args = _get_argument()

    if args.action == 'build':
        model_config = ModelConfig()
        model_config.img_size = args.image_size
        model_config.num_channels = args.num_channels
        model_config.class_names = args.class_names
        model_config.num_classes = args.num_classes
        model_config.dropout = args.dropout
        model, _ = ModelFactory.build(model_config)
        model_repository = ModelRepository(args.model)
        model_repository.save(model, model_config)

    elif args.action == 'print':
        model_repository = ModelRepository(args.model)
        model, model_config = ModelFactory.load(model_repository)

    LOGGER.info("Model config: " + repr(model_config))
    LOGGER.info("Model modules: " + str(model))
    model.eval()
    with torch.no_grad():
        _, model_it = print_model_summary(
            model, config=model_config, batch_size=args.batch_size
        )
        LOGGER.info("Encoder inference time %.3f sec." % (model_it,))

    sys.exit(0)
