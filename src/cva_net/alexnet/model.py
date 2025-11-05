import logging
import typing as t
from dataclasses import dataclass

import torch
from torch import nn

# Set up logging:
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - - \033[95m%(levelname)s\033[0m - %(message)s',
    handlers=[
        logging.FileHandler("alexnet_model.log"),
        logging.StreamHandler()
        ]
    )
LOGGER = logging.getLogger(__name__)


class AlexNet(nn.Module):
    def __init__(
        self,
        image_size: t.Tuple[int, int],
        num_channels: int=3,
        num_classes: int=1000,
        dropout: float=0.5
    ) -> None:
        super(AlexNet, self).__init__()
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
    img_size: t.Tuple[int, int] = (224, 224)
    num_channels: int = 3
    dropout: float = 0.5
    num_classes: int = 1000
    class_names: t.List[str] = None
    freeze_feature_layers: bool = False


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
        self._model_fp = self._folder / self.model_fn + '.' + self.formatting
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


class ModelFactory:
    @staticmethod
    def build_encoder(
        model_config: EncoderConfig = None,
        **kwargs: _t.Any
    ) -> TransformerEncoder:
        """
        Build a Transformer encoder model.

        Example usage:
            >>> # Build with default configuration
            >>> encoder = ModelFactory.build_encoder()
            >>>
            >>> # Build with existing configuration
            >>> config = EncoderConfig(d_model=768, num_heads=12)
            >>> encoder = ModelFactory.build_encoder(config)
            >>>
            >>> # Build with inline parameter overrides
            >>> encoder = ModelFactory.build_encoder(num_layers=8, dropout=0.2)

        :param model_config: Encoder configuration, defaults to None.
        :param kwargs: Additional parameters to override in configuration.
        :return: Configured Transformer encoder instance.
        """
        if model_config is None:
            model_config = EncoderConfig()
        model_config.__dict__.update(kwargs)
        model = TransformerEncoder(
            input_dim=model_config.input_dim,
            max_seq_len=model_config.max_seq_len, d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers, d_ff=model_config.d_ff,
            dropout=model_config.dropout
        )
        return model

    @staticmethod
    def build_decoder(
        model_config: DecoderConfig = None,
        **kwargs: _t.Any
    ) -> TransformerDecoder:
        """
        Build a Transformer decoder model.

        Example usage:
            >>> # Build with default configuration
            >>> decoder = ModelFactory.build_decoder()
            >>>
            >>> # Build with existing configuration
            >>> config = DecoderConfig(output_dim=100000, num_layers=4)
            >>> decoder = ModelFactory.build_decoder(config)
            >>>
            >>> # Build with inline parameter overrides
            >>> decoder = ModelFactory.build_decoder(vocab_size=45000, d_ff=1024)

        :param model_config: Decoder configuration, defaults to None.
        :param kwargs: Additional parameters to override in configuration.
        :return: Configured Transformer decoder instance.
        """
        if model_config is None:
            model_config = DecoderConfig()
        model_config.__dict__.update(kwargs)
        model = TransformerDecoder(
            vocab_size=model_config.vocab_size,
            max_seq_len=model_config.max_seq_len, d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers, d_ff=model_config.d_ff,
            dropout=model_config.dropout
        )
        return model

    @staticmethod
    def build(
        encoder_config: EncoderConfig = None,
        decoder_config: DecoderConfig = None
    ) -> _t.Tuple[TransformerEncoder, TransformerDecoder]:
        """
        Build both encoder and decoder models.

        Example usage:
            >>> # Build both with default configurations
            >>> encoder, decoder = ModelFactory.build()
            >>>
            >>> # Build with custom encoder configuration
            >>> enc_config = EncoderConfig(num_heads=16)
            >>> encoder, decoder = ModelFactory.build(encoder_config=enc_config)
            >>>
            >>> # Build with both custom configurations
            >>> enc_config = EncoderConfig(d_model=768)
            >>> dec_config = DecoderConfig(vocab_size=75000)
            >>> encoder, decoder = ModelFactory.build(enc_config, dec_config)

        :param encoder_config: Encoder configuration, defaults to None.
        :param decoder_config: Decoder configuration, defaults to None.
        :return: Tuple of (encoder, decoder) model instances.
        """
        encoder = ModelFactory.build_encoder(encoder_config)
        decoder = ModelFactory.build_decoder(decoder_config)
        return encoder, decoder

    @staticmethod
    def load_encoder(
        repository: ModelRepository
    ) -> _t.Tuple[TransformerEncoder, EncoderConfig]:
        """
        Load encoder model and configuration from repository.

        Example usage:
            >>> # Initialize repository
            >>> repo = ModelRepository("saved_models")
            >>>
            >>> # Load encoder with its configuration
            >>> encoder, encoder_config = ModelFactory.load_encoder(repo)
            >>>
            >>> # Use the loaded model and configuration
            >>> print(f"Encoder d_model: {encoder_config.d_model}")

        :param repository: Model repository instance.
        :return: Tuple of (loaded encoder, encoder configuration).
        """
        model_config = EncoderConfig()
        model_config = repository.load_model_config(model_config)
        model = ModelFactory.build_encoder(model_config)
        loaded_model = repository.load_model(model)
        return loaded_model, model_config

    @staticmethod
    def load_decoder(
        repository: ModelRepository
    ) -> _t.Tuple[TransformerDecoder, DecoderConfig]:
        """
        Load decoder model and configuration from repository.

        Example usage:
            >>> # Initialize repository
            >>> repo = ModelRepository("saved_models")
            >>>
            >>> # Load decoder with its configuration
            >>> decoder, decoder_config = ModelFactory.load_decoder(repo)
            >>>
            >>> # Use the loaded model and configuration
            >>> print(f"Decoder vocab_size: {decoder_config.vocab_size}")

        :param repository: Model repository instance.
        :return: Tuple of (loaded decoder, decoder configuration).
        """
        model_config = DecoderConfig()
        model_config = repository.load_model_config(model_config)
        model = ModelFactory.build_decoder(model_config)
        loaded_model = repository.load_model(model)
        return loaded_model, model_config

    @staticmethod
    def load(
        repository: ModelRepository
    ) -> _t.Tuple[
        TransformerEncoder, TransformerDecoder, EncoderConfig, DecoderConfig
    ]:
        """
        Load complete model system from repository.

        Example usage:
            >>> # Initialize repository
            >>> repo = ModelRepository("saved_models")
            >>>
            >>> # Load complete model system
            >>> encoder, decoder, enc_config, dec_config = ModelFactory.load(repo)
            >>>
            >>> # Use all loaded components
            >>> print(f"Encoder layers: {enc_config.num_layers}")
            >>> print(f"Decoder layers: {dec_config.num_layers}")

        :param repository: Model repository instance.
        :return: Tuple of (encoder, decoder, encoder_config, decoder_config).
        """
        encoder, encoder_config = ModelFactory.load_encoder(repository)
        decoder, decoder_config = ModelFactory.load_decoder(repository)
        return encoder, decoder, encoder_config, decoder_config


def _get_argument():
    import argparse

    parser = argparse.ArgumentParser()
    # batch_size: int = 1
    # max_seq_len: int = 150
    # output_dim: int = 130_000
    # d_model: int = 512
    # num_heads: int = 8
    # num_layers: int = 6
    # d_ff: int = 2048
    # dropout: float = 0.1
    parser.add_argument('action', type=str, choices=['build', 'print'])
    parser.add_argument(
        '-b', '--batch-size', type=int, default=1, help="Batch size."
    )
    parser.add_argument(
        '--input-dim', type=int, default=126, help="Input dimension."
    )
    parser.add_argument(
        '--vocab-size', type=int, default=130_000,
        help="Vocab size of output language."
    )
    parser.add_argument(
        '--encoder-max-seq-len', type=int, default=300,
        help="Max sequence length of encoder model."
    )
    parser.add_argument(
        '--decoder-max-seq-len', type=int, default=450,
        help="Max sequence length of decoder model."
    )
    parser.add_argument(
        '--d-model', type=int, default=512,
        help=(
            "The dimension of the representation space "
            "of the all transformer model."
        )
    )
    parser.add_argument(
        '--num-heads', type=int, default=8,
        help="The number of attention heads."
    )
    parser.add_argument(
        '--num-layers', type=int, default=6,
        help="The number of encoder layers and decoder layers."
    )
    parser.add_argument(
        '--d-ff', type=int, default=2048,
        help="The hidden dimension of the Pointwise feed forward."
    )
    parser.add_argument(
        '--dropout', type=float, default=0.5,
        help="The dropout probability."
    )
    parser.add_argument(
        '-o', '--output', type=str, default='outputs',
        help="The path to model directory."
    )
    # parser.add_argument(
    #     '-output', type=str, help="The path to model directory."
    # )
    return parser.parse_args()


def main() -> None:
    import sys
    args = _get_argument()
    if args.action == 'build':
        encoder_config = EncoderConfig()
        decoder_config = DecoderConfig()
        encoder_config.batch_size = args.batch_size
        encoder_config.max_seq_len = args.encoder_max_seq_len
        encoder_config.input_dim = args.input_dim
        encoder_config.d_model = args.d_model
        encoder_config.num_heads = args.num_heads
        encoder_config.num_layers = args.num_layers
        encoder_config.d_ff = args.d_ff
        encoder_config.dropout = args.dropout

        decoder_config.batch_size = args.batch_size
        decoder_config.max_seq_len = args.decoder_max_seq_len
        decoder_config.vocab_size = args.vocab_size
        decoder_config.d_model = args.d_model
        decoder_config.num_heads = args.num_heads
        decoder_config.num_layers = args.num_layers
        decoder_config.d_ff = args.d_ff
        decoder_config.dropout = args.dropout
        encoder, decoder = ModelFactory.build(encoder_config, decoder_config)

        model_repository = ModelRepository(args.output)
        model_repository.save(encoder, encoder_config)
        model_repository.save(decoder, decoder_config)

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            input_shape = (
                encoder_config.batch_size, encoder_config.max_seq_len,
                encoder_config.input_dim
            )
            x = torch.randn(input_shape)
            # Create sample tensors
            # Target sequence (what we want to generate)
            target_seq_shape = (
                encoder_config.batch_size, encoder_config.max_seq_len
            )
            target_seq = torch.randint(
                0, decoder_config.vocab_size - 1, target_seq_shape
            )
            # Encoder output (from the encoder module)
            encoder_output_shape = (
                encoder_config.batch_size, encoder_config.max_seq_len,
                encoder_config.d_model
            )
            encoder_output = torch.randn(encoder_output_shape)

            # Create look-ahead mask for decoder self-attention
            look_ahead_mask = \
                create_look_ahead_mask(encoder_config.max_seq_len)

            LOGGER.info("look_ahead_mask shape: " + str(look_ahead_mask.shape))
            LOGGER.info("look_ahead_mask:\n" + repr(look_ahead_mask))

            # Forward pass through decoder:
            # output = decoder(
            #     x=target_seq, encoder_output=encoder_output,
            #     self_attn_mask=look_ahead_mask
            # )
            _, encoder_it = print_model_summary(encoder, (x,))
            _, decoder_it = print_model_summary(
                decoder, (target_seq, encoder_output, look_ahead_mask)
            )
            print("Encoder inference time %.3f sec." % (encoder_it,))
            print("Decoder inference time %.3f sec." % (decoder_it,))

    sys.exit(0)

