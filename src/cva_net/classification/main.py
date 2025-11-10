import sys
import logging
import argparse
from pathlib import Path

from .dataset import Dataset, HDF5DatasetReader, HDF5Reader
from .optimizer import OptimizerConfig, OptimizerRepository, OptimizerFactory

# Set up logging.
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    format='\033[95m%(asctime)s\033[0m - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("classification.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)

DEFAULT_NUM_WORKERS = 4
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_GRAD_ACC = 128

DEFAULT_MODEL_FOLDER = 'saved_model'
DEFAULT_OPTIMIZER_FOLDER = 'saved_optimizer'


def get_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=42,
        default="The seed value of the random generators."
    )
    parser.add_argument(
        '-d', '--dataset', type=str, default=None,
        help="The path to the dataset file formated on HDF5."
    )
    parser.add_argument(
        '-b', '--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
        help="The batch size."
    )
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--drop-last', action="store_true")
    parser.add_argument('--pim-memory', action="store_true")

    parser.add_argument('-n', '--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--grad-acc', type=int, default=DEFAULT_GRAD_ACC)

    parser.add_argument(
        '-m', '--model', type=str, default=None,
        help='The path to the model built saved on a file.'
    )

    parser.add_argument(
        '--optimizer', type=str, default=None,
        help="The path to the file of model optimizer saved."
    )

    parser.add_argument('--device', type=str)
    parser.add_argument('-o', '--output-dir', type=str, default='outputs')

    args = parser.parse_args()
    LOGGER.info("Training arguments:")
    for arg, value in vars(args).items():
        LOGGER.info(f"\t{arg}: {value}")
    return args


def train() -> None:
    args = get_arguments()
    output = Path(args.output)
    model_folder = str(output / DEFAULT_MODEL_FOLDER)
    optimizer_folder = str(output / DEFAULT_OPTIMIZER_FOLDER)

    model_repository = None
    optimizer_repository = None

    if args.dataset is None:
        LOGGER.error("The path to the dataset file is not provided.")
        sys.exit(1)
    ds_reader = HDF5DatasetReader(args.dataset)
    ds_reader.open()
    train_dataset_source = ds_reader.get_dataset("train")
    test_dataset_source = ds_reader.get_dataset("test")
    ## Create datasets:
    train_dataset = Dataset(train_dataset_source)
    test_dataset = Dataset(test_dataset_source)

    model = None
    if args.model is None:
        from cva_net.alexnet import ModelFactory, ModelRepository

        LOGGER.warning("No model provided, we build one by default.")
        LOGGER.info("The model will be used is 'AlexNet'.")
        model = ModelFactory.build()
        model_repository = ModelRepository(model_folder)
    else:
        ...

    optimizer = None
    if args.optimizer is None:
        LOGGER.warning(
            "No optimizer provided, we build a new optimizer by default."
        )
        optimizer = OptimizerFactory.build()
        optimizer_repository = OptimizerRepository(optimizer_folder)
    else:
        ...
        

if __name__ == '__main__':
    train()
