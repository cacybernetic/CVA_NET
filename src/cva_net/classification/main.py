import logging
import argparse

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
    return args


def main() -> None:
    ...


if __name__ == '__main__':
    main()
