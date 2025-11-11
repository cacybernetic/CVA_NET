import sys
import logging
import json
import argparse
from pathlib import Path

import yaml
import torch

from .dataset import Dataset, HDF5DatasetReader
from .optimizer import OptimizerRepository, OptimizerFactory
from .trainer import fit
from .utils import ModelPredictionRender

# Set up logging.
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    format='\033[96m%(asctime)s\033[0m - [%(levelname)s] - %(message)s',
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
DEFAULT_VALIDATION_RPOP = 0.4

DEFAULT_MODEL_FOLDER = 'saved_model'
DEFAULT_OPTIMIZER_FOLDER = 'saved_optimizer'


def get_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=42,
        help="The seed value of the random generators."
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
    parser.add_argument('--pin-memory', action="store_true")
    parser.add_argument(
        '--val-prop', type=float, default=DEFAULT_VALIDATION_RPOP,
        help=(
            "The proportion of the test dataset which will be take "
            "for model validation."
        )
    )

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

    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('-o', '--output-dir', type=str, default='outputs')

    args = parser.parse_args()
    LOGGER.info("Training arguments:")
    for arg, value in vars(args).items():
        LOGGER.info(f"\t{arg}: {value}")
    return args


def _load_model_from_file(model_dir):

    ## Model loading from file.
    model = None
    model_summary = None
    try:
        ### Determination the model architecture.
        model_config_file = model_dir / 'configs.yaml'
        with open(model_config_file, mode='r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
            model_arch = model_config['arch']
            LOGGER.info('Model architecture determinated: ' + str(model_arch))
        ### Import the right implementation of the model component.
        if model_arch == 'AlexNet':
            from cva_net.alexnet import ModelRepository, ModelFactory, \
                print_model_summary
            model_summary = print_model_summary
        else:
            raise NotImplementedError(
                "The model architecture named `%s` is not implemented yet."
                % (model_arch,)
            )
        ### Model weight loaded according the implementation found.
        model_repository = ModelRepository(str(model_dir))
        model, model_config = ModelFactory.load(model_repository)
        LOGGER.info("Model weights loaded successfully.")
        return model, model_config, model_repository, model_summary

    except ImportError as e:
        LOGGER.error("Error: " + str(e))
        LOGGER.info("This model architecture is unknown or not implemented.")
        sys.exit(1)


def train() -> None:
    from cva_net.utils import set_seed, get_torch_device_name

    args = get_arguments()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.device is not None:
        device = get_torch_device_name(args.device)
    set_seed(args.seed, device)

    output = Path(args.output_dir)
    output.mkdir(exist_ok=True)

    model_folder = output / DEFAULT_MODEL_FOLDER if not args.model \
        else Path(args.model)
    optimizer_folder = output / DEFAULT_OPTIMIZER_FOLDER if not args.optimizer \
        else Path(args.optimizer)

    model_repository = None
    optimizer_repository = None

    if args.dataset is None:
        LOGGER.error("The path to the dataset file is not provided.")
        sys.exit(1)
    ds_reader = HDF5DatasetReader(args.dataset)
    ds_reader.open()
    train_dataset_source = ds_reader.get_dataset("train")
    test_dataset_source = ds_reader.get_dataset("test")
    class_names = list(train_dataset_source.get_attr('class_names'))
    LOGGER.info("=" * 80)
    LOGGER.info("TRAIN DATASET:")
    LOGGER.info("=" * 80)
    LOGGER.info(str(train_dataset_source))
    LOGGER.info("=" * 80)
    LOGGER.info("TEST DATASET:")
    LOGGER.info("=" * 80)
    LOGGER.info(str(test_dataset_source))
    ## Create datasets:
    train_dataset = Dataset(train_dataset_source)
    test_dataset = Dataset(test_dataset_source)
    # num_val_samples = int(args.val_prop * len(test_dataset_source))
    # val_dataset = Dataset(test_dataset_source, end_index=num_val_samples)

    model = None
    model_config = None
    model_summary = None
    if not model_folder.is_dir():
        from cva_net.alexnet import ModelFactory, ModelRepository, \
            print_model_summary

        LOGGER.warning("No model provided, we build one by default.")
        LOGGER.info("The model will be used is 'AlexNet'.")
        model, model_config = ModelFactory.build(class_names=class_names)
        model_repository = ModelRepository(str(model_folder))
        model_summary = print_model_summary
    else:
        ret = _load_model_from_file(model_folder)
        model, model_config, model_repository, model_summary = ret
        model_config.class_names = class_names
    LOGGER.info("Model config: " + repr(model_config))
    LOGGER.info("Model instance: \n" + str(model))
    model.eval()
    with torch.no_grad():
        model_summary(
            model, model_config, batch_size=args.batch_size
        )

    optimizer = None
    optimizer_config = None
    if not optimizer_folder.is_dir():
        LOGGER.warning(
            "No optimizer provided, we build a new optimizer by default."
        )
        optimizer, optimizer_config = OptimizerFactory.build(model)
        optimizer_repository = OptimizerRepository(str(optimizer_folder))
    else:
        optimizer_repository = OptimizerRepository(str(optimizer_folder))
        optimizer, optimizer_config = OptimizerFactory.load(
            model, repository=optimizer_repository
        )
    LOGGER.info("Optimizer config: " + repr(optimizer_config))
    LOGGER.info("Optimizer instance: " + str(optimizer))

    ret = fit(
        train_dataset, model, test_dataset, val_dataset=None,
        num_epochs=args.epochs, 
        batch_size=args.batch_size, gradient_acc=args.grad_acc,
        val_prop=args.val_prop, pin_memory=args.pin_memory,
        num_workers=args.num_workers, device=args.device,
    )
    train_results, val_results, test_results = ret
    print("\ntrain_results: \n" + json.dumps(train_results, indent=4))
    print("\nval_results: \n" + json.dumps(val_results, indent=4))
    print("test_results: \n" + json.dumps(test_results, indent=4))

    model_repository.save(model, model_config)
    optimizer_repository.save(optimizer, optimizer_config)

    mpr = ModelPredictionRender(
        model=model, dataset=test_dataset, class_names=model_config.class_names,
        image_size=model_config.img_size, output_dir=str(output)
    )
    mpr.make_predictions()
    ds_reader.close()


if __name__ == '__main__':
    train()
