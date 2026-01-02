import logging
from typing import Tuple
import yaml
from cvanet.alexnet.backbone.model import Config as BackboneConfig
from cvanet.alexnet.jepa.model import Config as ModelConfig
from cvanet.alexnet.jepa.training.model import Config as TrainingConfig
from .cmdparser import parse_args

LOGGER = logging.getLogger(__name__)
TASKS = ('jepa',)
MODES = ('train', 'eval', 'predict')


def _train_jepa(args) -> None:

    # Create the backbone config;
    backbone_config = BackboneConfig()
    if 'imch' in args:
        backbone_config.num_channels = int(args['imch'])
    if 'latent_dim' in args:
        backbone_config.latent_dim = int(args['latent_dim'])
    # Create the model config;
    model_config = ModelConfig()
    model_config.latent_dim = backbone_config.latent_dim
    if 'ema_tau_min' in args:
        model_config.ema_tau_min = float(args['ema_tau_min'])
    if 'ema_tau_max' in args:
        model_config.ema_tau_max = float(args['ema_tau_max'])
    if 'ema_total_steps' in args:
        model_config.ema_total_steps = int(args['ema_total_steps'])
    model_config.backbone = backbone_config
    # Create the training config;
    training_config = TrainingConfig()
    if 'device' in args:
        training_config.device = args['device']
    if 'batchs' in args:
        training_config.batch_size = int(args['batchs'])
    if 'accumulation' in args:
        training_config.gradient_accumulation = int(args['accumulation'])
    if 'num_workers' in args:
        training_config.num_workers = int(args['num_workers'])
    if 'amp' in args:
        training_config.amp = bool(args['amp'])
    if 'output' in args:
        training_config.output_dir = args['output']
    if 'best_model' in args:
        training_config.best_model_dir = args['best_model']
    if 'checkpoints' in args:
        training_config.checkpoint_dir = args['checkpoints']
    if 'max_ckpts' in args:
        training_config.max_ckpt_to_keep = int(args['max_ckpts'])
    if 'data_train' in args:
        training_config.train_dataset = args['data_train']
    if 'data_val' in args:
        training_config.val_dataset = args['data_val']
    if 'imgsz' in args:
        training_config.image_size = int(args['imgsz'])



def main() -> None:
    args = parse_args()
    task = args['1'].lower()
    mode = args['2'].lower()
    if task not in TASKS:
        print("The task named\"", task, "\"is not implemented yet.")
        exit(0)
    if mode not in MODES:
        print("The mode named\"", mode, "\"is not supported.")
        exit(0)
    # Get parametters;
    params = {name:val for name, val in args.items() if name not in ('1', '2')}
    LOGGER.info("=" * 120)
    LOGGER.info("LIST OF ARGUMENTS")
    for name, val in params.items():
        LOGGER.info("  " + name + ": " + val)
    # Call corresponding operation;
    if task == 'jepa' and mode == 'train':
        _train_jepa(params)
