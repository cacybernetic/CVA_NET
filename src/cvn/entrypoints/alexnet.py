import ast
import logging
from cvn.alexnet.model import Config as ModelConfig
from cvn.alexnet.training.model import Config as TrainingConfig
from cvn.alexnet.training.factory import model_trainer
from cvn.alexnet.training.optimizer.model import Config as OptimizerConfig

from .cmdparser import parse_args

LOGGER = logging.getLogger(__name__)
TASKS = ('classification',)
MODES = ('train', 'eval', 'predict')


def _train_alexnet(args) -> None:

    # Create the model config;
    model_config = ModelConfig()
    if 'classes' in args:
        model_config.class_names = [c.strip() for c in args['classes'].split(',') if c.strip()]
    if 'imchs' in args:
        model_config.img_channels = int(args['imchs'])
    if 'dropout' in args:
        model_config.dropout = float(args['dropout'])
    # Create the optimizer config;
    optimizer_config = OptimizerConfig()
    if 'lr0' in args:
        optimizer_config.lr0 = float(args['lr0'])
    if 'optimizer' in args:
        optimizer_config.optimizer = args['optimizer']
    if 'weight_decay' in args:
        optimizer_config.weight_decay = float(args['weight_decay'])
    if 'eps' in args:
        optimizer_config.eps = float(args['eps'])
    if 'momentum' in args:
        optimizer_config.momentum = float(args['momentum'])
    if 'dampening' in args:
        optimizer_config.dampening = float(args['dampening'])
    if 'betas' in args:
        bx = args['betas'].split(',')
        b1 = float(bx[0])
        b2 = float(bx[1])
        optimizer_config.betas = (b1, b2)
    # Create the training config;
    training_config = TrainingConfig()
    if 'seed' in args:
        training_config.seed = int(args['seed'])
    if 'device' in args:
        training_config.device = args['device']
    if 'batchs' in args:
        training_config.batch_size = int(args['batchs'])
    if 'gradient_accumulations' in args:
        training_config.gradient_accumulations = int(args['gradient_accumulations'])
    if 'amp' in args:
        training_config.amp = ast.literal_eval(args['amp'])
    if 'output' in args:
        training_config.output_dir = args['output']
    if 'best_model' in args:
        training_config.best_model_dir = args['best_model']
    if 'checkpoints' in args:
        training_config.checkpoint_dir = args['checkpoints']
    if 'max_ckpts' in args:
        training_config.max_ckpt_to_keep = int(args['max_ckpts'])
    if 'train_data' in args:
        training_config.train_dataset = args['train_data']
    if 'val_data' in args:
        training_config.val_dataset = args['val_data']
    if 'imgsz' in args:
        training_config.image_size = int(args['imgsz'])
    if 'workers' in args:
        training_config.num_workers = int(args['workers'])
    training_config.model = model_config
    training_config.optimizer = optimizer_config
    num_epochs = 2
    if 'epochs' in args:
        num_epochs = int(args['epochs'])
    trn, _ = model_trainer(training_config)
    trn.load_checkpoint()
    trn.compile()
    trn.execute(num_epochs)


def main() -> None:
    args = parse_args()
    mode = args['1'].lower()
    if mode not in MODES:
        print("The mode named\"", mode, "\"is not supported.")
        exit(0)
    # Get parametters;
    params = {name:val for name, val in args.items() if name not in ('1',)}
    LOGGER.info("=" * 120)
    LOGGER.info("LIST OF ARGUMENTS")
    for name, val in params.items():
        LOGGER.info("  " + name + ": " + val)
    # Call corresponding operation;
    if mode == 'train':
        _train_alexnet(params)
