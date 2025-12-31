from typing import Tuple
from torch.utils.data import Dataset
from .model import JEPATrainer, Config as JEPATrainerConfig
from cva_net.alexnet.jepa.model import JEPA, Config as JEPAConfig
from .optimizer.model import Optimizer, Config as OptimizerConfig
from .optimizer.lr_scheduler.model import LRScheduler, Config as LRSchedulerConfig


def jepa_trainer(
    model: JEPA,
    model_config: JEPAConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    optimizer: Optimizer=None,
    optimizer_config: OptimizerConfig=None,
    scheduler: LRScheduler=None,
    scheduler_config: LRSchedulerConfig=None,
    config: JEPATrainerConfig=None,
    **kwargs
) -> Tuple[JEPATrainer, JEPATrainerConfig]:
    if config is None:
        config = JEPATrainerConfig()
    config.__dict__.update(kwargs)
    pass
