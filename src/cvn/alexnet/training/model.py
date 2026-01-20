import os
import random
import shutil
import time as tm
from typing import  Dict, Any, Optional, Union, Callable, List, Tuple
from dataclasses import dataclass, field
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
# Optimizer
from .optimizer.model import Optimizer, Config as OptimizerConfig
from .optimizer.factory import optimizer as build_optimizer
# Scheduler:
from .optimizer.lr_scheduler.model import LRScheduler, Config as LRSchedulerConfig
from .optimizer.lr_scheduler.factory import lr_scheduler
# Alexnet model:
from cvn.alexnet.model import AlexNet, Config as ModelConfig
from cvn.alexnet.factory import alexnet
from cvn.alexnet import repository as model_repos, summary
# Others:
from .dataset import build as build_dataset
from .loss_fn import LossFunction
from .monitor import Monitor
from .history import History
from .checkpoint import CheckpointManager
from .metrics import accuracy, precision, recall
from .layer_freezing import apply as apply_layer_freezing


@dataclass
class Config:
    seed: int = 42
    train_dataset: str = 'datasets/train'
    val_dataset: str = 'datasets/val'
    batch_size: int = 16
    image_size: int = 224
    gradient_accumulations: int = 64
    num_workers: int = 2
    amp: bool = False
    device: str = 'cuda'
    output_dir: str = 'outputs'
    checkpoint_dir: str = 'checkpoints'
    max_ckpt_to_keep: int = 3
    best_model_dir: str = 'best'
    train_curves_file: str = 'training_curves.jpeg'
    freeze_layers: List[str] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig =  field(default_factory=OptimizerConfig)
    scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


def _post_process(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Post-processing method
    ----------------------

    :param logits: [batch_size, num_classes].
    :returns: tuple of two tensors of size [batch_size,], the first tensor contains the class ids predicted
      and the second tensor contains the softmax confidences.
    """
    probs = torch.softmax(logits, dim=-1)  # [n, num_classes]
    class_ids = torch.argmax(probs, dim=-1)  # [n,]
    confidences = torch.max(probs, dim=-1).values  # [n,]
    return class_ids, confidences


def _forward_pass_step(
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlexNet,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device=None,
) -> Dict[str, torch.Tensor]:
    outputs = model.forward(x)  # noqa
    loss = criterion(outputs, y)
    predictions, confs = _post_process(outputs.detach())
    return {
        'entropy_loss': loss,
        'predictions': predictions,
        'confidences': confs}


def _forward_pass_step_with_autocast(
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlexNet,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    with autocast(device_type=device.type, dtype=torch.float16):
        results = _forward_pass_step(x, y, model, criterion)
    return results


def _train_step(
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlexNet,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    num_accumulated: int,
    num_accumulations: int,
    gradient_accumulations: int,
    avg_entropy_loss: float,
    avg_acc_score: float,
    avg_precision: float,
    avg_recall: float,
    mon: Monitor,
    scaler: GradScaler=None,
    device: torch.device=None,
    optimize: bool=False,
) -> Dict[str, Any]:
    ## Forward pass;
    res = _forward_pass_step(x, y, model, criterion)
    loss = res['entropy_loss']
    loss_value = loss.item()
    loss = loss / num_accumulations
    ## Backward pass;
    loss.backward()
    preds = res['predictions']
    num_accumulated += preds.shape[0]
    if num_accumulated >= gradient_accumulations or optimize:
        ### Optimizer step;
        optimizer.step()
        ### Reset gradient;
        optimizer.zero_grad()
        mon.print(
            "  Entropy Loss: %7.4f - Accuracy Score: %5.1f%% - Precision: %5.1f%% - Recall: %5.1f%%"
            % (avg_entropy_loss, avg_acc_score*100, avg_precision*100, avg_recall*100))
        num_accumulated = 0
    confs = res['confidences']
    return {
        'num_accumulated': num_accumulated,
        "entropy_loss": loss_value,
        "predictions": preds,
        "confidences": confs}


def _train_step_with_scaler(
    x: torch.Tensor,
    y: torch.Tensor,
    model: AlexNet,
    criterion: LossFunction,
    optimizer: Optimizer,
    num_accumulated: int,
    num_accumulations: int,
    gradient_accumulations: int,
    avg_entropy_loss: float,
    avg_acc_score: float,
    avg_precision: float,
    avg_recall: float,
    mon: Monitor,
    scaler: GradScaler=None,
    device: torch.device=None,
    optimize: bool=False,
) -> Dict[str, Any]:
    ## Forward pass;
    with autocast(device_type=device.type, dtype=torch.float16):
        res = _forward_pass_step(x, y, model, criterion)
        loss = res['entropy_loss']
        loss_value = loss.item()
        loss = loss / num_accumulations
    ## Backward pass;
    scaler.scale(loss).backward()
    preds = res['predictions']
    num_accumulated += preds.shape[0]
    if num_accumulated >= gradient_accumulations or optimize:
        ### Gradient unscaling and clipping;
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ### Optimizer step;
        scaler.step(optimizer)
        scaler.update()
        ### Reset gradient;
        optimizer.zero_grad()
        mon.print(
            "  Entropy Loss: %7.4f - Accuracy Score: %5.1f%% - Precision: %5.1f%% - Recall: %5.1f%%"
            % (avg_entropy_loss, avg_acc_score*100, avg_precision*100, avg_recall*100))
        num_accumulated = 0
        confs = res['confidences']
    return {
        'num_accumulated': num_accumulated,
        "entropy_loss": loss_value,
        "predictions": preds,
        "confidences": confs}


class Trainer:

    def __init__(self, config: Config) -> None:
        """
        Method allows to create an instance of Alexnet trainer.
        """
        self._config = config
        self.model: AlexNet = None
        self._criterion: LossFunction = None
        self.optimizer: Optimizer = None
        self.scheduler: LRScheduler = None
        self._device = None
        self._mon: Monitor = Monitor()
        # Datasets
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._train_dataset_loader: DataLoader = None
        self._val_dataset_loader: DataLoader = None
        self._test_dataset_loader: DataLoader = None
        # History And Checkpoint
        self._history: History =  History()
        self._checkpoint_manager: CheckpointManager = None
        if self._config.checkpoint_dir is not None:
            ckpt_dir = self._config.checkpoint_dir
            assert ckpt_dir, "The path of directory that will be used to make checkpoint is an empty string."
            ckpt_dir = os.path.join(self._config.output_dir, ckpt_dir)
            self._checkpoint_manager = CheckpointManager(ckpt_dir, self._config.max_ckpt_to_keep)
        self._scaler = None
        self._best_loss = float('inf')
        self._best_acc_score = -float('inf')
        self._best_precision = -float('inf')
        self._best_recall = -float('inf')
        self._best_epoch = -1
        self._start_epoch_idx = 0
        self._num_accumulations = 1
        self._train_step = None
        self._forward_pass_step = None
        self._class_names = None
        self._compiled = False
        self._checkpoint_loaded = False
        # Create metrics;
        self._accuracy = accuracy.score
        self._precision = precision.Score()
        self._recall = recall.Score()

    def load_checkpoint(self) -> Optional[int]:
        if self._checkpoint_manager is None:
            return None
        epoch = self._checkpoint_manager.get_latest_checkpoint()
        if epoch is None:
            return None
        self._mon.log("Checkpoint found.")
        self._config = self._checkpoint_manager.load_config(epoch)
        self._start_epoch_idx = epoch
        self._checkpoint_loaded = True
        return epoch

    @staticmethod
    def set_seed(seed: int, device: Union[str, torch.device]=None) -> None:
        """
        Set seeds for reproducibility.

        :param seed: An integer value to define the seed for random generator.
        :param device: The selected device.
        """
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        # np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        if (
            (isinstance(device, torch.device) and device.type == 'gpu')
            or (isinstance(device, str) and device.startswith('cuda'))
        ):
            # Also set the deterministic flag for reproducibility;
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _device_setting(self) -> None:
        if self._config.device:
            device_name = self._config.device
            if device_name.startswith('cuda'):
                if not torch.cuda.is_available():
                    self._mon.warning(f"The device named \"{device_name}\" is not available. So we select CPU device.")
                    device_name = 'cpu'
            self._device = torch.device(device_name)
        else:
            self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._mon.log(f"Device selected is \"{self._device}\".")

    def _create_dataloaders(self) -> None:
        use_pin_memory = False
        if self._device.type == 'gpu':
            use_pin_memory = True
        img_channels = self._config.model.img_channels
        datasets = build_dataset(
            train_data_dir=self._config.train_dataset, test_data_dir=self._config.val_dataset,
            img_size=self._config.image_size, img_channels=img_channels)
        self._train_dataset = datasets['train_dataset']
        self._val_dataset = datasets['val_dataset']
        self._test_dataset = datasets['test_dataset']
        ## Build data loaders;
        self._train_dataset_loader = DataLoader(
            self._train_dataset, batch_size=self._config.batch_size, shuffle=True, num_workers=self._config.num_workers,
            pin_memory=use_pin_memory)
        self._val_dataset_loader = DataLoader(
            self._val_dataset, batch_size=self._config.batch_size, shuffle=True, num_workers=self._config.num_workers,
            pin_memory=use_pin_memory)
        self._test_dataset_loader = DataLoader(
            self._test_dataset, batch_size=self._config.batch_size, shuffle=True, num_workers=self._config.num_workers,
            pin_memory=use_pin_memory)
        ## Load class names from dataset;
        if not self._checkpoint_loaded:
            self._config.model.class_names = datasets['class_names']
        self._mon.log("Training dataset:")
        self._mon.log(f"  Number of samples  > {len(self._train_dataset)}")
        self._mon.log(f"  Number of batchs   > {len(self._train_dataset_loader)}")
        self._mon.log(f"  Class names        > {datasets['class_names']}")
        self._mon.log(f"  Number of classes  > {len(datasets['class_names'])}")
        self._mon.log("Validation dataset:")
        self._mon.log(f"  Number of samples  > {len(self._val_dataset)}")
        self._mon.log(f"  Number of batchs   > {len(self._val_dataset_loader)}")
        self._mon.log("Test dataset:")
        self._mon.log(f"  Number of samples  > {len(self._test_dataset)}")
        self._mon.log(f"  Number of batchs   > {len(self._test_dataset_loader)}")

    def _instanciate_model(self) -> None:
        if self.model is None:
            if self._config.model is None:
                self._config.model = ModelConfig()
            self.model, _ = alexnet(self._config.model)
            self.model = self.model.to(self._device)

    def compile(self) -> None:
        assert self._config.train_dataset, "The directory path of training dataset is not provided."
        assert self._config.val_dataset, "The directory path of validation dataset is not provided."
        assert self.model is None or self._config.model is not None, "The model config is not specified."
        assert self.optimizer is None or self._config.optimizer is not None, "The optimizer config is not specified."
        assert self.scheduler is None or self._config.scheduler is not None, "The scheduler config is not specified."
        ## Device setting;
        self._device_setting()
        # Create dataloaders;
        self._create_dataloaders()
        ## Class names;
        self._class_names = self._config.model.class_names
        ## Logging of training config;
        self._mon.log("=" * 120)
        self._mon.log("TRAINING CONFIG")
        self._mon.log("=" * 120)
        self._mon.log(str(self._config))
        ## Setting of seed value for random generators;
        self.set_seed(self._config.seed, self._device)
        ## Instanciation of the model;
        self._instanciate_model()
        ## Instanciate criterion function;
        self._criterion = LossFunction(num_classes=len(self._class_names))
        ## Instanciation of optimizer model;
        if self.optimizer is None:
            if self._config.optimizer is None:
                self._config.optimizer = OptimizerConfig()
            self.optimizer, _ = build_optimizer(self.model, self._config.optimizer)
        self._mon.log("Optimizer:")
        self._mon.log("  Config: " + str(self.optimizer))
        ## Instanciation of scheduler model;
        if self.scheduler is None:
            if self._config.scheduler is None:
                self._config.scheduler = LRSchedulerConfig()
            self.scheduler, _ = lr_scheduler(self.optimizer, self._config.scheduler)
        self._mon.log("Scheduler:")
        self._mon.log("  Config: " + str(self.scheduler))
        if self._checkpoint_loaded:
            self._checkpoint_manager.load_data(self._start_epoch_idx, self._config, self)
            self._start_epoch_idx += 1  # We will pass to the next epoch.
        ## Model layer editions applying;
        if self._config.freeze_layers:
            self.model = apply_layer_freezing(self.model, self._config.freeze_layers)
        ## Calculate number of accumulations;
        if self._config.gradient_accumulations > self._config.batch_size:
            self._num_accumulations = self._config.gradient_accumulations // self._config.batch_size
        ## Initialize gradient scaler;
        if self._config.amp:
            self._scaler = GradScaler(device=str(self._device), enabled=True)
            self._train_step = _train_step_with_scaler
            self._forward_pass_step = _forward_pass_step_with_autocast
            self._mon.log("Mean Average Precision enabled.")
        else:
            self._train_step = _train_step
            self._forward_pass_step = _forward_pass_step
        ## Print model summary;
        model_stat, inference_time = summary.build(
            self.model, self._config.model, batchs=self._config.batch_size, device=self._device)
        self._mon.log("=" * 120)
        self._mon.log("MODEL SUMMARY")
        self._mon.log('=' * 120)
        self._mon.log(f"\n{model_stat}")
        self._mon.log(f"Inference times: {inference_time:.3f} seconds.")
        ## Specify that all is ready;
        self._compiled = True

    def state_dict(self) -> Dict[str, Any]:
        return {
            'best_loss': self._best_loss,
            'best_acc_score': self._best_acc_score,
            'best_precision': self._best_precision,
            'best_recall': self._best_recall,
            'best_epoch': self._best_epoch,
            'history': self._history.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Method of training checkpoint loading.
        """
        self._best_val_loss = state_dict['best_loss']
        self._best_acc_score = state_dict['best_acc_score']
        self._best_precision = state_dict['best_precision']
        self._best_recall = state_dict['best_recall']
        self._best_epoch = state_dict['best_epoch']
        self._history.load_state_dict(state_dict['history'])

    def train_epoch(self) -> Dict[str, Any]:
        """
        Training function on one epoch.
        """
        self.model.train()
        # Total loss
        total_entropy_loss = 0
        total_accuracy_score = 0
        total_precision = 0
        total_recall = 0
        # Avg loss
        avg_entropy_loss = 0
        avg_accuracy_score = 0
        avg_precision = 0
        avg_recall = 0
        num_accumulated = 0
        total_batchs = len(self._train_dataset_loader)
        self._mon.create_pbar(total_batchs, desc="\033[96mTraining\033[0m")
        self.optimizer.zero_grad()
        for num_batchs, batch_data in enumerate(self._train_dataset_loader, 1):
            x, y = batch_data
            x = x.to(self._device)
            y = y.to(self._device)
            ## Forward pass;
            # optimize=(num_batchs>=total_batchs): if it is the last batchs then we compute optimization.
            results = self._train_step(
                x=x, y=y, model=self.model, criterion=self._criterion, optimizer=self.optimizer,
                num_accumulated=num_accumulated,
                num_accumulations=self._num_accumulations, gradient_accumulations=self._config.gradient_accumulations,
                avg_entropy_loss=avg_entropy_loss, avg_acc_score=avg_accuracy_score, avg_precision=avg_precision,
                avg_recall=avg_recall, mon=self._mon, scaler=self._scaler, device=self._device,
                optimize=(num_batchs >= total_batchs))
            num_accumulated = results['num_accumulated']
            ## Statistiques;
            total_entropy_loss += results['entropy_loss']
            total_accuracy_score += self._accuracy(results['predictions'], y)
            total_precision += self._precision(results['predictions'], y)
            total_recall += self._recall(results['predictions'], y)
            ## Calculate average;
            avg_entropy_loss = total_entropy_loss / num_batchs
            avg_accuracy_score = (total_accuracy_score / num_batchs).item()
            avg_precision = (total_precision / num_batchs).item()
            avg_recall = (total_recall / num_batchs).item()
            self._mon.pbar.set_postfix(
                {
                    'loss': f'{avg_entropy_loss:7.4f}',
                    'score': f'{avg_accuracy_score:6.4f}',
                    'P': f'{avg_precision:6.4f}',
                    'R': f'{avg_recall:6.4f}'
                }
            )
            self._mon.pbar.update(1)
        self._mon.close_pbar()
        return {
            'entropy_loss': avg_entropy_loss,
            'accuracy_score': avg_accuracy_score,
            'precision': avg_precision,
            'recall': avg_recall}

    def validate(self) -> Dict[str, Any]:
        """
        Validation function on one epoch.
        """
        self.model.eval()
        # Total loss
        total_entropy_loss = 0
        total_accuracy_score = 0
        total_precision = 0
        total_recall = 0
        # Avg loss
        avg_entropy_loss = 0
        avg_accuracy_score = 0
        avg_precision = 0
        avg_recall = 0
        self._mon.create_pbar(len(self._val_dataset_loader), desc="\033[93mValidation\033[0m")
        with torch.no_grad():
            for num_batchs, batch_data in enumerate(self._val_dataset_loader, 1):
                x, y = batch_data
                x = x.to(self._device)
                y = y.to(self._device)
                ## Forward pass and entropy loss compute;
                results = self._forward_pass_step(x, y, self.model, self._criterion, self._device)
                ## Statistiques;
                total_entropy_loss += results['entropy_loss'].detach().item()
                total_accuracy_score += self._accuracy(results['predictions'], y)
                total_precision += self._precision(results['predictions'], y)
                total_recall += self._recall(results['predictions'], y)
                ## Calculate average;
                avg_entropy_loss = total_entropy_loss / num_batchs
                avg_accuracy_score = (total_accuracy_score / num_batchs).item()
                avg_precision = (total_precision / num_batchs).item()
                avg_recall = (total_recall / num_batchs).item()
                self._mon.pbar.set_postfix(
                    {
                        'loss': f'{avg_entropy_loss:7.4f}',
                        'score': f'{avg_accuracy_score:6.4f}',
                        'P': f'{avg_precision:6.4f}',
                        'R': f'{avg_recall:6.4f}'
                    }
                )
                self._mon.pbar.update(1)
            self._mon.close_pbar()
            return {
                'entropy_loss': avg_entropy_loss,
                'accuracy_score': avg_accuracy_score,
                'precision': avg_precision,
                'recall': avg_recall}

    def execute(self, num_epochs: int=1) -> History:
        assert self._compiled is True, (
            "You must call the method called `compile()` in first, before call the `execute()` method.")
        # Training
        self._mon.log(f"{'=' * 120}")
        self._mon.log(f"STARTING OF ALEXNET TRAINING - {num_epochs} EPOCHS")
        self._mon.log(f"Start training at epoch number \"{self._start_epoch_idx + 1}\".")
        self._mon.log(f"Start training on device \"{self._device}\".")
        self._mon.log(f"{'=' * 120}\n")
        train_curves_file = os.path.join(self._config.output_dir, self._config.train_curves_file)
        os.makedirs(self._config.output_dir, exist_ok=True)
        for epoch in range(self._start_epoch_idx, num_epochs):
            self._mon.log(f"Epoch {epoch + 1}/{num_epochs}")
            ## train on one epoch;
            train_results = self.train_epoch()
            self._history.append_train(
                entropy_loss=train_results['entropy_loss'], accuracy_score=train_results['accuracy_score'],
                precision=train_results['precision'], recall=train_results['recall'])
            ## Validation;
            val_results = self.validate()
            self._history.append_val(
                entropy_loss=val_results['entropy_loss'], accuracy_score=val_results['accuracy_score'],
                precision=val_results['precision'], recall=val_results['recall'])
            ## Scheduler;
            self.scheduler.step()
            ## Print results;
            args1 = (
                train_results['entropy_loss'], train_results['accuracy_score']*100, train_results['precision']*100,
                train_results['recall']*100)
            args2 = (
                val_results['entropy_loss'], val_results['accuracy_score']*100, val_results['precision']*100,
                val_results['recall']*100)
            self._mon.log(f"        \t Entropy \t Accuracy score \t Precision \t Recall ")
            self._mon.log("Training \t %7.4f \t %7.3f       \t %7.3f   \t %7.3f" % args1)
            self._mon.log("Val      \t %7.4f \t %7.3f       \t %7.3f   \t %7.3f" % args2)
            ## Save the best model weights;
            if (
                val_results['entropy_loss'] <= self._best_loss
                and val_results['accuracy_score'] >= self._best_acc_score
                and val_results['precision'] >= self._best_precision
                and val_results['recall'] >= self._best_recall
            ):
                if (
                    val_results['entropy_loss'] == self._best_loss
                    and val_results['accuracy_score'] == self._best_acc_score
                    and val_results['precision'] == self._best_precision
                    and val_results['recall'] == self._best_recall
                ):
                    self._mon.log("Case where this model acc is equal to the last best model acc.")
                    continue
                self._best_loss = val_results['entropy_loss']
                self._best_acc_score = val_results['accuracy_score']
                self._best_precision = val_results['precision']
                self._best_recall = val_results['recall']
                curr_best_model_dir = os.path.join(self._config.output_dir, f"{self._config.best_model_dir}_{epoch:0d}")
                model_repos.save_config(self._config.model, dir_path=curr_best_model_dir)
                model_repos.save_data(self.model, dir_path=curr_best_model_dir, device_type=self._device.type)
                tm.sleep(1)
                ## Remove the old best model dir after 1 sec;
                old_best_model_dir = os.path.join(
                    self._config.output_dir, f"{self._config.best_model_dir}_{self._best_epoch:0d}")
                if os.path.isdir(old_best_model_dir):
                    shutil.rmtree(old_best_model_dir)
                ## Keep the current epoch asd best;
                self._best_epoch = epoch
                self._mon.log("✓ Best model saved!")
            ## Make a checkpoint;
            if self._checkpoint_manager is not None:
                self._checkpoint_manager.save_config(epoch, self._config)
                self._checkpoint_manager.save_data(epoch, self, device_type=self._device.type)
            ## Plotting of training progression into image file;
            self._history.plot(train_curves_file)
            self._mon.log("Training curves is plotted at \"" + train_curves_file + "\".")
        ## Generate visual validations;
        return self._history
