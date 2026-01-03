import os
from typing import  Dict, Any, Optional, Union
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
# JEPA model:
from cvanet.alexnet.jepa.model import JEPA, Config as JEPAConfig
from cvanet.alexnet.jepa.factory import jepa
from cvanet.alexnet.jepa import repository as jepa_repos, summary
# Others:
from .dataset import custom_dataloaders
from .loss_fn import compute_loss
from .monitor import Monitor
from .history import History
from .checkpoint import CheckpointManager


@dataclass
class Config:
    seed: int = 42
    train_dataset: str = 'datasets/train'
    val_dataset: str = 'datasets/val'
    batch_size: int = 32
    image_size: int = 224
    gradient_accumulations: int = 128
    num_workers: int = 2
    amp: bool = False
    device: str = 'cpu'
    output_dir: str = 'alexnet-jepa'
    checkpoint_dir: str = 'jepa-ckpts'
    max_ckpt_to_keep: int = 3
    best_model_dir: str = 'best'
    train_curves_file: str = 'training_curves.jpeg'
    model: JEPAConfig = field(default_factory=JEPAConfig)
    optimizer: OptimizerConfig =  field(default_factory=OptimizerConfig)
    scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


def _train_step(
    x: torch.Tensor,
    y: torch.Tensor,
    model: JEPA,
    optimizer: Optimizer,
    num_accumulated: int,
    num_accumulations: int,
    gradient_accumulations: int,
    avg_loss: float,
    avg_mse: float,
    avg_cosine: float,
    mon: Monitor,
    scaler: GradScaler=None,
    device: torch.device=None,
) -> Dict[str, Any]:
    # Forward pass;
    predicted, target, _ = model(x, y)  # noqa
    loss, mse, cosine = compute_loss(predicted, target)
    loss_value = loss.item()
    loss = loss / num_accumulations
    loss.backward()
    num_accumulated += predicted.shape[0]
    if num_accumulated >= gradient_accumulations:
        ### Optimizer step;
        optimizer.step()
        ### Update EMA;
        model.update_target_encoder()
        ### Reset gradient;
        optimizer.zero_grad()
        mon.print(" * Total loss %7.4f - MSE loss %7.4f - Cosine loss %7.4f" % (avg_loss, avg_mse, avg_cosine))
        num_accumulated = 0
    return {'num_accumulated': num_accumulated, "loss_value": loss_value, 'mse': mse.item(), 'cosine': cosine.item()}


def _train_step_with_scaler(
    x: torch.Tensor,
    y: torch.Tensor,
    model: JEPA,
    optimizer: Optimizer,
    num_accumulated: int,
    num_accumulations: int,
    gradient_accumulation: int,
    avg_loss: float,
    avg_mse: float,
    avg_cosine: float,
    mon: Monitor,
    scaler: GradScaler,
    device: torch.device,
) -> Dict[str, Any]:
    # Forward pass;
    with autocast(str(device), dtype=torch.float16):
        predicted, target, _ = model(x, y)  # noqa
        loss, mse, cosine = compute_loss(predicted, target)
        loss_value = loss.item()
        loss = loss / num_accumulations
    scaler.scale(loss).backward()
    num_accumulated += predicted.shape[0]
    if num_accumulated >= gradient_accumulation:
        ### Gradient unscaling and clipping;
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ### Optimizer step;
        scaler.step(optimizer)
        scaler.update()
        ### Update EMA;
        # model.update_target_encoder()
        ### Reset gradient;
        optimizer.zero_grad()
        mon.print(" * Total loss %7.4f - MSE loss %7.4f - Cosine loss %7.4f" % (avg_loss, avg_mse, avg_cosine))
        num_accumulated = 0
    return {'num_accumulated': num_accumulated, "loss_value": loss_value, 'mse': mse.item(), 'cosine': cosine.item()}


def _forward_pass_step(x, y, model: JEPA, device: torch.device=None) -> Dict[str, Any]:
    predicted, target, _ = model(x, y)  # noqa
    loss, mse, cosine = compute_loss(predicted, target)
    return {'loss': loss.item(), 'mse': mse.item(), 'cosine': cosine.item()}


def _forward_pass_step_with_autocast(x, y, model: JEPA, device: torch.device) -> Dict[str, Any]:
    with autocast(str(device), dtype=torch.float16):
        predicted, target, _ = model(x, y)  # noqa
        loss, mse, cosine = compute_loss(predicted, target)
    return {'loss': loss.item(), 'mse': mse.item(), 'cosine': cosine.item()}


class JEPATrainer:

    def __init__(self, config: Config) -> None:
        """
        Method allows to create an instance of JEPA trainer.
        """
        self._config = config
        self.model: JEPA = None
        self.optimizer: Optimizer = None
        self.scheduler: LRScheduler = None
        self._device = None
        self._mon: Monitor = Monitor()
        self._train_dataset_loader: DataLoader = None
        self._val_dataset_loader: DataLoader = None
        self._history: History =  History()
        self._checkpoint_manager: CheckpointManager = None
        if self._config.checkpoint_dir is not None:
            ckpt_dir = self._config.checkpoint_dir
            assert ckpt_dir, "The path of directory that will be used to make checkpoint is an empty string."
            ckpt_dir = os.path.join(self._config.output_dir, ckpt_dir)
            self._checkpoint_manager = CheckpointManager(ckpt_dir, self._config.max_ckpt_to_keep)
        self._scaler = None
        self._best_val_loss = float('inf')
        self._start_epoch_idx = 0
        self._num_accumulations = 1
        self._train_step = None
        self._forward_pass_step = None
        self._compiled = False
        self._checkpoint_loaded = False

    def load_checkpoint(self) -> Optional[int]:
        if self._checkpoint_manager is None:
            return None
        epoch = self._checkpoint_manager.get_latest_checkpoint()
        if not epoch:
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
        # random.seed(seed)
        # np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if (
            (isinstance(device, torch.device) and device.type == 'gpu')
            or (isinstance(device, str) and device.startswith('cuda'))
        ):
            # Also set the deterministic flag for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _device_setting(self) -> None:
        if self._config.device:
            device_name = self._config.device
            if device_name.startswith('cuda'):
                if not torch.cuda.is_available():
                    self._mon.log(f"The device named \"{device_name}\" is not available. So we select CPU device.")
                    device_name = 'cpu'
            self._device = torch.device(device_name)
        else:
            self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._mon.log(f"Device selected is \"{self._device}\".")

    def _create_dataloaders(self) -> None:
        use_pin_memory = False
        if self._device.type == 'gpu':
            use_pin_memory = True
        dataloaders = custom_dataloaders(
            train_data_dir=self._config.train_dataset, val_data_dir=self._config.val_dataset,
            img_size=self._config.image_size, batch_size=self._config.batch_size, num_workers=self._config.num_workers,
            pin_memory=use_pin_memory)
        self._train_dataset_loader = dataloaders[0]
        self._val_dataset_loader = dataloaders[1]
        self._mon.log("Training dataset:")
        self._mon.log(f"  Number of batchs -> {len(self._train_dataset_loader)}")
        self._mon.log("Validation dataset:")
        self._mon.log(f"  Number of batchs -> {len(self._val_dataset_loader)}")

    def _instanciate_model(self) -> None:
        if self.model is None:
            if self._config.model is None:
                self._config.model = JEPAConfig()
            self.model, _ = jepa(self._config.model)
            self.model = self.model.to(self._device)
        model_stat, inference_time = summary.build(self.model, device=self._device)
        self._mon.log("=" * 120)
        self._mon.log("MODEL SUMMARY")
        self._mon.log(f"{"=" * 120}")
        self._mon.log(f"\n{model_stat}")
        self._mon.log(f"Inference times: {inference_time:.3f} seconds.")

    def compile(self) -> None:
        assert self._config.train_dataset, "The directory path of training dataset is not provided."
        assert self._config.val_dataset, "The directory path of validation dataset is not provided."
        assert self.model is None or self._config.model is not None, "The model config is not specified."
        assert self.optimizer is None or self._config.optimizer is not None, "The optimizer config is not specified."
        assert self.scheduler is None or self._config.scheduler is not None, "The scheduler config is not specified."
        # Logging of training config;
        self._mon.log("=" * 120)
        self._mon.log("TRAINING CONFIG")
        self._mon.log(f"{"=" * 120}")
        self._mon.log(str(self._config))
        # Device setting;
        self._device_setting()
        # Setting of seed value for random generators;
        self.set_seed(self._config.seed, self._device)
        # Create dataloaders;
        self._create_dataloaders()
        # Instanciation of the model;
        self._instanciate_model()
        # Instanciation of optimizer model;
        if self.optimizer is None:
            if self._config.optimizer is None:
                self._config.optimizer = OptimizerConfig()
            self.optimizer, _ = build_optimizer(self.model, self._config.optimizer)
        self._mon.log("Optimizer:")
        self._mon.log("  Config: " + str(self.optimizer))
        # Instanciation of scheduler model;
        if self.scheduler is None:
            if self._config.scheduler is None:
                self._config.scheduler = LRSchedulerConfig()
            self.scheduler, _ = lr_scheduler(self.optimizer, self._config.scheduler)
        self._mon.log("Scheduler:")
        self._mon.log("  Config: " + str(self.scheduler))
        if self._checkpoint_loaded:
            self._checkpoint_manager.load_data(self._start_epoch_idx, self._config, self)
            self._start_epoch_idx += 1  # We will pass to the next epoch.
        # Calculate number of accumulations;
        if self._config.gradient_accumulations > self._config.batch_size:
            self._num_accumulations = self._config.gradient_accumulations // self._config.batch_size
        # Initialize gradient scaler;
        if self._config.amp:
            self._scaler = GradScaler(device=str(self._device), enabled=True)
            self._train_step = _train_step_with_scaler
            self._forward_pass_step = _forward_pass_step_with_autocast
            self._mon.log("Mean Average Precision enable.")
        else:
            self._train_step = _train_step
            self._forward_pass_step = _forward_pass_step
        # Specify that all is ready;
        self._compiled = True

    def state_dict(self) -> Dict[str, Any]:
        return {
            'best_val_loss': self._best_val_loss,
            'history': self._history.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Method of training checkpoint loading.
        """
        self._best_val_loss = state_dict['best_val_loss']
        self._history.load_state_dict(state_dict['history'])

    def train_epoch(self) -> Dict[str, Any]:
        """
        Training function on one epoch.
        """
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_cosine = 0
        avg_loss = 0
        avg_mse = 0
        avg_cosine = 0
        num_accumulated = 0
        self._mon.create_pbar(len(self._train_dataset_loader), desc="Training")
        self.optimizer.zero_grad()
        for num_batchs, batch_data in enumerate(self._train_dataset_loader, 1):
            view1, view2 = batch_data
            view1, view2 = view1.to(self._device), view2.to(self._device)
            # Forward pass;
            results = self._train_step(
                x=view1, y=view1, model=self.model, optimizer=self.optimizer, num_accumulated=num_accumulated,
                num_accumulations=self._num_accumulations, gradient_accumulations=self._config.gradient_accumulations,
                avg_loss=avg_loss, avg_mse=avg_mse, avg_cosine=avg_cosine, mon=self._mon, scaler=self._scaler,
                device=self._device)
            num_accumulated = results['num_accumulated']
            # Statistiques;
            total_loss += results['loss_value']
            total_mse += results['mse']
            total_cosine += results['cosine']
            ## Calculate average;
            avg_loss = total_loss / num_batchs
            avg_mse = total_mse / num_batchs
            avg_cosine = total_cosine / num_batchs
            self._mon.pbar.set_postfix(
                {
                    'loss': f'{avg_loss:.4f}',
                    'mse': f'{avg_mse:.4f}',
                    'cos': f'{avg_cosine:.4f}'
                }
            )
            self._mon.pbar.update(1)
        self._mon.close_pbar()
        return {
            'total_loss': avg_loss,
            'mse_loss': avg_mse,
            'cosine_loss': avg_cosine,
        }

    def validate(self) -> Dict[str, Any]:
        """
        Validation function on one epoch.
        """
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_cosine = 0
        avg_loss = 0
        avg_mse = 0
        avg_cosine = 0
        self._mon.create_pbar(len(self._val_dataset_loader), desc="Validation")
        with torch.no_grad():
            for num_batchs, batch_data in enumerate(self._val_dataset_loader, 1):
                view1, view2 = batch_data
                view1, view2 = view1.to(self._device), view2.to(self._device)
                # Forward pass;
                results = self._forward_pass_step(x=view1, y=view2, model=self.model, device=self._device)
                # Statistiques;
                total_loss += results['loss']
                total_mse += results['mse']
                total_cosine += results['cosine']
                ## Calculate average;
                avg_loss = total_loss / num_batchs
                avg_mse = total_mse / num_batchs
                avg_cosine = total_cosine / num_batchs
                self._mon.pbar.set_postfix(
                    {
                        'loss': f'{avg_loss:.4f}',
                        'mse': f'{avg_mse:.4f}',
                        'cos': f'{avg_cosine:.4f}'
                    }
                )
                self._mon.pbar.update(1)
        self._mon.close_pbar()
        return {
            'total_loss': avg_loss,
            'mse_loss': avg_mse,
            'cosine_loss': avg_cosine,
        }

    def execute(self, num_epochs: int=1) -> History:
        assert self._compiled is True, (
            "You must call the method called `compile()` in first, before call the `execute()` method.")
        # Training;
        self._mon.log(f"{'=' * 120}")
        self._mon.log(f"STARTING OF JEPA TRAINING - {num_epochs} EPOCHS")
        self._mon.log(f"Start training at epoch number \"{self._start_epoch_idx}\".")
        self._mon.log(f"Start training on device \"{self._device}\".")
        self._mon.log(f"{'=' * 120}\n")
        train_curves_file = os.path.join(self._config.output_dir, self._config.train_curves_file)
        os.makedirs(self._config.output_dir, exist_ok=True)
        for epoch in range(self._start_epoch_idx, num_epochs):
            self._mon.log(f"Epoch {epoch + 1}/{num_epochs}")
            # train on one epoch;
            train_results = self.train_epoch()
            self._history.append_train(
                total_loss=train_results['total_loss'], mse_loss=train_results['mse_loss'],
                cosine_loss=train_results['cosine_loss'])
            # Validation;
            val_results = self.validate()
            self._history.append_val(
                total_loss=val_results['total_loss'], mse_loss=val_results['mse_loss'],
                cosine_loss=val_results['cosine_loss'])
            # Update model EMA;
            self.model.update_target_encoder()
            # Scheduler;
            self.scheduler.step()
            self._mon.log(f"Train Loss: {train_results['total_loss']:.4f} | Val Loss: {val_results['total_loss']:.4f}")
            self._mon.log(f"Train MSE: {train_results['mse_loss']:.4f} | Val MSE: {val_results['mse_loss']:.4f}")
            self._mon.log(
                f"Train Cosine: {train_results['cosine_loss']:.4f} | Val Cosine: {val_results['cosine_loss']:.4f}")
            # Save the best model weights;
            if val_results['total_loss'] < self._best_val_loss:
                self._best_val_loss = val_results['total_loss']
                curr_best_model_dir = os.path.join(self._config.output_dir, f"{self._config.best_model_dir}_{epoch:0d}")
                jepa_repos.save_config(self._config.model, dir_path=curr_best_model_dir)
                jepa_repos.save_data(self.model, dir_path=curr_best_model_dir, device_type=self._device.type)
                self._mon.log("✓ Best model saved!")
            # Make a checkpoint;
            if self._checkpoint_manager is not None:
                self._checkpoint_manager.save_config(epoch, self._config)
                self._checkpoint_manager.save_data(epoch, self, device_type=self._device.type)
            self._history.plot(train_curves_file)
            self._mon.log("Training curves is plotted at \"" + train_curves_file + "\".")
        return self._history
