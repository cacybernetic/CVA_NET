import os
from typing import  Dict, Any, Optional
from dataclasses import dataclass, field
import torch
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
    train_dataset: str = 'datasets/train'
    val_dataset: str = 'datasets/val'
    batch_size: int = 32
    image_size: int = 224
    gradient_accumulation: int = 128
    num_workers: int = 2
    amp: bool = True
    device: str = 'cpu'
    output_dir: str = 'alexnet-jepa'
    checkpoint_dir: str = 'jepa-ckpts'
    max_ckpt_to_keep: int = 5
    best_model_dir: str = 'best'
    model: JEPAConfig = field(default_factory=JEPAConfig)
    optimizer: OptimizerConfig =  field(default_factory=OptimizerConfig)
    scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


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
        self._best_val_loss = float('inf')
        self._start_epoch_idx = 0
        self._num_accumulations = 1
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

    def compile(self) -> None:
        assert self._config.train_dataset, "The directory path of training dataset is not provided."
        assert self._config.val_dataset, "The directory path of validation dataset is not provided."
        assert self.model is None or self._config.model is not None, "The model config is not specified."
        assert self.optimizer is None or self._config.optimizer is not None, "The optimizer config is not specified."
        assert self.scheduler is None or self._config.scheduler is not None, "The scheduler config is not specified."
        # Device setting;
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
        # Create dataloaders;
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
        # Instanciation of the model;
        if self.model is None:
            if self._config.model is None:
                self._config.model = JEPAConfig()
            self.model, _ = jepa(self._config.model)
            self.model = self.model.to(self._device)
        model_stat = summary.build(self.model, device=self._device)
        self._mon.log("=" * 120)
        self._mon.log("MODEL SUMMARY")
        self._mon.log(f"{"=" * 120}")
        self._mon.log(f"\n{model_stat}")
        # Instanciation of optimizer model;
        if self.optimizer is None:
            if self._config.optimizer is None:
                self._config.optimizer = OptimizerConfig()
            self.optimizer, _ = build_optimizer(self.model, self._config.optimizer)
        self._mon.log("Optimizer:")
        self._mon.log("  Config: " + repr(self.optimizer))
        # Instanciation of scheduler model;
        if self.scheduler is None:
            if self._config.scheduler is None:
                self._config.scheduler = LRSchedulerConfig()
            self.scheduler, _ = lr_scheduler(self.optimizer, self._config.scheduler)
        self._mon.log("Scheduler:")
        self._mon.log("  Config: " + repr(self.scheduler))
        if self._checkpoint_loaded:
            self._checkpoint_manager.load_data(self._start_epoch_idx, self._config, self)
            self._start_epoch_idx += 1
        # Calculate number of accumulations;
        if self._config.gradient_accumulation > self._config.batch_size:
            self._num_accumulations = self._config.gradient_accumulation // self._config.batch_size
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
        num_accumulation = 0
        self._mon.create_pbar(len(self._train_dataset_loader), desc="Training")
        self.optimizer.zero_grad()
        for num_batchs, batch_data in enumerate(self._train_dataset_loader, 1):
            view1, view2 = batch_data
            view1, view2 = view1.to(self._device), view2.to(self._device)
            # Forward pass;
            predicted, target, _ = self.model(view1, view2)  # noqa
            loss, mse, cosine = compute_loss(predicted, target)
            loss_value = loss.item()
            loss = loss / self._num_accumulations
            loss.backward()
            num_accumulation += predicted.shape[0]
            if num_accumulation >= self._config.gradient_accumulation:
                # Gradient descent;
                self.optimizer.step()
                # Update EMA;
                self.model.update_target_encoder()
                # Reset gradient;
                self.optimizer.zero_grad()
                self._mon.print(
                    " * Total loss %7.4f - MSE loss %7.4f - Cosine loss %7.4f" % (avg_loss, avg_mse, avg_cosine))
                num_accumulation = 0
            # Statistiques;
            total_loss += loss_value
            total_mse += mse.item()
            total_cosine += cosine.item()
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
                predicted, target, _ = self.model(view1, view2)  # noqa
                loss, mse, cosine = compute_loss(predicted, target)
                # Statistiques;
                total_loss += loss.item()
                total_mse += mse.item()
                total_cosine += cosine.item()
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
        return self._history
