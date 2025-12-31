from typing import Union, Dict, Any
from dataclasses import dataclass
import torch
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from cva_net.alexnet.jepa.model import JEPA
from cva_net.alexnet.jepa import repository as jepa_repos
from .loss_fn import compute_loss
from .monitor import Monitor
from .history import History


@dataclass
class Config:
    batch_size: int = 32
    gradient_accumulation: int = 1
    num_workers: int = 2
    amp: bool = True
    device: str = 'cpu'
    output_dir: str = 'alexnet-jepa'
    checkpoint_dir: str = 'jepa-ckpts'
    best_model_dir: str = 'best'


class JEPATrainer:

    def __init__(
        self,
        model: JEPA ,
        train_dataset: Dataset,
        val_dataset: Dataset,
        optimizer: Optimizer=None,
        scheduler: LRScheduler=None,
        batch_size: int=32,
        gradient_accumulatation: int=1,
        lr0: float=1e-4,
        weight_decay: float=0.005,
        scheduler_max_iter: int=10,
        num_workers: int=2,
        amp: bool=True,
        device: Union[str, torch.device]='cpu',
        checkpoint_dir: str='jepa-ckpts/',
        output_dir: str='alexnet-jepa/',
    ) -> None:
        """
        Method allows to create an instance of JEPA trainer.
        """
        super().__init__()
        self._model = model
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._batch_size = batch_size
        self._gradient_accumulation = gradient_accumulatation
        self._lr0 = lr0
        self._weight_decay = weight_decay
        self._scheduler_max_iter = scheduler_max_iter
        self._num_workers = num_workers
        self._amp = amp
        self._device = device
        self._checkpoint_dir = checkpoint_dir
        self._output_dir = output_dir
        self._mon: Monitor = None
        self._train_dataset_loader: DataLoader = None
        self._val_dataset_loader: DataLoader = None
        self._history: History = None

    def compile(self) -> None:
        self._mon = Monitor()
        # Create dataloaders;
        assert isinstance(self._device, (str, torch.device))
        use_pin_memory = False
        if isinstance(self._device, str) and self._device.startswith('cuda'):
            use_pin_memory = True
        elif self._device.type == 'gpu':
            use_pin_memory = True
        self._train_dataset_loader = DataLoader(
            self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers,
            pin_memory=use_pin_memory)
        self._val_dataset_loader = DataLoader(
            self._val_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers,
            pin_memory=use_pin_memory)
        # Model moving into selected device;
        self._model = self._model.to(self._device)
        # Instanciation of optimizer model;
        if self._optimizer is None:
            self._optimizer = AdamW(
                params=list(self._model.context_encoder.parameters()) + list(self._model.predictor.parameters()),
                lr=self._lr0, weight_decay=self._weight_decay)
        # Instanciation of scheduler model;
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self._scheduler_max_iter)
        # Create empty history;
        self._history = History()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass

    def train_epoch(self):
        """Training function on one epoch."""
        self._model.train()
        total_loss = 0
        total_mse = 0
        total_cosine = 0
        avg_loss = 0
        avg_mse = 0
        avg_cosine = 0
        num_accumulation = 0
        self._mon.create_pbar(len(self._train_dataset_loader), desc="Training")
        self._optimizer.zero_grad()
        for num_batchs, batch_data in enumerate(self._train_dataset_loader, 1):
            view1, view2 = batch_data
            view1, view2 = view1.to(self._device), view2.to(self._device)
            # Forward pass;
            predicted, target, _ = self._model(view1, view2)
            loss, mse, cosine = compute_loss(predicted, target)
            loss.backward()
            num_accumulation += predicted.shape[0]
            if num_accumulation >= self._gradient_accumulation:
                # Gradient descent;
                self._optimizer.step()
                # Update EMA;
                self._model.update_target_encoder()
                # Reset gradient;
                self._optimizer.zero_grad()
                self._mon.print(
                    " * Total loss %7.4f - MSE loss %7.4f - Cosine loss %7.4f" % (avg_loss, avg_mse, avg_cosine))
                num_accumulation = 0
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

    def validate(self):
        """Training function on one epoch."""
        self._model.eval()
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
                predicted, target, _ = self._model(view1, view2)
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
        # Training;
        self._mon.log(f"\n{'=' * 120}")
        self._mon.log(f"DÉBUT DE L'ENTRAÎNEMENT JEPA - {num_epochs} ÉPOQUES")
        self._mon.log(f"{'=' * 120}\n")
        best_val_loss = float('inf')
        start_epoch = self._history.count
        for epoch in range(start_epoch, num_epochs):
            self._mon.log(f"\nEpoch {epoch + 1}/{num_epochs}")
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
            # Scheduler
            self._scheduler.step()
            self._mon.log(f"Train Loss: {train_results['total_loss']:.4f} | Val Loss: {val_results['total_loss']:.4f}")
            self._mon.log(f"Train MSE: {train_results['mse_loss']:.4f} | Val MSE: {val_results['mse_loss']:.4f}")
            self._mon.log(
                f"Train Cosine: {train_results['cosine_loss']:.4f} | Val Cosine: {val_results['cosine_loss']:.4f}")
            # Save the best model weights;
            if val_results['total_loss'] < best_val_loss:
                best_val_loss = val_results['total_loss']
                jepa_repos.save(self._model, 'best_jepa_model.pth')
                self._mon.log("✓ Meilleur modèle sauvegardé!")
        # Évaluation finale avec Linear Probing
        self._mon.log("\n" + "=" * 120)
        self._mon.log("ÉVALUATION FINALE - Linear Probing")
        self._mon.log("=" * 120)
        return self._history
