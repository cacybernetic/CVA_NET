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
# JEPA model:
from yolo.v3.model import YOLO, Config as ModelConfig
from yolo.v3.factory import yolo3
from yolo.v3 import repository as yolo_repos, summary
# Others:
from .dataset import build as build_dataset
from .loss_fn import YOLOLoss
from .monitor import Monitor
from .history import History
from .checkpoint import CheckpointManager
from .metrics import *
from .layer_freezing import apply as apply_layer_freezing


@dataclass
class Config:
    seed: int = 42
    train_dataset: str = 'datasets/train'
    val_dataset: str = 'datasets/val'
    batch_size: int = 2
    image_size: int = 416
    gradient_accumulations: int = 48
    num_workers: int = 2
    amp: bool = False
    device: str = 'cuda'
    output_dir: str = 'runs'
    checkpoint_dir: str = 'yolo-ckpts'
    max_ckpt_to_keep: int = 3
    best_model_dir: str = 'best'
    train_curves_file: str = 'training_curves.jpeg'
    freeze_layers: List[str] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig =  field(default_factory=OptimizerConfig)
    scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


def _forward_pass_step(
    x: torch.Tensor,
    y: torch.Tensor,
    scaled_anchors: torch.Tensor,
    model: YOLO,
    criterion: Callable,
    device: torch.device,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    x = x.to(device)
    y0 = y[0].to(device)
    y1 = y[1].to(device)
    y2 = y[2].to(device)
    outputs = model.forward(x)  # noqa
    losses0 = criterion(outputs[0], y0, scaled_anchors[0])
    losses1 = criterion(outputs[1], y1, scaled_anchors[1])
    losses2 = criterion(outputs[2], y2, scaled_anchors[2])
    losses = {
        'box_loss': losses0['box_loss'] + losses1['box_loss'] + losses2['box_loss'],
        'pobj_loss': losses0['pobj_loss'] + losses1['pobj_loss'] + losses2['pobj_loss'],
        'noobj_loss': losses0['noobj_loss'] + losses1['noobj_loss'] + losses2['noobj_loss'],
        'cls_loss': losses0['cls_loss'] + losses1['cls_loss'] + losses2['cls_loss'],
    }
    total_loss = losses['box_loss'] + losses['pobj_loss'] + losses['noobj_loss'] + losses['cls_loss']
    results = {'total_loss': total_loss, **losses}
    return outputs, results


def _forward_pass_step_with_autocast(
    x: torch.Tensor,
    y: torch.Tensor,
    scaled_anchors: torch.Tensor,
    model: YOLO,
    criterion: Callable,
    device: torch.device,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    with autocast(device_type=device.type, dtype=torch.float16):
        outputs, results = _forward_pass_step(x, y, scaled_anchors, model, criterion, device)
    return outputs, results


def _train_step(
    x: torch.Tensor,
    y: torch.Tensor,
    scaled_anchors: torch.Tensor,
    model: YOLO,
    criterion: YOLOLoss,
    optimizer: Optimizer,
    num_accumulated: int,
    num_accumulations: int,
    gradient_accumulations: int,
    avg_box_loss: float,
    avg_pobj_loss: float,
    avg_noobj_loss: float,
    avg_cls_loss: float,
    mon: Monitor,
    scaler: GradScaler=None,
    device: torch.device=None,
    optimize: bool=False,
) -> Dict[str, Any]:
    # Forward pass;
    outputs, results = _forward_pass_step(x, y, scaled_anchors, model, criterion, device)
    loss = results['total_loss']
    loss_value = loss.item()
    loss = loss / num_accumulations
    loss.backward()
    num_accumulated += outputs[0].shape[0]
    if num_accumulated >= gradient_accumulations or optimize:
        ### Optimizer step;
        optimizer.step()
        ### Reset gradient;
        optimizer.zero_grad()
        mon.print(
            "  Box loss %7.4f - PObj loss %7.4f - NoObj loss %7.4f - Class loss %7.4f"
            % (avg_box_loss, avg_pobj_loss, avg_noobj_loss, avg_cls_loss))
        num_accumulated = 0
    return {
        'num_accumulated': num_accumulated,
        "total_loss": loss_value,
        'box_loss': results['box_loss'].item(),
        'pobj_loss': results['pobj_loss'].item(),
        'noobj_loss': results['noobj_loss'].item(),
        'cls_loss': results['cls_loss'].item()
        }


def _train_step_with_scaler(
    x: torch.Tensor,
    y: torch.Tensor,
    scaled_anchors: torch.Tensor,
    model: YOLO,
    criterion: YOLOLoss,
    optimizer: Optimizer,
    num_accumulated: int,
    num_accumulations: int,
    gradient_accumulations: int,
    avg_box_loss: float,
    avg_pobj_loss: float,
    avg_noobj_loss: float,
    avg_cls_loss: float,
    mon: Monitor,
    scaler: GradScaler=None,
    device: torch.device=None,
    optimize: bool=False,
) -> Dict[str, Any]:
    # Forward pass;
    with autocast(device_type=device.type, dtype=torch.float16):
        outputs, results = _forward_pass_step(x, y, scaled_anchors, model, criterion, device)
        loss = results['total_loss']
        loss_value = loss.item()
        loss = loss / num_accumulations
    scaler.scale(loss).backward()
    num_accumulated += outputs[0].shape[0]
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
            "  Box loss %7.4f - PObj loss %7.4f - NoObj loss %7.4f - Class loss %7.4f"
            % (avg_box_loss, avg_pobj_loss, avg_noobj_loss, avg_cls_loss))
        num_accumulated = 0
    return {
        'num_accumulated': num_accumulated,
        "total_loss": loss_value,
        'box_loss': results['box_loss'].item(),
        'pobj_loss': results['box_loss'].item(),
        'noobj_loss': results['noobj_loss'].item(),
        'cls_loss': results['cls_loss'].item()
        }


class YOLOTrainer:

    def __init__(self, config: Config) -> None:
        """
        Method allows to create an instance of JEPA trainer.
        """
        self._config = config
        self.model: YOLO = None
        self._criterion: YOLOLoss = None
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
        self._best_cls_acc = -float('inf')
        self._best_pobj_acc = -float('inf')
        self._best_noobj_acc = -float('inf')
        self._best_epoch = -1
        self._start_epoch_idx = 0
        self._num_accumulations = 1
        self._train_step = None
        self._forward_pass_step = None
        self._scaled_anchors = None
        self._class_names = None
        self._compiled = False
        self._checkpoint_loaded = False
        # Create metrics;
        self._precision = None
        self._recall = None
        self._map50 = None
        self._map50_95 = None

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
            # Also set the deterministic flag for reproducibility
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
        anchors = self._config.model.anchors
        S = self._config.model.S
        img_channels = self._config.model.img_channels
        dataloaders = build_dataset(
            anchors=anchors, train_data_dir=self._config.train_dataset, val_data_dir=self._config.val_dataset,
            img_size=self._config.image_size, img_channels=img_channels, s=S, batch_size=self._config.batch_size,
            num_workers=self._config.num_workers, pin_memory=use_pin_memory)
        self._train_dataset_loader = dataloaders[0]
        self._val_dataset_loader = dataloaders[1]
        self._mon.log("Training dataset:")
        self._mon.log(f"  Number of batchs -> {len(self._train_dataset_loader)}")
        self._mon.log("Validation dataset:")
        self._mon.log(f"  Number of batchs -> {len(self._val_dataset_loader)}")

    def _instanciate_model(self) -> None:
        if self.model is None:
            if self._config.model is None:
                self._config.model = ModelConfig()
            self.model, _ = yolo3(self._config.model)
            self.model = self.model.to(self._device)

    def compile(self) -> None:
        assert self._config.train_dataset, "The directory path of training dataset is not provided."
        assert self._config.val_dataset, "The directory path of validation dataset is not provided."
        assert self.model is None or self._config.model is not None, "The model config is not specified."
        assert self.optimizer is None or self._config.optimizer is not None, "The optimizer config is not specified."
        assert self.scheduler is None or self._config.scheduler is not None, "The scheduler config is not specified."
        ## Class names;
        self._class_names = self._config.model.class_names
        # Logging of training config;
        self._mon.log("=" * 120)
        self._mon.log("TRAINING CONFIG")
        self._mon.log("=" * 120)
        self._mon.log(str(self._config))
        # Device setting;
        self._device_setting()
        # Setting of seed value for random generators;
        self.set_seed(self._config.seed, self._device)
        # Create dataloaders;
        self._create_dataloaders()
        # Instanciation of the model;
        self._instanciate_model()
        ## Prepare the scaled anchors;
        anchors = self._config.model.anchors
        S = self._config.model.S
        S = torch.tensor(S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
        self._scaled_anchors = (torch.tensor(anchors) * S).to(self._device)
        ## Instanciate criterion function;
        self._criterion = YOLOLoss()
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
        ## Model layer editions applying;
        if self._config.freeze_layers:
            self.model = apply_layer_freezing(self.model, self._config.freeze_layers)
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
        # Create metrics;
        num_classes = len(self._class_names)
        self._precision = Precision(num_classes=num_classes, iou_threshold=0.5)
        self._recall = Recall(num_classes=num_classes, iou_threshold=0.5)
        self._map50 = mAP50(num_classes=num_classes)
        self._map50_95 = mAP50_95(num_classes=num_classes)
        ## Print model summary;
        model_stat, inference_time = summary.build(self.model, batchs=self._config.batch_size, device=self._device)
        self._mon.log("=" * 120)
        self._mon.log("MODEL SUMMARY")
        self._mon.log('=' * 120)
        self._mon.log(f"\n{model_stat}")
        self._mon.log(f"Inference times: {inference_time:.3f} seconds.")
        # Specify that all is ready;
        self._compiled = True

    def state_dict(self) -> Dict[str, Any]:
        return {
            'best_val_loss': self._best_val_loss,
            'best_cls_acc': self._best_cls_acc,
            'best_pobj_acc': self._best_pobj_acc,
            'best_noobj_acc': self._best_noobj_acc,
            'best_epoch': self._best_epoch,
            'history': self._history.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Method of training checkpoint loading.
        """
        self._best_val_loss = state_dict['best_val_loss']
        self._best_cls_acc = state_dict['best_cls_acc']
        self._best_pobj_acc = state_dict['best_pobj_acc']
        self._best_noobj_acc = state_dict['best_noobj_acc']
        self._best_epoch = state_dict['best_epoch']
        self._history.load_state_dict(state_dict['history'])

    def train_epoch(self) -> Dict[str, Any]:
        """
        Training function on one epoch.
        """
        self.model.train()
        # Total loss
        total_loss = 0
        box_loss = 0
        pobj_loss = 0
        noobj_loss = 0
        cls_loss = 0
        # Avg loss
        avg_total_loss = 0
        avg_box_loss = 0
        avg_pobj_loss = 0
        avg_noobj_loss = 0
        avg_cls_loss = 0
        num_accumulated = 0
        total_batchs = len(self._train_dataset_loader)
        self._mon.create_pbar(total_batchs, desc="Training")
        self.optimizer.zero_grad()
        for num_batchs, batch_data in enumerate(self._train_dataset_loader, 1):
            x, y = batch_data
            ## Forward pass;
            # optimize=(num_batchs>=total_batchs): if it is the last batchs then we compute optimization.
            results = self._train_step(
                x=x, y=y, scaled_anchors=self._scaled_anchors, model=self.model, criterion=self._criterion,
                optimizer=self.optimizer, num_accumulated=num_accumulated,
                num_accumulations=self._num_accumulations, gradient_accumulations=self._config.gradient_accumulations,
                avg_box_loss=avg_box_loss, avg_pobj_loss=avg_pobj_loss, avg_noobj_loss=avg_noobj_loss,
                avg_cls_loss=avg_cls_loss, mon=self._mon, scaler=self._scaler, device=self._device,
                optimize=(num_batchs>=total_batchs))
            num_accumulated = results['num_accumulated']
            ## Statistiques;
            total_loss += results['total_loss']
            box_loss += results['box_loss']
            pobj_loss += results['pobj_loss']
            noobj_loss += results['noobj_loss']
            cls_loss += results['cls_loss']
            ## Calculate average;
            avg_total_loss = total_loss / num_batchs
            avg_box_loss = box_loss / num_batchs
            avg_pobj_loss = pobj_loss / num_batchs
            avg_noobj_loss = noobj_loss / num_batchs
            avg_cls_loss = cls_loss / num_batchs
            self._mon.pbar.set_postfix(
                {
                    'box_loss': f'{avg_box_loss:7.4f}',
                    'pobj_loss': f'{avg_pobj_loss:7.4f}',
                    'noobj_loss': f'{avg_noobj_loss:7.4f}',
                    'cls_loss': f'{avg_cls_loss:7.4f}'
                }
            )
            self._mon.pbar.set_description(f"Training [total_loss={avg_total_loss:7.4f}]")
            self._mon.pbar.update(1)
        self._mon.close_pbar()
        return {
            'total_loss': avg_total_loss,
            'box_loss': avg_box_loss,
            'pobj_loss': avg_pobj_loss,
            'noobj_loss': avg_noobj_loss,
            'cls_loss': avg_cls_loss,
        }

    def validate(self) -> Dict[str, Any]:
        """
        Validation function on one epoch.
        """
        self.model.eval()
        # Total loss
        total_loss = 0
        box_loss = 0
        pobj_loss = 0
        noobj_loss = 0
        cls_loss = 0
        cls_acc = 0
        pobj_acc = 0
        noobj_acc = 0
        precision = 0
        recall = 0
        map50 = 0
        map50_95 = 0
        # Avg loss
        avg_total_loss = 0
        avg_box_loss = 0
        avg_pobj_loss = 0
        avg_noobj_loss = 0
        avg_cls_acc = 0
        avg_pobj_acc = 0
        avg_noobj_acc = 0
        # Predictions and targets;
        # predictions = []
        # targets = []
        threshold = 0.05
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0
        self._mon.create_pbar(len(self._val_dataset_loader), desc="Validation")
        with torch.no_grad():
            for num_batchs, batch_data in enumerate(self._val_dataset_loader, 1):
                x, y = batch_data
                # Forward pass;
                outputs, results = self._forward_pass_step(
                    x, y, self._scaled_anchors, self.model, self._criterion, self._device)
                # Statistiques;
                total_loss += results['total_loss'].item()
                box_loss += results['box_loss'].item()
                pobj_loss += results['pobj_loss'].item()
                noobj_loss += results['noobj_loss'].item()
                cls_loss += results['cls_loss'].item()
                ## Compute predictions;
                for i in range(3):
                    y[i] = y[i].to(self._device)
                    obj = y[i][..., 0] == 1 # in paper this is Iobj_i
                    noobj = y[i][..., 0] == 0  # in paper this is Iobj_i
                    # Compute acc;
                    out = outputs[i].detach()
                    correct_class += torch.sum(torch.argmax(out[..., 5:][obj], dim=-1) == y[i][..., 5][obj])
                    tot_class_preds += torch.sum(obj)
                    # ===================================
                    obj_preds = torch.sigmoid(out[..., 0]) > threshold
                    correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
                    tot_obj += torch.sum(obj)
                    correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
                    tot_noobj += torch.sum(noobj)
                    ## Finalyse;
                    avg_cls_acc = (correct_class/(tot_class_preds+1e-16)).item()
                    avg_pobj_acc = (correct_obj/(tot_obj+1e-16)).item()
                    avg_noobj_acc = (correct_noobj/(tot_noobj+1e-16)).item()
                ## Calculate average;
                avg_total_loss = total_loss / num_batchs
                avg_box_loss = box_loss / num_batchs
                avg_pobj_loss = pobj_loss / num_batchs
                avg_noobj_loss = noobj_loss / num_batchs
                avg_cls_loss = cls_loss / num_batchs
                # avg_cls_acc = cls_acc / num_batchs
                # avg_pobj_acc = pobj_acc / num_batchs
                # avg_noobj_acc = noobj_acc / num_batchs

                # outputs = outputs.detach().cpu()
                # anchors = self._scaled_anchors.cpu()
                # preds = _compute_predictions(outputs, anchors)
                # print(preds)
                # exit(0)
                # predictions.append(preds)
                # targets.append(target)
                ## Show results;
                self._mon.pbar.set_postfix(
                    {
                        'box_loss': f'{avg_box_loss:7.4f}',
                        'pobj_loss': f'{avg_pobj_loss:7.4f}',
                        'noobj_loss': f'{avg_noobj_loss:7.4f}',
                        'cls_loss': f'{avg_cls_loss:7.4f}',
                        'cls_acc': f'{avg_cls_acc:7.4f}',
                        'pobj_acc': f'{avg_pobj_acc:7.4f}',
                        'noobj_acc': f'{avg_noobj_acc:7.4f}',
                    }
                )
                self._mon.pbar.set_description(f"Validation [total_loss={avg_total_loss:7.4f}]")
                self._mon.pbar.update(1)
        self._mon.close_pbar()
        ## Compute metrics;
        # precision = self._precision(predictions, targets).compute()
        # recall = self._recall(predictions, targets)
        # mean_precision = self._precision.compute()
        # mean_recall = self._recall.compute()
        # map50 = self._map50(predictions, targets)
        # map50_95 = self._map50_95(predictions, targets)
        return {
            'total_loss': avg_total_loss,
            'box_loss': avg_box_loss,
            'pobj_loss': avg_pobj_loss,
            'noobj_loss': avg_noobj_loss,
            'cls_loss': avg_cls_loss,
            'cls_acc': avg_cls_acc,
            'pobj_acc': avg_pobj_acc,
            'noobj_acc': avg_noobj_acc,
            # 'mean_precision': mean_precision,
            # 'mean_recall': mean_recall,
            # 'map50': map50,
            # 'map50_95': map50_95,

        }

    def execute(self, num_epochs: int=1) -> History:
        assert self._compiled is True, (
            "You must call the method called `compile()` in first, before call the `execute()` method.")
        # Training;
        self._mon.log(f"{'=' * 120}")
        self._mon.log(f"STARTING OF YOLOv3 TRAINING - {num_epochs} EPOCHS")
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
                box_loss=train_results['box_loss'], pobj_loss=train_results['pobj_loss'],
                noobj_loss=train_results['noobj_loss'], cls_loss=train_results['cls_loss'])
            ## Validation;
            val_results = self.validate()
            self._history.append_val(
                box_loss=val_results['box_loss'], pobj_loss=val_results['pobj_loss'],
                noobj_loss=val_results['noobj_loss'], cls_loss=val_results['cls_loss'],
                cls_acc=val_results['cls_acc'], pobj_acc=val_results['pobj_acc'], noobj_acc=val_results['noobj_acc'])
            ## Scheduler;
            self.scheduler.step()
            self._mon.log(f"TRAIN | VAL")
            self._mon.log(f"Total Loss: {train_results['total_loss']:.4f} | {val_results['total_loss']:.4f}")
            self._mon.log(f"Box Loss:   {train_results['box_loss']:.4f}   | {val_results['box_loss']:.4f}")
            self._mon.log(f"Pobj Loss:  {train_results['pobj_loss']:.4f}  | {val_results['pobj_loss']:.4f}")
            self._mon.log(f"Noobj Loss: {train_results['noobj_loss']:.4f} | {val_results['noobj_loss']:.4f}")
            self._mon.log(f"Class Loss: {train_results['cls_loss']:.4f}   | {val_results['cls_loss']:.4f}")
            self._mon.log(("-" * 10) + "Accuracies" + ("-" * 10))
            self._mon.log(f"Class Acc: {val_results['cls_acc']*100:2f}%")
            self._mon.log(f"Pobj Acc: {val_results['pobj_acc']*100:2f}%")
            self._mon.log(f"Noobj Acc: {val_results['noobj_acc']*100:2f}%")
            ## Save the best model weights;
            if (
                val_results['total_loss'] <= self._best_val_loss
                and val_results['cls_acc'] >= self._best_cls_acc
                and val_results['pobj_acc'] >= self._best_pobj_acc
                and val_results['noobj_acc'] >= self._best_noobj_acc
            ):
                if (
                    val_results['total_loss'] == self._best_val_loss
                    and val_results['cls_acc'] == self._best_cls_acc
                    and val_results['pobj_acc'] == self._best_pobj_acc
                    and val_results['noobj_acc'] == self._best_noobj_acc
                ):
                    self._mon.log("Case where this model acc is equal to the last best model acc.")
                    continue
                self._best_val_loss = val_results['total_loss']
                self._best_cls_acc = val_results['cls_acc']
                self._best_pobj_acc = val_results['pobj_acc']
                self._best_noobj_acc = val_results['noobj_acc']
                curr_best_model_dir = os.path.join(self._config.output_dir, f"{self._config.best_model_dir}_{epoch:0d}")
                yolo_repos.save_config(self._config.model, dir_path=curr_best_model_dir)
                yolo_repos.save_data(self.model, dir_path=curr_best_model_dir, device_type=self._device.type)
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
        return self._history
