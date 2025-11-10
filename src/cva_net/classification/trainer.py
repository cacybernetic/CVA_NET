import logging
import typing as t

from typing_extensions import Self
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)
EXECUTION_RESULTS = t.Tuple[
    t.Dict[str, t.List[float]], t.Optional[t.Dict[str, t.List[float]]]
]


def _accuracy_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    r"""
    Compute accuracy score using only PyTorch tensors.

    :param y_pred: torch.Tensor of predicted labels.
    :param y_true: torch.Tensor of true labels.
    :returns: A torch.Tensor containing the accuracy score.
    """
    # Ensure both tensors are on the same device and have the same shape.
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Shapes of y_true %s and y_pred %s must match."
            % (str(y_true.shape), str(y_pred.shape))
        )

    # Calculate number of correct predictions:
    correct = torch.eq(y_true, y_pred).sum()
    # Calculate total number of predictions:
    total = torch.tensor(
        y_true.shape[0], dtype=torch.float32, device=y_true.device
    )
    # Compute accuracy
    accuracy = correct.float() / total
    return accuracy


def _precision_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
        average: t.Literal[
        'binary', 'micro', 'macro', 'weighted', 'none'
    ]='weighted',
    pos_label: int=1,
    zero_division: float=0.0
) -> torch.Tensor:
    r"""
    Compute precision score using only PyTorch tensors.

    :param y_true: torch.Tensor or array-like True labels.
    :param y_pred: torch.Tensor or array-like Predicted labels.
    :param average: str, default='binary'
      One of ['binary', 'micro', 'macro', 'weighted', 'none'].
      - 'binary': Only report results for the class specified by pos_label.
      - 'micro': Calculate metrics globally by counting total TP and FP.
      - 'macro': Calculate metrics for each label, return unweighted mean.
      - 'weighted': Calculate metrics for each label, return weighted mean by support.
      - 'none': Return precision for each class.
    :param pos_label : int, default=1
      The label of the positive class (for binary classification).
    :param zero_division : float, default=0.0
      Value to return when there is a zero division.

    :returns: Precision score(s).
    """
    # Convert to tensors if needed
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    
    # Ensure both tensors have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} must match")

    # Get unique classes
    classes = torch.unique(torch.cat([y_true, y_pred]))
    classes = classes.sort().values

    # Store original device
    device = y_true.device
    y_true_cpu = y_true.cpu()
    y_pred_cpu = y_pred.cpu()

    if average == 'binary':
        if len(classes) > 2:
            raise ValueError(
                "Target is multiclass but average='binary'. "
                "Please choose another average setting."
            )

        # Binary precision calculation.
        true_positive = ((y_true_cpu == pos_label) & (y_pred_cpu == pos_label)).sum().float()
        false_positive = ((y_true_cpu != pos_label) & (y_pred_cpu == pos_label)).sum().float()

        denominator = true_positive + false_positive
        if denominator == 0:
            precision = torch.tensor(
                zero_division, dtype=torch.float32, device=device
            )
        else:
            precision = true_positive / denominator
            precision = precision.to(device)
        return precision

    elif average in ['micro', 'macro', 'weighted', 'none']:
        # Multi-class precision calculation.
        precisions = []
        supports = []

        for cls in classes:
            # For each class, treat it as positive and others as negative.
            true_positive = ((y_true_cpu == cls) & (y_pred_cpu == cls)).sum().float()
            false_positive = ((y_true_cpu != cls) & (y_pred_cpu == cls)).sum().float()

            denominator = true_positive + false_positive
            if denominator == 0:
                precision_cls = torch.tensor(zero_division, dtype=torch.float32)
            else:
                precision_cls = true_positive / denominator

            precisions.append(precision_cls)
            supports.append((y_true_cpu == cls).sum().float())

        precisions = torch.stack(precisions)
        supports = torch.stack(supports)

        if average == 'micro':
            # Micro-precision: global TP / (TP + FP).
            total_true_positive = sum([
                ((y_true_cpu == cls) & (y_pred_cpu == cls)).sum().float() 
                for cls in classes
            ])
            total_false_positive = sum([
                ((y_true_cpu != cls) & (y_pred_cpu == cls)).sum().float() 
                for cls in classes
            ])

            denominator = total_true_positive + total_false_positive
            if denominator == 0:
                result = torch.tensor(zero_division, dtype=torch.float32)
            else:
                result = total_true_positive / denominator

        elif average == 'macro':
            # Simple average of per-class precisions.
            result = precisions.mean()

        elif average == 'weighted':
            # Weighted average by support.
            if supports.sum() == 0:
                result = torch.tensor(zero_division, dtype=torch.float32)
            else:
                result = (precisions * supports).sum() / supports.sum()

        elif average == 'none':
            # Return precision for each class.
            result = precisions.to(device)
            return result

        result = result.to(device)
        return result

    else:
        raise ValueError(
            "Average should be one of "
            "['binary', 'micro', 'macro', 'weighted', 'none']."
        )


def _recall_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    average: t.Literal[
        'binary', 'micro', 'macro', 'weighted', 'none'
    ]='weighted',
    pos_label: int=1,
    zero_division: float=0.0
) -> torch.Tensor:
    r"""
    Compute recall score using only PyTorch tensors.

    :param y_true: torch.Tensor or array-like True labels.
    :param y_pred: torch.Tensor or array-like Predicted labels.
    :param average: str, default='binary'
      One of ['binary', 'micro', 'macro', 'weighted', 'none'].
      - 'binary': Only report results for the class specified by pos_label.
      - 'micro': Calculate metrics globally by counting total TP and FN.
      - 'macro': Calculate metrics for each label, return unweighted mean.
      - 'weighted': Calculate metrics for each label, return weighted mean
        by support.
      - 'none': Return recall for each class.
    :param pos_label: int, default=1.
      The label of the positive class (for binary classification).
    :param zero_division: float, default=0.0
      Value to return when there is a zero division.

    returns: torch.Tensor of float of Recall score(s).
    """
    # Convert to tensors if needed
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    
    # Ensure both tensors have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} must match")

    # Get unique classes
    classes = torch.unique(torch.cat([y_true, y_pred]))
    classes = classes.sort().values

    # Store original device
    device = y_true.device
    y_true_cpu = y_true.cpu()
    y_pred_cpu = y_pred.cpu()

    if average == 'binary':
        if len(classes) > 2:
            raise ValueError(
                "Target is multiclass but average='binary'. "
                "Please choose another average setting."
            )

        # Binary recall calculation: TP / (TP + FN)
        true_positive = ((y_true_cpu == pos_label) & (y_pred_cpu == pos_label)).sum().float()
        false_negative = ((y_true_cpu == pos_label) & (y_pred_cpu != pos_label)).sum().float()

        denominator = true_positive + false_negative
        if denominator == 0:
            recall = torch.tensor(zero_division, dtype=torch.float32, device=device)
        else:
            recall = true_positive / denominator
            recall = recall.to(device)

        return recall

    elif average in ['micro', 'macro', 'weighted', 'none']:
        # Multi-class recall calculation
        recalls = []
        supports = []

        for cls in classes:
            # For each class, treat it as positive and others as negative
            true_positive = ((y_true_cpu == cls) & (y_pred_cpu == cls)).sum().float()
            false_negative = ((y_true_cpu == cls) & (y_pred_cpu != cls)).sum().float()

            denominator = true_positive + false_negative
            if denominator == 0:
                recall_cls = torch.tensor(zero_division, dtype=torch.float32)
            else:
                recall_cls = true_positive / denominator

            recalls.append(recall_cls)
            supports.append((y_true_cpu == cls).sum().float())

        recalls = torch.stack(recalls)
        supports = torch.stack(supports)

        if average == 'micro':
            # Micro-recall: global TP / (TP + FN)
            total_true_positive = sum([
                ((y_true_cpu == cls) & (y_pred_cpu == cls)).sum().float() 
                for cls in classes
            ])
            total_false_negative = sum([
                ((y_true_cpu == cls) & (y_pred_cpu != cls)).sum().float() 
                for cls in classes
            ])

            denominator = total_true_positive + total_false_negative
            if denominator == 0:
                result = torch.tensor(zero_division, dtype=torch.float32)
            else:
                result = total_true_positive / denominator

        elif average == 'macro':
            # Simple average of per-class recalls
            result = recalls.mean()

        elif average == 'weighted':
            # Weighted average by support
            if supports.sum() == 0:
                result = torch.tensor(zero_division, dtype=torch.float32)
            else:
                result = (recalls * supports).sum() / supports.sum()

        elif average == 'none':
            # Return recall for each class
            result = recalls.to(device)
            return result

        return result.to(device)

    else:
        raise ValueError(
            "Average should be one of ['binary', 'micro', 'macro', "
            "'weighted', 'none']"
        )


class Result:

    def __init__(self) -> None:
        self._values = {}

    @property
    def column_names(self) -> t.List[str]:
        return list(self._values.keys())
    
    @property
    def values(self) -> t.Dict[str, t.List[float]]:
        return self._values

    def add(self, new: t.Dict[str, torch.Tensor]) -> Self:
        for name, value in new.items():
            if name not in self._values:
                self._values[name] = list()
            self._values[name].append(value.cpu().detach().item())
        return self

    def __add__(self, new: t.Dict[str, torch.Tensor]) -> Self:
        return self.add(new)

    def __radd__(self, new: t.Dict[str, torch.Tensor]) -> Self:
        return self.add(new)

    def __getitem__(
        self,
        index: t.Union[int, str],
    ) -> t.Union[t.Dict[str, float], t.List[float]]:
        if isinstance(index, str):
            return self._values[index]
        value = {}
        for name in self._values:
            value[name] = self._values[name][index]
        return value


def get_validation_dataset(dataset_instance: Dataset, p: float) -> Dataset:
    ds_dataset_len = len(dataset_instance)
    val_dataset_len = int(p * ds_dataset_len)
    indices = torch.randint(0, ds_dataset_len, (val_dataset_len,))
    features = []
    targets = []
    for index in indices:
        feature, target = dataset_instance[index]
        features.append(feature)
        targets.append(target)
    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.int64)
    features = torch.tensor(features)
    targets = torch.tensor(targets)
    val_dataset = TensorDataset(features, targets)
    return val_dataset


def output_function(
    logits: torch.Tensor
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Post-processing method
    ----------------------

    :param logits: [batch_size, num_classes];
    :returns: tuple of two tensors of size [batch_size,],
        the first tensor contains the class ids predicted
        and the second tensor contains the softmax confidences.
    """
    probs = torch.softmax(logits, dim=-1)  # [n, num_classes]
    class_ids = torch.argmax(probs, dim=-1)  # [n,]
    confidences = torch.max(probs, dim=-1).values  # [n,]
    return class_ids, confidences


class Trainer:

    def __init__(
        self,
        *,
        train_dataset: Dataset,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        test_dataset: Dataset=None,
        val_dataset: Dataset=None,
        post_processing_func: t.Callable[
            [torch.Tensor],
            t.Tuple[
                  torch.Tensor,
                  torch.Tensor
            ]
        ]=output_function,
        num_epochs: int=1,
        batch_size: int=16,
        num_workers: int=4,
        drop_last: bool=False,
        pin_memory: bool=False,
        val_prop: float=None,
        gradient_acc: int=256,
        device: t.Union[str, torch.device]=None,
    ) -> None:
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._device = device
        self._post_process = post_processing_func
        if self._device is None:
            self._device = torch.device('cuda') if torch.cuda.is_available() \
                else torch.device('cpu')

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_acc = gradient_acc
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.val_prop = val_prop
        self.num_batchs = 0
        self.num_train_batchs = -1
        self.num_val_batchs = -1
        self.num_test_batchs = -1
        self.epoch_idx = 0
        self.batch_idx = 0
        self.num_acc = 0
        self.train_result = Result()
        self.test_result = Result()
        self.step = ''

        if not self.val_prop:
            self.val_prop = 0.2

        self.train_loss = torch.tensor(0.0, device=self._device)
        self.train_accuracy_score = torch.tensor(0.0, device=self._device)
        self.train_precision_score = torch.tensor(0.0, device=self._device)
        self.train_recall_score = torch.tensor(0.0, device=self._device)
        self.train_avg_confidence = torch.tensor(0.0, device=self._device)
        self.eval_loss = torch.tensor(0.0, device=self._device)
        self.eval_accuracy_score = torch.tensor(0.0, device=self._device)
        self.eval_precision_score = torch.tensor(0.0, device=self._device)
        self.eval_recall_score = torch.tensor(0.0, device=self._device)
        self.eval_avg_confidence = torch.tensor(0.0, device=self._device)
        self.loss = torch.tensor(0.0, device=self._device)
        self.accuracy_score = torch.tensor(0.0, device=self._device)
        self.precision_score = torch.tensor(0.0, device=self._device)
        self.recall_score = torch.tensor(0.0, device=self._device)
        self.avg_confidence = torch.tensor(0.0, device=self._device)

        # self._accuracy_score = AccuracyScore()
        # self._precision_score = PrecisionScore()
        # self._recall_score = RecallScore()
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._compiled = False
        self._completed = False
    
    def completed(self) -> bool:
        return self._completed

    def compile(self) -> None:
        """This method must be called before execution method."""
        # Creation of data loader.
        self._train_loader = DataLoader(
            dataset=self._train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.num_train_batchs = len(self._train_loader)
        if self._val_dataset is None:
            if self._test_dataset is not None:
                self._val_dataset = get_validation_dataset(
                    self._test_dataset, self.val_prop
                )
                self._val_loader = DataLoader(
                    dataset=self._val_dataset, batch_size=self.batch_size,
                    shuffle=False, num_workers=self.num_workers,
                    pin_memory=self.pin_memory, drop_last=self.drop_last,
                )
                self.num_val_batchs = len(self._val_loader)
        else:
            self._val_loader = DataLoader(
                dataset=self._val_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        if self._test_dataset is not None:
            self._test_loader = DataLoader(
                dataset=self._test_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers,
                pin_memory=self.pin_memory, drop_last=self.drop_last,
            )
            self.num_test_batchs = len(self._test_loader)

        self._model = self._model.to(self._device)
        self._compiled = True

    def feed_forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor=None
    ) -> t.Tuple[torch.Tensor, t.Optional[torch.Tensor]]:
        ## Forward pass.
        logits = self._model.forward(features)
        ## Loss computing if targets provided.
        loss = None
        if targets is not None:
            loss = self._criterion(logits, targets)
        return logits, loss

    def compute_metric(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> t.Tuple[t.Dict[str, torch.Tensor], torch.Tensor]:
        ## Computing of class predictions from model logits.
        results = self._post_process(logits)
        predictions = results[0]
        confidences = results[1]
        ## Metric calculation.
        accuracy_score = _accuracy_score(predictions, targets)
        prediction_score = _precision_score(predictions, targets)
        recall_score = _recall_score(predictions, targets)
        return dict(
            accuracy_score=accuracy_score, precision_score=prediction_score,
            recall_score=recall_score,
        ), confidences

    def train_step(
        self,
        samples_batch: torch.Tensor,
        targets_batch: torch.Tensor,
    ) ->  t.Tuple[t.Dict[str, torch.Tensor], torch.Tensor]:
        ## Forward pass.
        logits, loss = self.feed_forward(samples_batch, targets_batch)
        ## Backward pass: compute gradient.
        loss.backward()
        loss_value = torch.tensor(loss.item(), device=self._device)
        ## Gradient accumulation.
        self.num_acc += logits.shape[0]
        if self.num_acc >= self.gradient_acc:
            self._optimizer.step()
            self.num_acc = 0
            ### Cleaning gradient accumulated.
            self._optimizer.zero_grad()
        ## Metric calculation.
        results, confs = self.compute_metric(logits, targets_batch)
        results.update(dict(loss=loss_value))
        return results, confs

    def eval_step(
        self,
        samples_batch: torch.Tensor,
        targets_batch: torch.Tensor,
    ) -> t.Tuple[t.Dict[str, torch.Tensor], torch.Tensor]:
        ## Forward pass.
        logits, loss = self.feed_forward(samples_batch, targets_batch)
        loss_value = torch.tensor(loss.item(), device=self._device)
        ## Metric calculation.
        results, confs = self.compute_metric(logits, targets_batch)
        results.update(dict(loss=loss_value))
        return results, confs

    def train(self, data_loader) -> t.Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=self._device)
        total_accuracy = torch.tensor(0.0, device=self._device)
        total_precision = torch.tensor(0.0, device=self._device)
        total_recall = torch.tensor(0.0, device=self._device)
        total_confs = torch.tensor(0.0, device=self._device)

        self._model.train()
        for batch_idx, (features, targets) in enumerate(data_loader):
            self.batch_idx = batch_idx
            features = features.to(self._device)
            targets = targets.to(self._device)
            results, confs = self.train_step(features, targets)

            total_loss += results['loss']
            total_accuracy += results['accuracy_score']
            total_precision += results['precision_score']
            total_recall += results['recall_score']
            total_confs += confs.mean(dtype=torch.float32)
            total_batchs = batch_idx + 1

            self.train_loss = total_loss / total_batchs
            self.train_accuracy_score = total_accuracy / total_batchs
            self.train_precision_score = total_precision / total_batchs
            self.train_recall_score = total_recall / total_batchs
            self.train_avg_confidence = total_confs / total_batchs

            self.loss = self.train_loss
            self.accuracy_score = self.train_accuracy_score
            self.precision_score = self.train_precision_score
            self.recall_score = self.train_recall_score
            self.avg_confidence = self.train_avg_confidence

        final_results = dict(
            train_loss=self.train_loss,
            train_accuracy_score=self.train_accuracy_score,
            train_precision_score=self.train_precision_score,
            train_recall_score=self.train_recall_score,
            train_avg_confidence=self.train_avg_confidence,
        )
        return final_results

    def eval(self, data_loader) -> t.Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=self._device)
        total_accuracy = torch.tensor(0.0, device=self._device)
        total_precision = torch.tensor(0.0, device=self._device)
        total_recall = torch.tensor(0.0, device=self._device)
        total_confs = torch.tensor(0.0, device=self._device)

        self._model.eval()
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(data_loader):
                self.batch_idx = batch_idx
                features = features.to(self._device)
                targets = targets.to(self._device)
                results, confs = self.eval_step(features, targets)

                total_loss += results['loss']
                total_accuracy += results['accuracy_score']
                total_precision += results['precision_score']
                total_recall += results['recall_score']
                total_confs += confs.mean(dtype=torch.float32)
                total_batchs = batch_idx + 1

                self.eval_loss = total_loss / total_batchs
                self.eval_accuracy_score = total_accuracy / total_batchs
                self.eval_precision_score = total_precision / total_batchs
                self.eval_recall_score = total_recall / total_batchs
                self.eval_avg_confidence = total_confs / total_batchs

                self.loss = self.eval_loss
                self.accuracy_score = self.eval_accuracy_score
                self.precision_score = self.eval_precision_score
                self.recall_score = self.eval_recall_score
                self.avg_confidence = self.eval_avg_confidence

        final_results = dict(
            eval_loss=self.eval_loss,
            eval_accuracy_score=self.eval_accuracy_score,
            eval_precision_score=self.eval_precision_score,
            eval_recall_score=self.eval_recall_score,
            eval_avg_confidence=self.eval_avg_confidence
        )
        return final_results

    def execute(self) -> EXECUTION_RESULTS:
        if not self._compiled:
            raise RuntimeError(
                "You must call the method called `compile()` in first, "
                "before call the `execute()` method."
            )

        for epoch in range(self.num_epochs):
            self.epoch_idx = epoch
            self.step = "train"
            self.num_batchs = self.num_train_batchs
            results = self.train(self._train_loader)
            self.train_result += results

            if self._val_loader is None:
                continue
            self.step = "val"
            self.num_batchs = self.num_val_batchs
            results = self.eval(self._val_loader)
            results = {
                'val_loss': results['eval_loss'],
                'val_accuracy_score': results['eval_accuracy_score'],
                'val_precision_score': results['eval_precision_score'],
                'val_recall_score': results['eval_recall_score'],
                'val_avg_confidence': results['eval_avg_confidence'],
            }
            self.train_result += results

        if self._test_loader is None:
            return self.train_result.values, None
        self.step = 'test'
        self.num_batchs = self.num_test_batchs
        results = self.eval(self._test_loader)
        self.test_result += results
        return self.train_result.values, self.test_result.values


def fit(
    train_dataset: Dataset,
    model: nn.Module,
    test_dataset: Dataset=None,
    val_dataset: Dataset=None,
    criterion: nn.Module=None,
    optimizer: optim.Optimizer=None,
    post_processing_func: t.Callable[
        [torch.Tensor],
        t.Tuple[
            torch.Tensor,
            torch.Tensor
        ]
    ]=output_function,
    num_epochs: int=3,
    batch_size: int=16,
    num_workers: int=4,
    drop_last: bool=False,
    pin_memory: bool=False,
    val_prop: float=None,
    gradient_acc: int=256,
    device: t.Union[str, torch.device]=None,
) -> EXECUTION_RESULTS:
    """
    Fit a model with concurrent training and monitoring using asyncio.
    """
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    trainer = Trainer(
        train_dataset=train_dataset, val_dataset=val_dataset, model=model,
        criterion=criterion, optimizer=optimizer, test_dataset=test_dataset,
        num_epochs=num_epochs, gradient_acc=gradient_acc,
        batch_size=batch_size, num_workers=num_workers, drop_last=drop_last,
        pin_memory=pin_memory, val_prop=val_prop, device=device,
        post_processing_func=post_processing_func,
    )
    trainer.compile()
    
    import asyncio
    from tqdm import tqdm

    # Shared state for monitoring
    training_complete = asyncio.Event()
    train_results = None
    test_results = None


    async def train():
        """
        Execute training in a separate thread to avoid blocking
        the event loop.
        """
        nonlocal train_results, test_results

        loop = asyncio.get_running_loop()
        # Run the blocking trainer.execute() in the default thread
        # pool executor.
        train_results, test_results = await loop.run_in_executor(
            None, trainer.execute
        )
    
        # Signal that training is complete.
        training_complete.set()
        return train_results, test_results

    """
    async def monitoring():
        \"""Monitor training progress until completion.\"""
        while not training_complete.is_set():
            # Check if trainer has started and has valid epoch information
            # if hasattr(trainer, 'epoch_idx') and hasattr(trainer, 'num_epochs'):
            epochs = trainer.epoch_idx + 1
            remaining = trainer.num_epochs - epochs
            step_info = f"{trainer.step}: " if hasattr(trainer, 'step') else ""
            print(
                f"\r{step_info}[" + ("=" * epochs) + ("." * remaining) + "]",
                end='', flush=True
            )

            # Check more frequently for responsiveness, but not too frequently.
            await asyncio.sleep(0.1)

        # Print final progress bar
        # if hasattr(trainer, 'num_epochs'):
        print(f"\r[" + ("=" * trainer.num_epochs) + "] Complete!", flush=True)
    """

    # async def monitoring():
    #     """Monitor training progress until completion."""
    #     LOGGER.info(
    #         "Number of train batchs: " + str(trainer.num_train_batchs)
    #     )
    #     LOGGER.info(
    #         "Number of val batchs: " + str(trainer.num_val_batchs)
    #     )
    #     LOGGER.info(
    #         "Number of test batchs: " + str(trainer.num_test_batchs)
    #     )
    #     pbar = None
    #     current_step = ''
    #     while not training_complete.is_set():
    #         if trainer.step != current_step or current_progress >= num_batchs:
    #             if pbar is not None:
    #                 # pbar.clear()
    #                 pbar.close()
    #             current_progress = 0
    #             current_step = trainer.step
    #             num_batchs = trainer.num_batchs
    #             pbar = tqdm(total=num_batchs, unit=" batch(s)")
    #             c = len(str(trainer.num_epochs))
    #             pbar.set_description(
    #                 "Epoch: "
    #                 + ("%" + str(c) + "d") % (trainer.epoch_idx + 1,)
    #                 + "/"
    #                 + str(trainer.num_epochs)
    #                 + " - ["
    #                 + f"{current_step:5s}"
    #                 + "]"
    #             )

    #         loss = trainer.loss.detach().cpu()
    #         accuracy_score = trainer.accuracy_score.detach().cpu()
    #         precision_score = trainer.precision_score.detach().cpu()
    #         recall_score = trainer.recall_score.detach().cpu()
    #         avg_confidence = trainer.avg_confidence.detach().cpu()
    #         pbar.set_postfix({
    #             "loss": f"{loss.item():8.6f}",
    #             "acc": f"{accuracy_score.item():4.2f}",
    #             "precision": f"{precision_score.item():4.2f}",
    #             "recall": f"{recall_score.item():4.2f}",
    #             "avg_conf": f"{avg_confidence.item():4.2f}",
    #         })
    #         batchs = trainer.batch_idx + 1
    #         inc = batchs - current_progress
    #         if inc >= 1:
    #             pbar.update(inc)
    #             current_progress = batchs
    #         # Check more frequently for responsiveness,
    #         # but not too frequently.
    #         await asyncio.sleep(0.001)

    async def monitoring():
        """Monitor training progress until completion."""

        # Log initial information
        LOGGER.info(f"Number of train batches: {trainer.num_train_batchs}")
        LOGGER.info(f"Number of val batches: {trainer.num_val_batchs}")
        LOGGER.info(f"Number of test batches: {trainer.num_test_batchs}")

        pbar = None
        current_step = ''
        current_progress = 0
        epoch_width = len(str(trainer.num_epochs))
        metrics = {}
        s_mem = ''

        try:
            while not training_complete.is_set():
                # Check if we need to create a new progress bar
                # (new step or completed previous).
                num_batchs =  getattr(trainer, 'num_batchs', 0)
                if trainer.step != current_step or current_progress >= num_batchs:
                    # Close previous progress bar if it exists.
                    if pbar is not None:
                        curr_progress =  getattr(trainer, 'batch_idx', 0) + 1
                        if s_mem != current_step and curr_progress >= num_batchs:
                            pbar.write(
                                f"Epoch {trainer.epoch_idx + 1:>{epoch_width}}/{trainer.num_epochs} - [{current_step:5s}] "
                                "- " + " - ".join([name + ": " + val for name, val in metrics.items()])
                            )
                            s_mem = current_step
                        pbar.close()

                    # Reset tracking variables.
                    current_progress = 0
                    current_step = trainer.step
                    num_batchs = getattr(trainer, 'num_batchs', 0)

                    # Create new progress bar for current step.
                    pbar = tqdm(
                        total=num_batchs,
                        desc=f"Epoch {trainer.epoch_idx + 1:>{epoch_width}}/{trainer.num_epochs} - [{current_step:5s}]",
                        unit="batch",
                        # ncols=120,  # Fixed width for consistent display;
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]',
                        leave=False  # Don't leave progress bar after completion;
                    )

                # Only update metrics if progress bar exists and trainer
                # has valid data.
                if pbar is not None and hasattr(trainer, 'loss'):
                    try:
                        # Safely extract metrics with error handling.
                        if hasattr(trainer.loss, 'detach'):
                            loss_val = trainer.loss.detach().cpu().item()
                            metrics['loss'] = f"{loss_val:8.6f}"
                        
                        if hasattr(trainer, 'accuracy_score') and hasattr(trainer.accuracy_score, 'detach'):
                            acc_val = trainer.accuracy_score.detach().cpu().item()
                            metrics['acc'] = f"{acc_val:4.2f}"

                        if hasattr(trainer, 'precision_score') and hasattr(trainer.precision_score, 'detach'):
                            prec_val = trainer.precision_score.detach().cpu().item()
                            metrics['precision'] = f"{prec_val:4.2f}"
                        
                        if hasattr(trainer, 'recall_score') and hasattr(trainer.recall_score, 'detach'):
                            rec_val = trainer.recall_score.detach().cpu().item()
                            metrics['recall'] = f"{rec_val:4.2f}"
                        
                        if hasattr(trainer, 'avg_confidence') and hasattr(trainer.avg_confidence, 'detach'):
                            conf_val = trainer.avg_confidence.detach().cpu().item()
                            metrics['conf'] = f"{conf_val:4.2f}"
                        
                        # Update postfix with available metrics.
                        if metrics:
                            pbar.set_postfix(metrics, refresh=False)

                        # Update progress bar position.
                        if hasattr(trainer, 'batch_idx'):
                            current_batch = trainer.batch_idx + 1
                            increment = current_batch - current_progress

                            if increment > 0:
                                # Ensure we don't exceed total.
                                increment = min(
                                    increment, num_batchs - current_progress
                                )
                                pbar.update(increment)
                                current_progress = current_batch

                    except (AttributeError, RuntimeError) as e:
                        # Handle cases where tensors might
                        # not be available yet.
                        pbar.write("Error: %s" % (str(e),))

                # Yield control to event loop.
                # 0.05s is a good balance between responsiveness and CPU usage.
                await asyncio.sleep(0.05)
        
        finally:
            # Ensure progress bar is closed when monitoring ends.
            if pbar is not None:
                pbar.close()
            LOGGER.info("Training monitoring completed.")

    async def run_async_process():
        """Run both tasks concurrently and handle results."""
        # Create tasks
        train_task = asyncio.create_task(train())
        monitor_task = asyncio.create_task(monitoring())

        # Wait for both to complete
        results = await train_task
        await monitor_task  # Ensure monitoring finishes cleanly
        return results

    # Run the async process and return results.
    train_results, test_results = asyncio.run(run_async_process())
    return train_results, test_results


def test_fit_function() -> None:
    import json
    from cva_net.alexnet import ModelFactory as AlexnetModel

    torch.manual_seed(42)
    model = AlexnetModel.build()
    train_dataset = TensorDataset(
        torch.randn((1000, 3, 224, 224)),
        torch.randint(0, 32, (1000,), dtype=torch.int64)
    )
    test_dataset = TensorDataset(
        torch.randn((700, 3, 224, 224)),
        torch.randint(0, 32, (700,), dtype=torch.int64)
    )
    ret = fit(
        train_dataset, model, test_dataset, num_epochs=2, gradient_acc=8,
        val_prop=0.4
    )
    train_results, test_results = ret
    print("\ntrain_results: \n" + json.dumps(train_results, indent=4))
    print("test_results: \n" + json.dumps(test_results, indent=4))
