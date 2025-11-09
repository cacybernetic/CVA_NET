import logging
import typing as t

from typing_extensions import Self
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

LOGGER = logging.getLogger(__name__)
EXECUTION_RESULTS = t.Tuple[
    t.Dict[str, t.List[float]], t.Optional[t.Dict[str, t.List[float]]]
]


def _accuracy_score(y_pred: torch.Tensor, y_true: torch.Tensor):
    """
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
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    average: str='binary',
    pos_label: int=1,
    zero_division: float=0.0
) -> torch.Tensor:
    """
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
    y_true,
    y_pred,
    average='binary',
    pos_label=1,
    zero_division=0.0
) -> torch.Tensor:
    """
    Compute recall score using only PyTorch tensors.

    :param y_true: torch.Tensor or array-like True labels.
    :param y_pred: torch.Tensor or array-like Predicted labels.
    :param average: str, default='binary'
      One of ['binary', 'micro', 'macro', 'weighted', 'none'].
      - 'binary': Only report results for the class specified by pos_label.
      - 'micro': Calculate metrics globally by counting total TP and FN.
      - 'macro': Calculate metrics for each label, return unweighted mean.
      - 'weighted': Calculate metrics for each label, return weighted mean by support.
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
            "Average should be one of ['binary', 'micro', 'macro', 'weighted', 'none']"
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


def output_function(
    logits: torch.Tensor
) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        Post-processing method
        ----------------------

        :param logits: [batch_size, num_classes];
        :returns: tuple of two tensors of size [batch_size,],
          the first tensor contains the class ids predicted
          and the second tensor contains the softmax confidences.
        """
        probs = torch.log_softmax(logits, dim=-1)  # [n, num_classes]
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
        val_dataset: Dataset=None,
        test_dataset: Dataset=None,
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
        if self._device is None:
            self._device = torch.device('cuda') if torch.cuda.is_available() \
                else torch.device('cpu')

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_acc = gradient_acc
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.epoch_idx = 0
        self.batch_idx = 0
        self.num_acc = 0
        self.result = Result()
        self.step = None

        self.train_loss = None
        self.train_accuracy_score = None
        self.train_precision_score = None
        self.train_recall_score = None
        self.train_avg_confidence = None
        self.eval_loss = None
        self.eval_accuracy_score = None
        self.eval_precision_score = None
        self.eval_recall_score = None
        self.eval_avg_confidence = None

        self._post_process = post_processing_func
        # self._accuracy_score = AccuracyScore()
        # self._precision_score = PrecisionScore()
        # self._recall_score = RecallScore()
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._compiled = False

    def compile(self) -> None:
        """This method must be called before execution method."""
        # Creation of data loader.
        self._train_loader = DataLoader(
            dataset=self._train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=self.pin_memory, drop_last=self.drop_last,
        )
        if self._val_dataset is not None:
            self._val_loader = DataLoader(
                dataset=self._val_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        if self._test_dataset is not None:
            self._test_loader = DataLoader(
                dataset=self._test_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
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
        results = self.post_processing_func(logits)
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
        logits, loss = self.feed_forward(samples_batch)
        ## Backward pass: compute gradient.
        loss.backword()
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
        logits, loss = self.feed_forward(samples_batch)
        loss_value = torch.tensor(loss.item(), device=self._device)
        ## Metric calculation.
        results, confs = self.compute_metric(logits, targets_batch)
        results.update(dict(loss=loss_value))
        return results, confs

    def train(self, data_loader) -> t.Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0)
        total_accuracy = torch.tensor(0)
        total_precision = torch.tensor(0)
        total_recall = torch.tensor(0)
        total_confs = torch.tensor(0)

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

        final_results += dict(
            train_loss=self.train_loss,
            train_accuracy_score=self.train_accuracy_score,
            train_precision_score=self.train_precision_score,
            train_recall_score=self.train_recall_score,
            train_avg_confidence=self.train_avg_confidence,
        )
        return final_results

    def eval(self, data_loader) -> t.Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0)
        total_accuracy = torch.tensor(0)
        total_precision = torch.tensor(0)
        total_recall = torch.tensor(0)
        total_confs = torch.tensor(0)

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

        final_results += dict(
            eval_loss=self.eval_loss,
            eval_accuracy_score=self.eval_accuracy_score,
            eval_precision_score=self.eval_precision_score,
            eval_recall_score=self.eval_recall_score,
            eval_avg_score=self.eval_avg_confidence
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
            results = self.train(self._train_loader)
            self.result += results
            self.step = "val"
            results = self.eval(self._val_loader)
            self.result += results

        if self._test_loader is None:
            return self.result.values
        self.step = 'test'
        test_results = self.eval(self._test_loader)
        return self.result.values, test_results
