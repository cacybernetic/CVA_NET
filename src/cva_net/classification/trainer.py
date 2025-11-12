import logging
import random
import math
import typing as t

from typing_extensions import Self
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)
EXECUTION_RESULTS = t.Tuple[
    t.Dict[str, t.List[float]], t.Optional[t.Dict[str, t.List[float]]]
]

'''
class DataIterator:
    """Dynamic data loading implementation.

    :type dataset: `typing.Any`
    :type batch_size: `int`
    :type shuffle: `bool`
    :type initial_samples_indices: `list` of `int`
    """

    def __init__(
        self, dataset, batch_size=1, shuffle=False,
        initial_samples_indices=None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._sample_indices = initial_samples_indices

        if not self._sample_indices \
                or len(self._sample_indices) > len(self.dataset):
            self._sample_indices = list(range(len(dataset)))

        self._batch_index = 0
        self._length = None

    @property
    def batch_index(self):
        """int: returns the current value of batch index"""
        return self._batch_index

    @property
    def sample_indices(self):
        """list of int: returns the list of batch indices"""
        return self._sample_indices

    @sample_indices.setter
    def sample_indices(self, value):
        if not value:
            return
        self._sample_indices = value

    def reset_iteration(self):
        if self.shuffle:
            random.shuffle(self._sample_indices)
        self._batch_index = 0

    def set_batch_index(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f"Value index {index} is out of range.")
        self._batch_index = index

    def __len__(self):
        if self._length is None:
            sample_len = len(self.dataset)
            batch_count = math.ceil(sample_len / self.batch_size)
            self._length = batch_count
        return self._length

    def __iter__(self):
        return self

    def __next__(self):
        if self._batch_index < len(self):
            sample_indices_len = len(self._sample_indices)
            sample_index = self._batch_index * self.batch_size
            samples = []
            first_sample = self.dataset[self._sample_indices[sample_index]]
            if not first_sample:
                # raise StopIteration("Sample is none. So, end of iteration.")
                self._batch_index += 1
                return self.__next__()
            if not isinstance(first_sample, tuple):
                raise TypeError(
                    "All the sample returned by the dataset"
                    "must be formatted in tuple.")
            samples.append(first_sample)
            col_len = len(first_sample)

            start = sample_index + 1
            end = sample_index + self.batch_size
            if end > sample_indices_len:
                end = sample_indices_len
            batch_iter = range(start, end)

            for i in batch_iter:
                sample = self.dataset[self._sample_indices[i]]
                if not sample:
                    self._batch_index += 1
                    return self.__next__()
                if len(sample) != col_len:
                    raise ValueError(
                        "Some sample has not same length with the first.")
                samples.append(sample)

            samples_len = len(samples)
            sample_batch = []
            for col in range(col_len):
                # arr = samples[0][col]
                arrays = []
                for i in range(samples_len):
                    x = samples[i][col]
                    arrays.append(x)
                arrays = np.array(arrays)
                arrays = torch.tensor(arrays)
                sample_batch.append(arrays)

            sample_batch = tuple(sample_batch)
            self._batch_index += 1
            return sample_batch
        else:
            self.reset_iteration()
            raise StopIteration("End of data iteration!")
'''

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
    indices = torch.randint(0, val_dataset_len, (val_dataset_len,))
    # indices = torch.arange(0, val_dataset_len, 1)
    features = []
    targets = []
    for index in indices:
        feature, target = dataset_instance[index.item()]
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
        verbose: bool=True,
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.post_process = post_processing_func
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() \
                else torch.device('cpu')
        elif isinstance(self.device, str):
            self.device = torch.device(self.device)

        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._gradient_acc = gradient_acc
        self._num_workers = num_workers
        self._drop_last = drop_last
        self._pin_memory = pin_memory
        self._val_prop = val_prop
        if not self._val_prop:
            self._val_prop = 0.2
        self._verbose = verbose
        
        self.num_batchs = 0
        self.num_train_batchs = -1
        self.num_val_batchs = -1
        self.num_test_batchs = -1
        self.epoch_idx = 0
        self.batch_idx = 0
        self.num_acc = 0
        self.train_result = Result()
        self.val_result = Result()
        self.test_result = Result()
        self.step = ''

        self.train_loss = torch.tensor(0.0, device=self.device)
        self.train_accuracy_score = torch.tensor(0.0, device=self.device)
        self.train_precision_score = torch.tensor(0.0, device=self.device)
        self.train_recall_score = torch.tensor(0.0, device=self.device)
        self.train_avg_confidence = torch.tensor(0.0, device=self.device)
        self.eval_loss = torch.tensor(0.0, device=self.device)
        self.eval_accuracy_score = torch.tensor(0.0, device=self.device)
        self.eval_precision_score = torch.tensor(0.0, device=self.device)
        self.eval_recall_score = torch.tensor(0.0, device=self.device)
        self.eval_avg_confidence = torch.tensor(0.0, device=self.device)
        # self.loss = torch.tensor(0.0, device=self.device)
        # self.accuracy_score = torch.tensor(0.0, device=self.device)
        # self.precision_score = torch.tensor(0.0, device=self.device)
        # self.recall_score = torch.tensor(0.0, device=self.device)
        # self.avg_confidence = torch.tensor(0.0, device=self.device)

        # self._accuracy_score = AccuracyScore()
        # self._precision_score = PrecisionScore()
        # self._recall_score = RecallScore()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self._compiled = False
        self.epoch_str_width = len(str(self._num_epochs))
        self.pbar = None

    def compile(self) -> None:
        """This method must be called before execution method."""
        # Creation of data loader.
        self.train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self._batch_size,
            shuffle=True, num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )
        # self.train_loader = DataIterator(
        #     dataset=self.train_dataset, batch_size=self._batch_size,
        #     shuffle=True,
        # )
        self.num_train_batchs = len(self.train_loader)
        if self.val_dataset is None:
            if self.test_dataset is not None:
                self.val_dataset = get_validation_dataset(
                    self.test_dataset, self._val_prop
                )
                self.val_loader = DataLoader(
                    dataset=self.val_dataset, batch_size=self._batch_size,
                    shuffle=False, num_workers=self._num_workers,
                    pin_memory=self._pin_memory, drop_last=self._drop_last,
                )
                # self.val_loader = DataIterator(
                #     dataset=self.val_dataset, batch_size=self._batch_size,
                #     shuffle=False,
                # )
                self.num_val_batchs = len(self.val_loader)
        else:
            self.val_loader = DataLoader(
                dataset=self.val_dataset, batch_size=self._batch_size,
                shuffle=False, num_workers=self._num_workers,
                pin_memory=self._pin_memory, drop_last=self._drop_last,
            )
            # self.val_loader = DataIterator(
            #     dataset=self.val_dataset, batch_size=self._batch_size,
            #     shuffle=False,
            # )
        if self.test_dataset is not None:
            self.test_loader = DataLoader(
                dataset=self.test_dataset, batch_size=self._batch_size,
                shuffle=False, num_workers=self._num_workers,
                pin_memory=self._pin_memory, drop_last=self._drop_last,
            )
            # self.test_loader = DataIterator(
            #     dataset=self.test_dataset, batch_size=self._batch_size,
            #     shuffle=False,
            # )
            self.num_test_batchs = len(self.test_loader)

        self.model = self.model.to(self.device)
        self._compiled = True

    def feed_forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor=None
    ) -> t.Tuple[torch.Tensor, t.Optional[torch.Tensor]]:
        ## Forward pass.
        logits = self.model.forward(features)
        ## Loss computing if targets provided.
        loss = None
        if targets is not None:
            loss = self.criterion(logits, targets)
        return logits, loss

    def compute_metric(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> t.Tuple[t.Dict[str, torch.Tensor], torch.Tensor]:
        ## Computing of class predictions from model logits.
        results = self.post_process(logits)
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
        loss_value = torch.tensor(loss.item(), device=self.device)
        ## Gradient accumulation.
        self.num_acc += logits.shape[0]
        if self.num_acc >= self._gradient_acc:
            self.optimizer.step()
            self.num_acc = 0
            ### Cleaning gradient accumulated.
            self.optimizer.zero_grad()
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
        loss_value = torch.tensor(loss.item(), device=self.device)
        ## Metric calculation.
        results, confs = self.compute_metric(logits, targets_batch)
        results.update(dict(loss=loss_value))
        return results, confs

    def get_progress_bar(self):
        desc = ''
        unit = ''
        total = None
        if self.step in ('train', 'test', 'val'):
            desc = f"Epoch {self.epoch_idx + 1:>{self.epoch_str_width}}/{self._num_epochs} - [{self.step:5s}]"
            unit = ' batch(s)'
        if self.step == 'train':
            total = self.num_train_batchs
        elif self.step == 'val':
            total = self.num_val_batchs
        elif self.step == 'test':
            total = self.num_test_batchs

        iterator = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            # ncols=120,  # Fixed width for consistent display;
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]',
            leave=False  # Don't leave progress bar after completion;
        )
        return iterator

    def _result_json_format(self, **metric: t.Dict[str, torch.Tensor]) -> None:
        json_metric = {}
        for name, value in metric.items():
            value = value.detach().cpu().item()
            if 'loss' in name:
                json_metric[name] = "%7.4f" % (value,)
            elif 'score' in name or 'confidence' in name:
                value = value * 100.0
                json_metric[name] = "%5.1f%%" % (value,)
            else:
            	json_metric[name] = "%.3f" % (value,)
        return json_metric

    def train(self, data_loader) -> t.Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=self.device)
        total_accuracy = torch.tensor(0.0, device=self.device)
        total_precision = torch.tensor(0.0, device=self.device)
        total_recall = torch.tensor(0.0, device=self.device)
        total_confs = torch.tensor(0.0, device=self.device)
        metrics = {}

        self.model.train()
        for batch_idx, (features, targets) in enumerate(data_loader):
            self.batch_idx = batch_idx
            features = features.to(self.device)
            targets = targets.to(self.device)
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

            # self.loss = self.train_loss
            # self.accuracy_score = self.train_accuracy_score
            # self.precision_score = self.train_precision_score
            # self.recall_score = self.train_recall_score
            # self.avg_confidence = self.train_avg_confidence
            if self._verbose:
                self.pbar.update(1)

            metrics = dict(
                loss=self.train_loss,
                accuracy_score=self.train_accuracy_score,
                precision_score=self.train_precision_score,
                recall_score=self.train_recall_score,
                avg_confidence=self.train_avg_confidence,
            )
            if self._verbose:
                self.pbar.set_postfix(
                    self._result_json_format(**metrics), refresh=False
                )

        return metrics

    def eval(self, data_loader) -> t.Dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=self.device)
        total_accuracy = torch.tensor(0.0, device=self.device)
        total_precision = torch.tensor(0.0, device=self.device)
        total_recall = torch.tensor(0.0, device=self.device)
        total_confs = torch.tensor(0.0, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(data_loader):
                self.batch_idx = batch_idx
                features = features.to(self.device)
                targets = targets.to(self.device)
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

                # self.loss = self.eval_loss
                # self.accuracy_score = self.eval_accuracy_score
                # self.precision_score = self.eval_precision_score
                # self.recall_score = self.eval_recall_score
                # self.avg_confidence = self.eval_avg_confidence
                if self._verbose:
                    self.pbar.update(1)

                metrics = dict(
                    loss=self.eval_loss,
                    accuracy_score=self.eval_accuracy_score,
                    precision_score=self.eval_precision_score,
                    recall_score=self.eval_recall_score,
                    avg_confidence=self.eval_avg_confidence,
                )
                if self._verbose:
                    self.pbar.set_postfix(
                        self._result_json_format(**metrics), refresh=False
                    )

        return metrics

    def execute(self) -> EXECUTION_RESULTS:
        if not self._compiled:
            raise RuntimeError(
                "You must call the method called `compile()` in first, "
                "before call the `execute()` method."
            )

        for epoch in range(self._num_epochs):
            self.epoch_idx = epoch

            self.step = "train"
            self.num_batchs = self.num_train_batchs
            if self._verbose:
                self.pbar = self.get_progress_bar()

            results = self.train(self.train_loader)
            self.train_result += results

            # update iterator:
            if self._verbose:
                results = self._result_json_format(**results)
                self.pbar.write(
                    f"Epoch {self.epoch_idx + 1:>{self.epoch_str_width}}/{self._num_epochs} - [{self.step:5s}] "
                    "- " + " - ".join([name + ": " + val for name, val in results.items()])
                )
                self.pbar.close()

            if self.val_loader is None:
                continue
            ###################################################################
            self.step = "val"
            self.num_batchs = self.num_val_batchs
            if self._verbose:
                self.pbar = self.get_progress_bar()

            results = self.eval(self.val_loader)
            self.val_result += results

            # update progress bar:
            if self._verbose:
                results = self._result_json_format(**results)
                self.pbar.write(
                    f"Epoch {self.epoch_idx + 1:>{self.epoch_str_width}}/{self._num_epochs} - [{self.step:5s}] "
                    "- " + " - ".join([name + ": " + val for name, val in results.items()])
                )
                self.pbar.close()

        if self.test_loader is None:
            return (
                self.train_result.values,
                self.val_result.values,
                self.test_result.values
            )
        #######################################################################
        self.step = 'test'
        self.num_batchs = self.num_test_batchs
        if self._verbose:
            self.pbar = self.get_progress_bar()

        results = self.eval(self.test_loader)
        self.test_result += results

        # update progress bar:
        if self._verbose:
            results = self._result_json_format(**results)
            self.pbar.write(
                f"\n[{self.step:5s}] " \
                + " - ".join([name + ": " + val for name, val in results.items()])
            )
            self.pbar.close()

        return (
            self.train_result.values,
            self.val_result.values,
            self.test_result.values
        )


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
        post_processing_func=post_processing_func, verbose=True,
    )
    trainer.compile()
    results = trainer.execute()
    return results


def test_fit_function() -> None:
    import json
    from cva_net.alexnet import ModelFactory as AlexnetModel

    torch.manual_seed(42)
    model, _ = AlexnetModel.build()
    train_dataset = TensorDataset(
        torch.randn((100, 3, 224, 224)),
        torch.randint(0, 32, (100,), dtype=torch.int64)
    )
    test_dataset = TensorDataset(
        torch.randn((70, 3, 224, 224)),
        torch.randint(0, 32, (70,), dtype=torch.int64)
    )
    ret = fit(
        train_dataset, model, test_dataset, num_epochs=2, gradient_acc=8,
        val_prop=0.4, num_workers=1, batch_size=1
    )
    train_results, val_results, test_results = ret
    print("\ntrain_results: \n" + json.dumps(train_results, indent=4))
    print("\nval_results: \n" + json.dumps(val_results, indent=4))
    print("test_results: \n" + json.dumps(test_results, indent=4))
