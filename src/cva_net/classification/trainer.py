import typing as t

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset


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
        probs = torch.softmax(logits, dim=-1)  # [n, num_classes]
        class_ids = torch.argmax(probs, dim=-1)  # [n,]
        confidences = torch.max(probs, dim=-1).values  # [n,]
        return class_ids, confidences

class Trainer:

    def __init__(
        self,
        *,
        train_dataset: Dataset,
        test_dataset: Dataset,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        post_processing_func: t.Callable[
             [torch.Tensor],
             t.Tuple[torch.Tensor, torch.Tensor]
        ]=output_function,
        num_epochs: int=1,
        batch_size: int=16,
        gradient_acc: int=256,
    ) -> None:
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_acc = gradient_acc
        self.num_batchs = None
        self.epoch_idx = 0
        self.batch_idx = 0
        self.num_acc = 0
        self.history = []

        self._accuracy_score = AccuracyScore()
        self._precision_score = PrecisionScore()
        self._recall_score = RecallScore()
        self._post_process = post_processing_func

    def train_step(
        self,
        sample_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ):
        ## Forward pass.
        logits = self._model.forward(sample_batch)
        ## Loss compute.
        loss = self._criterion(logits, target_batch)
        ## Backward pass: compute gradient.
        loss.backword()
        ## Gradient accumulation.
        self.num_acc += logits.shape[0]
        if self.num_acc >= self.gradient_acc:
            self._optimizer.step()
            self.num_acc = 0
            ### Cleaning gradient accumulated.
            self._optimizer.zero_grad()
        ## Computing of class predictions from model logits.
        results = self.post_processing_func()
        predictions = results[0]
        confidences = results[1]
        ## Metric calculation.
        accuracy_score = self._accuracy_score(predictions, target_batch)
        prediction_score = self._precision_score(predictions, target_batch)
        recall_score = self._recall_score(predictions, target_batch)
        return dict(
            confidences=confidences,
            accuracy_score=accuracy_score,
            prediction_score=prediction_score,
            recall_score=recall_score,
        )

    def execute(self) -> t.List[t.Dict[str, t.Any]]:
        ...
