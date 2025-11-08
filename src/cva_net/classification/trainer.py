import typing as t

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader


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
        self.history = []
        self.step = None

        self._post_process = post_processing_func
        self._accuracy_score = AccuracyScore()
        self._precision_score = PrecisionScore()
        self._recall_score = RecallScore()
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

    def compute_metric(self, logits, targets) -> t.Dict[str, torch.Tensor]:
        ## Computing of class predictions from model logits.
        results = self.post_processing_func(logits)
        predictions = results[0]
        confidences = results[1]
        ## Metric calculation.
        accuracy_score = self._accuracy_score(predictions, targets)
        prediction_score = self._precision_score(predictions, targets)
        recall_score = self._recall_score(predictions, targets)
        return dict(
            confidences=confidences, accuracy_score=accuracy_score,
            prediction_score=prediction_score, recall_score=recall_score,
        )

    def eval_step(
        self,
        sample_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> t.Dict[str, torch.Tensor]:
        ## Forward pass.
        logits = self._model.forward(sample_batch)
        ## Loss compute.
        loss = self._criterion(logits, target_batch)
        ## Metric calculation.
        results = self.compute_metric(logits, target_batch)
        results.update(dict(loss=torch.tensor(loss.item())))
        return results

    def train_step(
        self,
        sample_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> t.Dict[str, torch.Tensor]:
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
        ## Metric calculation.
        results = self.compute_metric(logits, target_batch)
        results.update(dict(loss=torch.tensor(loss.item())))
        return results
    
    def _update_history(self, results) -> None:
        ...

    def execute(self) -> t.List[t.Dict[str, t.Any]]:
        for epoch in range(self.num_epochs):
            self.epoch_idx = epoch
            self.step = "train"
            self._model.train()
            train_data_iterator = enumerate(self._train_loader)
            for batch_idx, (features, targets) in train_data_iterator:
                self.batch_idx = batch_idx
                features = features.to(self._device)
                targets = targets.to(self._device)
                results = self.train_step(features, targets)
                self._update_history(results)

            if self._val_loader is None:
                continue
            self.step = 'val'
            self._model.eval()
            with torch.no_grad():
                val_data_iterator = enumerate(self._val_loader)
                for batch_idx, (features, targets) in val_data_iterator:
                    self.batch_idx = batch_idx
                    features = features.to(self._device)
                    targets = targets.to(self._device)
                    results = self.eval_step(features, targets)
                    self._update_history(results)
        
        if self._test_loader is None:
            return self.history
        self.step = 'test'
        self._model.eval()
        with torch.no_grad():
            val_data_iterator = enumerate(self._val_loader)
            for batch_idx, (features, targets) in val_data_iterator:
                self.batch_idx = batch_idx
                features = features.to(self._device)
                targets = targets.to(self._device)
                results = self.eval_step(features, targets)
                self._update_history(results)
        return self.history
