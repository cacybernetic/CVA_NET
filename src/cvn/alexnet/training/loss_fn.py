import torch
import torch.nn.functional as F
from torch import nn


class LossFunction(nn.Module):

    def __init__(self, num_classes: int=1)  -> None:
        super().__init__()
        self._compute_loss = self._cross_entropy if num_classes > 1 else self._binary_cross_entropy

    def _binary_cross_entropy(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = F.sigmoid(y_pred)
        y_true = y_true.float()
        return F.binary_cross_entropy(y_pred, y_true)

    def _cross_entropy(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(y_pred, y_true)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self._compute_loss(y_pred, y_true)
