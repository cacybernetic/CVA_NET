import typing as t
import torch


class Score:

    def __init__(
        self,
        average: t.Literal['binary', 'micro', 'macro', 'weighted', 'none']='macro',
        pos_label: int=1,
        zero_division: float=0.0,
    ) -> None:
        """
        Method to create a new instance of precision score.

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
        """
        self._average = average
        self._pos_label = pos_label
        self._zero_division = zero_division

    def compute(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""
        Compute recall score using only PyTorch tensors.

        :param y_true: torch.Tensor or array-like True labels.
        :param y_pred: torch.Tensor or array-like Predicted labels.

        returns: torch.Tensor of float of Recall score(s).
        """
        # Convert to tensors if needed.
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)

        # Ensure both tensors have the same shape.
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} must match.")

        # Get unique classes.
        classes = torch.unique(torch.cat([y_true, y_pred]))
        classes = classes.sort().values

        # Store original device.
        device = y_true.device

        if self._average == 'binary':
            if len(classes) > 2:
                raise ValueError("Target is multiclass but average='binary'. Please choose another average setting.")

            # Binary recall calculation: TP / (TP + FN).
            true_positive = ((y_true == self._pos_label) & (y_pred == self._pos_label)).sum().float()
            false_negative = ((y_true == self._pos_label) & (y_pred != self._pos_label)).sum().float()

            denominator = true_positive + false_negative
            if denominator == 0:
                recall = torch.tensor(self._zero_division, dtype=torch.float32, device=device)
            else:
                recall = true_positive / denominator
                recall = recall.to(device)

            return recall

        elif self._average in ['micro', 'macro', 'weighted', 'none']:
            # Multi-class recall calculation
            recalls = []
            supports = []

            for cls in classes:
                # For each class, treat it as positive and others as negative
                true_positive = ((y_true == cls) & (y_pred == cls)).sum().float()
                false_negative = ((y_true == cls) & (y_pred != cls)).sum().float()

                denominator = true_positive + false_negative
                if denominator == 0:
                    recall_cls = torch.tensor(self._zero_division, dtype=torch.float32, device=device)
                else:
                    recall_cls = true_positive / denominator

                recalls.append(recall_cls)
                supports.append((y_true == cls).sum().float())

            recalls = torch.stack(recalls)
            supports = torch.stack(supports)

            if self._average == 'micro':
                # Micro-recall: global TP / (TP + FN)
                total_true_positive = sum([((y_true == cls) & (y_pred == cls)).sum().float() for cls in classes])
                total_false_negative = sum([((y_true == cls) & (y_pred != cls)).sum().float() for cls in classes])

                denominator = total_true_positive + total_false_negative
                if denominator == 0:
                    result = torch.tensor(self._zero_division, dtype=torch.float32)
                else:
                    result = total_true_positive / denominator

            elif self._average == 'macro':
                # Simple average of per-class recalls
                result = recalls.mean()

            elif self._average == 'weighted':
                # Weighted average by support
                if supports.sum() == 0:
                    result = torch.tensor(self._zero_division, dtype=torch.float32)
                else:
                    result = (recalls * supports).sum() / supports.sum()

            elif self._average == 'none':
                # Return recall for each class
                result = recalls.to(device)
                return result

            return result.to(device)

        else:
            raise ValueError("Average should be one of ['binary', 'micro', 'macro', 'weighted', 'none']")

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.compute(y_pred, y_true)
