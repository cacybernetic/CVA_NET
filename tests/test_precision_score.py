import logging
import torch
import numpy as np
from sklearn.metrics import precision_score
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


"""
def precision_score_torch(
    y_true,
    y_pred,
    average='binary',
    pos_label=1,
    zero_division=0.0
) -> torch.Tensor:
    \"""
    Compute precision score using only PyTorch tensors.

    :param y_true: torch.Tensor of true labels.
    :param y_pred: torch.Tensor of predicted labels.
    :param average: str, ['binary', 'micro', 'macro', 'weighted', 'none'].
    :param pos_label: int, label of the positive class
      (for binary classification).
    :param zero_division: float, value to return when there is a zero division.
    
    :returns: precision: torch.Tensor containing the precision score(s).
    \"""
    # Ensure both tensors are on the same device and have the same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} must match")

    # Get unique classes
    classes = torch.unique(torch.cat([y_true, y_pred]))
    classes = classes.sort().values

    # Move tensors to CPU for computation if they're on GPU.
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

        return precision.to(device)

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
            # Micro-precision is the same as accuracy for multi-class.
            total_true_positive = sum([((y_true_cpu == cls) & (y_pred_cpu == cls)).sum().float() for cls in classes])
            total_false_positive = sum([((y_true_cpu != cls) & (y_pred_cpu == cls)).sum().float() for cls in classes])

            denominator = total_true_positive + total_false_positive
            if denominator == 0:
                result = torch.tensor(zero_division, dtype=torch.float32)
            else:
                result = total_true_positive / denominator

        elif average == 'macro':
            # Simple average of per-class precisions.
            result = precisions.mean()

        elif average == 'weighted':
            # Weighted average by support
            # (number of true instances for each class).
            if supports.sum() == 0:
                result = torch.tensor(zero_division, dtype=torch.float32)
            else:
                result = (precisions * supports).sum() / supports.sum()

        elif average == 'none':
            # Return precision for each class.
            result = precisions

        return result.to(device)

    else:
        raise ValueError("Average should be one of ['binary', 'micro', 'macro', 'weighted', 'none']")
"""


def precision_score_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    average: str='binary',
    pos_label: int=1,
    zero_division: float=0.0
):
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

        # Binary precision calculation
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
        return precision.item()

    elif average in ['micro', 'macro', 'weighted', 'none']:
        # Multi-class precision calculation
        precisions = []
        supports = []

        for cls in classes:
            # For each class, treat it as positive and others as negative
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
            # Micro-precision: global TP / (TP + FP)
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
            # Simple average of per-class precisions
            result = precisions.mean()

        elif average == 'weighted':
            # Weighted average by support
            if supports.sum() == 0:
                result = torch.tensor(zero_division, dtype=torch.float32)
            else:
                result = (precisions * supports).sum() / supports.sum()

        elif average == 'none':
            # Return precision for each class
            result = precisions.to(device)
            return result.numpy()

        return result.to(device).item()

    else:
        raise ValueError(
            "Average should be one of "
            "['binary', 'micro', 'macro', 'weighted', 'none']."
        )


def test_precision_function():
    """
    Test the PyTorch precision function against sklearn's implementation.
    """
    LOGGER.debug("=" * 70)
    LOGGER.debug("Testing PyTorch Precision Score Function vs sklearn")
    LOGGER.debug("=" * 70)
    
    # Test 1: Binary classification - perfect predictions
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 1: Binary classification - Perfect predictions")
    LOGGER.debug("=" * 70)
    # torch.manual_seed(42)
    y_true = torch.randint(0, 2, (8,))
    y_pred = y_true.clone()
    
    torch_prec = precision_score_torch(y_true, y_pred, average='binary')
    sklearn_prec = precision_score(y_true.numpy(), y_pred.numpy(), average='binary')
    
    LOGGER.debug(f"PyTorch precision: {torch_prec:.6f}")
    LOGGER.debug(f"sklearn precision: {sklearn_prec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_prec, sklearn_prec)}")
    
    # Test 2: Binary classification - with false positives
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 2: Binary classification - With false positives")
    LOGGER.debug("=" * 70)
    # torch.manual_seed(123)
    y_true = torch.randint(0, 2, (100,))
    y_pred = torch.randint(0, 2, (100,))
    
    torch_prec = precision_score_torch(y_true, y_pred, average='binary')
    sklearn_prec = precision_score(y_true.numpy(), y_pred.numpy(), average='binary')
    
    LOGGER.debug(f"PyTorch precision: {torch_prec:.6f}")
    LOGGER.debug(f"sklearn precision: {sklearn_prec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_prec, sklearn_prec)}")
    
    # Test 3: Binary classification - random data
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 3: Binary classification - Random data (100 samples)")
    LOGGER.debug("=" * 70)
    # torch.manual_seed(456)
    y_true = torch.randint(0, 2, (100,))
    y_pred = torch.randint(0, 2, (100,))
    
    torch_prec = precision_score_torch(y_true, y_pred, average='binary')
    sklearn_prec = precision_score(y_true.numpy(), y_pred.numpy(), average='binary')
    
    LOGGER.debug(f"PyTorch precision: {torch_prec:.6f}")
    LOGGER.debug(f"sklearn precision: {sklearn_prec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_prec, sklearn_prec)}")
    
    # Test 4: Multi-class - macro average
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 4: Multi-class (3 classes) - Macro average")
    LOGGER.debug("=" * 70)
    # torch.manual_seed(789)
    y_true = torch.randint(0, 3, (1500,))
    y_pred = torch.randint(0, 3, (1500,))
    
    torch_prec = precision_score_torch(y_true, y_pred, average='macro', zero_division=0)
    sklearn_prec = precision_score(y_true.numpy(), y_pred.numpy(), average='macro', zero_division=0)
    
    LOGGER.debug(f"PyTorch precision: {torch_prec:.6f}")
    LOGGER.debug(f"sklearn precision: {sklearn_prec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_prec, sklearn_prec)}")
    
    # Test 5: Multi-class - micro average
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 5: Multi-class (5 classes) - Micro average")
    LOGGER.debug("=" * 70)
    # torch.manual_seed(1011)
    y_true = torch.randint(0, 5, (2000,))
    y_pred = torch.randint(0, 5, (2000,))
    
    torch_prec = precision_score_torch(y_true, y_pred, average='micro', zero_division=0)
    sklearn_prec = precision_score(y_true.numpy(), y_pred.numpy(), average='micro', zero_division=0)
    
    LOGGER.debug(f"PyTorch precision: {torch_prec:.6f}")
    LOGGER.debug(f"sklearn precision: {sklearn_prec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_prec, sklearn_prec)}")
    
    # Test 6: Multi-class - weighted average
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 6: Multi-class (4 classes) - Weighted average")
    LOGGER.debug("=" * 70)
    # torch.manual_seed(1213)
    y_true = torch.randint(0, 50, (1800,))
    y_pred = torch.randint(0, 50, (1800,))
    
    torch_prec = precision_score_torch(y_true, y_pred, average='weighted', zero_division=0)
    sklearn_prec = precision_score(y_true.numpy(), y_pred.numpy(), average='weighted', zero_division=0)
    
    LOGGER.debug(f"PyTorch precision: {torch_prec:.6f}")
    LOGGER.debug(f"sklearn precision: {sklearn_prec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_prec, sklearn_prec)}")
    
    # Test 7: Multi-class - per-class precision (none)
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 7: Multi-class (3 classes) - Per-class precision")
    LOGGER.debug("=" * 70)
    # torch.manual_seed(1415)
    y_true = torch.randint(0, 34, (1200,))
    y_pred = torch.randint(0, 34, (1200,))
    
    torch_prec = precision_score_torch(y_true, y_pred, average='none', zero_division=0)
    sklearn_prec = precision_score(y_true.numpy(), y_pred.numpy(), average=None, zero_division=0)
    
    LOGGER.debug(f"PyTorch precision per class: {torch_prec}")
    LOGGER.debug(f"sklearn precision per class: {sklearn_prec}")
    LOGGER.debug(f"Match: {np.allclose(torch_prec, sklearn_prec)}")
    
    # Test 8: Edge case - zero division
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 8: Edge case - No positive predictions (zero division)")
    LOGGER.debug("=" * 70)
    # torch.manual_seed(1617)
    y_true = torch.randint(0, 2, (6,))
    y_pred = torch.zeros(6, dtype=torch.long)
    
    torch_prec = precision_score_torch(y_true, y_pred, average='binary', zero_division=0)
    sklearn_prec = precision_score(y_true.numpy(), y_pred.numpy(), average='binary', zero_division=0)
    
    LOGGER.debug(f"PyTorch precision: {torch_prec:.6f}")
    LOGGER.debug(f"sklearn precision: {sklearn_prec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_prec, sklearn_prec)}")
    
    # Test 9: Large random dataset
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 9: Large random dataset (10 classes, 1000 samples)")
    LOGGER.debug("=" * 70)
    # torch.manual_seed(2024)
    y_true = torch.randint(0, 100, (1500,))
    y_pred = torch.randint(0, 100, (1500,))
    
    for avg_type in ['micro', 'macro', 'weighted']:
        torch_prec = precision_score_torch(y_true, y_pred, average=avg_type, zero_division=0)
        sklearn_prec = precision_score(y_true.numpy(), y_pred.numpy(), average=avg_type, zero_division=0)

        LOGGER.debug(f"\n{avg_type.capitalize()} average:")
        LOGGER.debug(f"  PyTorch: {torch_prec:.6f}")
        LOGGER.debug(f"  sklearn: {sklearn_prec:.6f}")
        LOGGER.debug(f"  Match: {np.isclose(torch_prec, sklearn_prec)}")
    
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("All tests completed!")
    LOGGER.debug("=" * 70)


if __name__ == "__main__":
    test_precision_function()
