import logging
import torch
import numpy as np
from sklearn.metrics import recall_score

LOGGER = logging.getLogger(__name__)


def recall_score_torch(
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


def test_recall_function():
    """
    Test the PyTorch recall function against sklearn's implementation.
    """
    LOGGER.debug("=" * 70)
    LOGGER.debug("Testing PyTorch Recall Score Function vs sklearn")
    LOGGER.debug("=" * 70)
    
    # Test 1: Binary classification - perfect predictions
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 1: Binary classification - Perfect predictions")
    LOGGER.debug("=" * 70)
    y_true = torch.randint(0, 2, (8,))
    y_pred = y_true.clone()
    
    torch_rec = recall_score_torch(y_true, y_pred, average='binary')
    sklearn_rec = recall_score(y_true.numpy(), y_pred.numpy(), average='binary')

    LOGGER.debug(f"PyTorch recall: {torch_rec:.6f}")
    LOGGER.debug(f"sklearn recall: {sklearn_rec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_rec.numpy(), sklearn_rec)}")
    
    # Test 2: Binary classification - with false negatives
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 2: Binary classification - With false negatives")
    LOGGER.debug("=" * 70)
    y_true = torch.randint(0, 2, (100,))
    y_pred = torch.randint(0, 2, (100,))
    
    torch_rec = recall_score_torch(y_true, y_pred, average='binary')
    sklearn_rec = recall_score(y_true.numpy(), y_pred.numpy(), average='binary')
    
    LOGGER.debug(f"PyTorch recall: {torch_rec:.6f}")
    LOGGER.debug(f"sklearn recall: {sklearn_rec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_rec.numpy(), sklearn_rec)}")
    
    # Test 3: Binary classification - random data
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 3: Binary classification - Random data (100 samples)")
    LOGGER.debug("=" * 70)
    y_true = torch.randint(0, 2, (100,))
    y_pred = torch.randint(0, 2, (100,))
    
    torch_rec = recall_score_torch(y_true, y_pred, average='binary')
    sklearn_rec = recall_score(y_true.numpy(), y_pred.numpy(), average='binary')
    
    LOGGER.debug(f"PyTorch recall: {torch_rec:.6f}")
    LOGGER.debug(f"sklearn recall: {sklearn_rec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_rec.numpy(), sklearn_rec)}")
    
    # Test 4: Multi-class - macro average
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 4: Multi-class (3 classes) - Macro average")
    LOGGER.debug("=" * 70)
    y_true = torch.randint(0, 3, (1500,))
    y_pred = torch.randint(0, 3, (1500,))
    
    torch_rec = recall_score_torch(y_true, y_pred, average='macro', zero_division=0)
    sklearn_rec = recall_score(y_true.numpy(), y_pred.numpy(), average='macro', zero_division=0)
    
    LOGGER.debug(f"PyTorch recall: {torch_rec:.6f}")
    LOGGER.debug(f"sklearn recall: {sklearn_rec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_rec.numpy(), sklearn_rec)}")
    
    # Test 5: Multi-class - micro average
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 5: Multi-class (5 classes) - Micro average")
    LOGGER.debug("=" * 70)
    y_true = torch.randint(0, 5, (2000,))
    y_pred = torch.randint(0, 5, (2000,))
    
    torch_rec = recall_score_torch(y_true, y_pred, average='micro', zero_division=0)
    sklearn_rec = recall_score(y_true.numpy(), y_pred.numpy(), average='micro', zero_division=0)
    
    LOGGER.debug(f"PyTorch recall: {torch_rec:.6f}")
    LOGGER.debug(f"sklearn recall: {sklearn_rec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_rec.numpy(), sklearn_rec)}")
    
    # Test 6: Multi-class - weighted average
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 6: Multi-class (50 classes) - Weighted average")
    LOGGER.debug("=" * 70)
    y_true = torch.randint(0, 50, (1800,))
    y_pred = torch.randint(0, 50, (1800,))
    
    torch_rec = recall_score_torch(y_true, y_pred, average='weighted', zero_division=0)
    sklearn_rec = recall_score(y_true.numpy(), y_pred.numpy(), average='weighted', zero_division=0)
    
    LOGGER.debug(f"PyTorch recall: {torch_rec:.6f}")
    LOGGER.debug(f"sklearn recall: {sklearn_rec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_rec.numpy(), sklearn_rec)}")
    
    # Test 7: Multi-class - per-class recall (none)
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 7: Multi-class (34 classes) - Per-class recall")
    LOGGER.debug("=" * 70)
    y_true = torch.randint(0, 34, (1200,))
    y_pred = torch.randint(0, 34, (1200,))
    
    torch_rec = recall_score_torch(y_true, y_pred, average='none', zero_division=0)
    sklearn_rec = recall_score(y_true.numpy(), y_pred.numpy(), average=None, zero_division=0)
    
    LOGGER.debug(f"PyTorch recall per class: {torch_rec}")
    LOGGER.debug(f"sklearn recall per class: {sklearn_rec}")
    LOGGER.debug(f"Match: {np.allclose(torch_rec.numpy(), sklearn_rec)}")
    
    # Test 8: Edge case - zero division
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 8: Edge case - No positive predictions (zero division)")
    LOGGER.debug("=" * 70)
    y_true = torch.randint(0, 2, (6,))
    y_pred = torch.zeros(6, dtype=torch.long)
    
    torch_rec = recall_score_torch(y_true, y_pred, average='binary', zero_division=0)
    sklearn_rec = recall_score(y_true.numpy(), y_pred.numpy(), average='binary', zero_division=0)
    
    LOGGER.debug(f"PyTorch recall: {torch_rec:.6f}")
    LOGGER.debug(f"sklearn recall: {sklearn_rec:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_rec.numpy(), sklearn_rec)}")
    
    # Test 9: Large random dataset
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("Test 9: Large random dataset (100 classes, 1500 samples)")
    LOGGER.debug("=" * 70)
    y_true = torch.randint(0, 100, (1500,))
    y_pred = torch.randint(0, 100, (1500,))
    
    for avg_type in ['micro', 'macro', 'weighted']:
        torch_rec = recall_score_torch(y_true, y_pred, average=avg_type, zero_division=0)
        sklearn_rec = recall_score(y_true.numpy(), y_pred.numpy(), average=avg_type, zero_division=0)

        LOGGER.debug(f"\n{avg_type.capitalize()} average:")
        LOGGER.debug(f"  PyTorch: {torch_rec:.6f}")
        LOGGER.debug(f"  sklearn: {sklearn_rec:.6f}")
        LOGGER.debug(f"  Match: {np.isclose(torch_rec.numpy(), sklearn_rec)}")
    
    LOGGER.debug("\n" + "=" * 70)
    LOGGER.debug("All tests completed!")
    LOGGER.debug("=" * 70)


if __name__ == "__main__":
    test_recall_function()
