import logging
import torch
import numpy as np
from sklearn.metrics import accuracy_score
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def accuracy_torch(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute accuracy score using only PyTorch tensors.

    :param y_true: torch.Tensor of true labels.
    :param y_pred: torch.Tensor of predicted labels.
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

"""
def accuracy_torch(y_true, y_pred):
    \"""
    Compute accuracy score using only PyTorch tensors.
    
    Parameters:
    -----------
    y_true : torch.Tensor
        Ground truth labels (1D tensor)
    y_pred : torch.Tensor
        Predicted labels (1D tensor)
    
    Returns:
    --------
    float
        Accuracy score between 0 and 1
    \"""
    # Ensure inputs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    
    # Ensure same shape
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    
    # Calculate accuracy: (number of correct predictions) / (total predictions)
    correct = torch.sum(y_true == y_pred).float()
    total = torch.tensor(y_true.numel(), dtype=torch.float)
    accuracy = correct / total
    
    return accuracy.item()
"""


def test_accuracy_function():
    """
    Test the PyTorch accuracy function against sklearn's implementation.
    """
    LOGGER.debug("=" * 60)
    LOGGER.debug("Testing PyTorch Accuracy Function vs sklearn")
    LOGGER.debug("=" * 60)
    
    # Test 1: Perfect predictions
    LOGGER.debug("\nTest 1: Perfect predictions")
    y_true = torch.tensor([0, 1, 2, 3, 4])
    y_pred = torch.tensor([0, 1, 2, 3, 4])
    
    torch_acc = accuracy_torch(y_true, y_pred)
    sklearn_acc = accuracy_score(y_true.numpy(), y_pred.numpy())
    
    LOGGER.debug(f"PyTorch accuracy: {torch_acc:.6f}")
    LOGGER.debug(f"sklearn accuracy: {sklearn_acc:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_acc, sklearn_acc)}")
    
    # Test 2: All wrong predictions
    LOGGER.debug("\nTest 2: All wrong predictions")
    y_true = torch.tensor([0, 0, 0, 0, 0])
    y_pred = torch.tensor([1, 1, 1, 1, 1])
    
    torch_acc = accuracy_torch(y_true, y_pred)
    sklearn_acc = accuracy_score(y_true.numpy(), y_pred.numpy())
    
    LOGGER.debug(f"PyTorch accuracy: {torch_acc:.6f}")
    LOGGER.debug(f"sklearn accuracy: {sklearn_acc:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_acc, sklearn_acc)}")
    
    # Test 3: 50% accuracy
    LOGGER.debug("\nTest 3: 50% accuracy")
    y_true = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0])
    
    torch_acc = accuracy_torch(y_true, y_pred)
    sklearn_acc = accuracy_score(y_true.numpy(), y_pred.numpy())
    
    LOGGER.debug(f"PyTorch accuracy: {torch_acc:.6f}")
    LOGGER.debug(f"sklearn accuracy: {sklearn_acc:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_acc, sklearn_acc)}")
    
    # Test 4: Binary classification (larger dataset)
    LOGGER.debug("\nTest 4: Binary classification (100 samples)")
    torch.manual_seed(42)
    y_true = torch.randint(0, 2, (100,))
    y_pred = torch.randint(0, 2, (100,))
    
    torch_acc = accuracy_torch(y_true, y_pred)
    sklearn_acc = accuracy_score(y_true.numpy(), y_pred.numpy())
    
    LOGGER.debug(f"PyTorch accuracy: {torch_acc:.6f}")
    LOGGER.debug(f"sklearn accuracy: {sklearn_acc:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_acc, sklearn_acc)}")
    
    # Test 5: Multi-class classification
    LOGGER.debug("\nTest 5: Multi-class classification (10 classes, 200 samples)")
    torch.manual_seed(42)
    y_true = torch.randint(0, 10, (200,))
    y_pred = torch.randint(0, 10, (200,))
    
    torch_acc = accuracy_torch(y_true, y_pred)
    sklearn_acc = accuracy_score(y_true.numpy(), y_pred.numpy())
    
    LOGGER.debug(f"PyTorch accuracy: {torch_acc:.6f}")
    LOGGER.debug(f"sklearn accuracy: {sklearn_acc:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_acc, sklearn_acc)}")
    
    # Test 6: With numpy arrays as input
    LOGGER.debug("\nTest 6: Testing with numpy arrays as input")
    y_true_np = np.array([1, 2, 3, 4, 5])
    y_pred_np = np.array([1, 2, 3, 3, 5])

    y_true_torch = torch.tensor(y_true_np)
    y_pred_torch = torch.tensor(y_pred_np)

    torch_acc = accuracy_torch(y_true_torch, y_pred_torch)
    sklearn_acc = accuracy_score(y_true_np, y_pred_np)
    
    LOGGER.debug(f"PyTorch accuracy: {torch_acc:.6f}")
    LOGGER.debug(f"sklearn accuracy: {sklearn_acc:.6f}")
    LOGGER.debug(f"Match: {np.isclose(torch_acc, sklearn_acc)}")
    
    LOGGER.debug("\n" + "=" * 60)
    LOGGER.debug("All tests completed!")
    LOGGER.debug("=" * 60)
