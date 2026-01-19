import torch


def score(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    r"""
    Compute accuracy score using only PyTorch tensors.

    :param y_pred: torch.Tensor of predicted labels.
    :param y_true: torch.Tensor of true labels.
    :returns: A torch.Tensor containing the accuracy score.
    """
    # Ensure both tensors are on the same device and have the same shape.
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true %s and y_pred %s must match." % (str(y_true.shape), str(y_pred.shape)))

    # Calculate number of correct predictions:
    correct = torch.eq(y_true, y_pred).sum()

    # Calculate total number of predictions:
    total = torch.tensor(y_true.shape[0], dtype=torch.float32, device=y_true.device)

    # Compute accuracy
    accuracy = correct.float() / total
    return accuracy
