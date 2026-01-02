import torch
import torch.nn.functional as F


def compute_loss(predicted_target: torch.Tensor, target_emb: torch.Tensor):
    """
    Calculation of loss combined MSE + cosine.
    """
    # Perte MSE:
    mse_loss = F.mse_loss(predicted_target, target_emb)
    # Perte cosinus:
    pred_norm = F.normalize(predicted_target, dim=1)
    target_norm = F.normalize(target_emb, dim=1)
    cosine_loss = 1 - torch.mean(torch.sum(pred_norm * target_norm, dim=1))
    # Total losses:
    total_loss = mse_loss + cosine_loss
    return total_loss, mse_loss, cosine_loss
