import torch
import numpy as np
import random

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mae(preds, labels):
    if not torch.isnan(labels).any():
        return torch.mean(torch.abs(preds - labels))
    
    mask = ~torch.isnan(labels)
    mask = mask.float()
    mask /= torch.mean(mask)
    loss = torch.abs(preds - labels) * mask
    return torch.mean(torch.where(torch.isnan(loss), torch.zeros_like(loss), loss))


def masked_rmse(preds, labels):
    if not torch.isnan(labels).any():
        return torch.sqrt(torch.mean((preds - labels) ** 2))
    
    mask = ~torch.isnan(labels)
    mask = mask.float()
    mask /= torch.mean(mask)
    loss = (preds - labels) ** 2 * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))


def masked_r2(preds, labels):
    if not torch.isnan(labels).any():
        ss_res = torch.sum((labels - preds) ** 2)
        ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
        return 1 - ss_res / ss_tot
    
    mask = ~torch.isnan(labels)
    mask = mask.float()
    mask /= torch.mean(mask)
    
    masked_labels = labels * mask
    masked_preds = preds * mask

    ss_res = torch.sum(((masked_labels - masked_preds) ** 2))
    mean_label = torch.sum(masked_labels) / torch.sum(mask)
    ss_tot = torch.sum(((masked_labels - mean_label) ** 2))
    
    return 1 - ss_res / ss_tot


def normalize_adj(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    对稠密邻接矩阵做对称归一化  A_hat = D^{-1/2} A D^{-1/2}
    """
    deg = adj.sum(-1, keepdim=True).clamp(min=eps)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt * adj * d_inv_sqrt.transpose(0, 1)
