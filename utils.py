from typing import Optional

import torch


def update_ewma(obs: float, prev: Optional[float], alpha: float) -> float:
    if prev is None:
        return obs
    else:
        return alpha * obs + (1.0 - alpha) * prev


def accuracy(preds, targets):
    return (torch.max(preds, 1)[1] == targets).float().mean()
