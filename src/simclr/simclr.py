"""Minimal pytorch implementation of SimCLR [1].


The main method exposed is `step`.

[1] - https://arxiv.org/abs/2002.05709

TODO - GPU support with `device` arguments.
"""
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLearner(nn.Module):
    """Minimal interface for constrastive learning models."""

    def __init__(self, encoder: nn.Module, projection: nn.Module):
        super(ContrastiveLearner, self).__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z


def encode_simclr(
    x: torch.Tensor,
    model: ContrastiveLearner,
    transform: Callable[[torch.Tensor], torch.Tensor],
    normalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper encoding function for SimCLR"""
    t = transform(x)
    h, z = model(t)
    s = F.normalize(z) if normalize else z
    return t, h, s


def get_sim_matrix(
    features: torch.Tensor, temp: Optional[float] = 1.0
) -> torch.Tensor:
    """Computes the matrix of similarities in SimCLR"""
    sims = torch.matmul(features, torch.t(features))
    return sims / temp


def simclr_loss(
    s1: torch.Tensor, s2: torch.Tensor, temp: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Computes and returns the contrastive learning loss from SimCLR + auxiliary data for evaluation"""
    assert s1.size() == s2.size()
    assert s2.dim() == 2
    n, p = s1.size()
    features = torch.cat([s1, s2])
    assert features.size() == (2 * n, p), (features.size(), n, p)
    sims = get_sim_matrix(features, temp)
    assert sims.size() == (2 * n, 2 * n)
    mask = torch.eye(2 * n, dtype=torch.bool)
    logits = sims[~mask].view(2 * n, 2 * n - 1)
    labels = torch.cat([(torch.arange(n) + n - 1), torch.arange(n)])
    return (
        nn.CrossEntropyLoss(reduction="mean")(logits, labels),
        (logits, labels),
    )


DebugInfo = Tuple[torch.Tensor, ...]


def step(
    x: torch.Tensor,
    model: ContrastiveLearner,
    transform: Callable[[torch.Tensor], torch.Tensor],
    temp: Optional[float] = 1.0,
    normalize: Optional[bool] = True,
) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor], DebugInfo]:
    """SimCLR learning step"""
    t1, h1, s1 = encode_simclr(x, model, transform, normalize)
    t2, h2, s2 = encode_simclr(x, model, transform, normalize)
    loss, logits_labels = simclr_loss(s1, s2, temp)
    return (
        loss,
        logits_labels,
        (t1, t2, h1, h2, s1, s2),
    )
