"""Minimal pytorch implementation of SimCLR [1].  

[1] - https://arxiv.org/abs/2002.05709

TODO - GPU support
"""
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType  # type: ignore


class ContrastiveLearner(nn.Module):
    """Minimal module class for constrastive learning models."""

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
    model: nn.Module,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, TensorType["b", "e1"], TensorType["b", "e2"]]:
    """Helper encoding function for SimCLR"""
    t = transform(x)
    h, z = model(t)
    s = F.normalize(z)
    return t, h, s


def get_sim_matrix(
    features: TensorType["2b", "e2"], temp: Optional[float] = 1.0
) -> TensorType["2b", "2b"]:
    """Computes the matrix of similarities in SimCLR"""
    sims = torch.matmul(features, torch.t(features))
    return sims / temp


def manual_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor,
) -> TensorType[1, torch.float]:
    """Naive cross entropy impl, to double check"""
    assert labels.dim() == 1, labels.size()
    assert logits.dim() == 2, logits.size()
    assert logits.size(0) == labels.size(0), (logits.size(0), labels.size(0))
    labels = labels.unsqueeze(1)
    positive = torch.gather(logits, 1, labels).squeeze(1)
    return torch.mean(torch.logsumexp(logits, dim=1) - positive)


def simclr_loss(
    s1: TensorType["b", "e2"],
    s2: TensorType["b", "e2"],
    temp: Optional[float] = 1.0,
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


def step(
    x: torch.Tensor,
    model: nn.Module,
    transform: Callable[[torch.Tensor], torch.Tensor],
    temp: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, float, Tuple[torch.Tensor, torch.Tensor]]:
    """SimCLR learning step"""
    t1, h1, s1 = encode_simclr(x, model, transform)
    t2, h2, s2 = encode_simclr(x, model, transform)
    loss, logits_labels = simclr_loss(s1, s2, temp)
    return (t1, t2, s1, s2), loss, logits_labels
