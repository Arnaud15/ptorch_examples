"""Training utilities."""
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from scipy.stats import beta
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float, n_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Mixup to a batch of training examples in classification."""
    assert alpha > 0.0
    assert x.dim() > 1
    assert x.size(0) == y.size(0)
    if y.dim() == 1:
        # we must one-hot encode y before the convex combination
        y = nn.functional.one_hot(y, num_classes=n_classes)
    else:
        assert y.dim() == 2
    permut = torch.randperm(x.size(0))
    lamb = beta.rvs(alpha, alpha)
    lamb = max(1.0 - lamb, lamb)
    x_right = x[permut]
    y_right = y[permut]
    x_mix = x * lamb + x_right * (1.0 - lamb)
    y_mix = y * lamb + y_right * (1.0 - lamb)
    return (x_mix, y_mix)


def update_ewma(obs: float, prev: Optional[float], alpha: float) -> float:
    """Updates an exponentially weighted moving average."""
    if prev is None:
        return obs
    else:
        return alpha * obs + (1.0 - alpha) * prev


def accuracy(logits, labels, top_k: int = 5):
    """Compute the top-k accuracy of logits given labels."""
    assert isinstance(top_k, int)
    assert top_k >= 1
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    else:
        assert labels.dim() == 2
    argmaxes = torch.topk(logits, k=top_k, dim=1)[1]
    return (argmaxes == labels).sum(1).float().mean()


def get_optimizer(
    decay_params: nn.parameter.Parameter,
    no_decay_params: nn.parameter.Parameter,
    lr: float,
    momentum: float,
    weight_decay: float,
    use_adam: bool = False,
) -> optim.Optimizer:
    """Utility to get the optimizer and handle weight decay."""
    if use_adam:
        raise NotImplementedError("Adam not implemented yet")
    else:
        # SGD with momentum
        other_params = {"lr": lr, "momentum": momentum, "nesterov": True}
        no_decay = dict({"params": no_decay_params}, **other_params)
        decay = dict(
            {"params": decay_params},
            **other_params,
            weight_decay=weight_decay,
        )
        return optim.SGD([decay, no_decay])


def write_lr(scheduler: Any, writer: SummaryWriter, step: int):
    """Utility to write learning rate(s) to tensorboard."""
    lr = scheduler.get_last_lr()
    if isinstance(lr, list):
        for ix, x in enumerate(lr):
            writer.add_scalar(
                f"LearningRate_{ix}", x, step,
            )
    else:
        writer.add_scalar(
            f"LearningRate_{ix}", x, step,
        )


def show(img: torch.Tensor):
    """Small utility to plot a tensort of img"""
    img = img.detach()
    try:
        img = F.to_pil_image(img)
    except ValueError:  # handle batched images to plot
        img = make_grid(img)
        img = F.to_pil_image(img)
    plt.imshow(np.asarray(img))
    plt.xticks([])  # remove pyplot borders
    plt.yticks([])
    plt.show()
    plt.close()
