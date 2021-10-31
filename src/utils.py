from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def update_ewma(obs: float, prev: Optional[float], alpha: float) -> float:
    if prev is None:
        return obs
    else:
        return alpha * obs + (1.0 - alpha) * prev


def accuracy(preds, targets):
    return (torch.max(preds, 1)[1] == targets).float().mean()


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
        raise NotImplementedError("coming up soon")
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
