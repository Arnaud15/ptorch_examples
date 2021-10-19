from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils import show


def test_loop(
    test_loader: DataLoader,
    model: nn.Module,
    device: Any,
    metric_fn: Callable,
    plot: bool = True,
    loss_fn: Optional[Callable] = None,
):
    model.eval()
    metric_total = 0.0
    loss_total = 0.0 if loss_fn is not None else None
    steps = 0
    with torch.no_grad():
        for (x, y) in test_loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            metric_total += metric_fn(y_hat, y)

            if loss_total is not None:
                loss_total += loss_fn(y_hat, y)

            steps += 1
    metric_total /= steps
    if loss_total is not None:
        loss_total /= steps
        print(
            f"Testing completed, Metric: {metric_total:.5f}, Loss: {loss_total:.5f}"
        )
    else:
        print(f"Testing completed, Metric: {metric_total:.5f}")

    if plot:
        (x, y) = next(iter(test_loader))
        show(make_grid(x))
        y_hat = torch.max(model(x.to(device)), 1)[1]
        print(f"Grid generated with labels: {y.detach().numpy()}")
        print(f"Corresponding predictions: {y_hat.detach().cpu().numpy()}")
