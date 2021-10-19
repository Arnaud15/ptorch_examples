import os
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import update_ewma

# TODO move in constants file
SAVE_PATH = os.path.join("data", "checkpoints")
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

Scheduler = (Any,)


def training_loop(
    model: nn.Module,
    opt: optim.Optimizer,
    scheduler: Scheduler,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    loss_fn: Callable,
    device: Any,
    n_epochs: int,
    print_every: int = 0,
    write_every: int = 0,
    check_every: int = 0,
    smoothing_alpha: float = 0.9,
    metric_fn: Optional[Callable] = None,
):
    assert n_epochs > 0, f"got a nonpositive number of epochs {n_epochs}"
    assert (
        smoothing_alpha > 0.0 and smoothing_alpha <= 1.0
    ), f"got smoothing alpha={smoothing_alpha}"
    if print_every:
        assert print_every > 0
    if write_every:
        assert write_every > 0
    if check_every:
        assert check_every > 0
    writer = SummaryWriter()

    train_loss = None
    eval_loss = None
    train_steps = 0
    eval_steps = 0
    for epoch_ix in range(n_epochs):
        model.train()
        for (x, y) in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss_item = loss_fn(y_hat, y)
            opt.zero_grad()
            loss_item.backward()
            opt.step()
            scheduler.step()

            train_loss = update_ewma(
                obs=loss_item, prev=train_loss, alpha=smoothing_alpha
            )
            train_steps += 1

            metric_item = metric_fn(y_hat, y) if metric_fn is not None else 1.0
            if print_every and train_steps % print_every == 0:
                print(f"Step: {train_steps} | Training Loss: {loss_item:.5f}")
                print(
                    f"Step: {train_steps} | Training Metric: {metric_item:.5f}"
                )
            if write_every and train_steps % write_every == 0:
                writer.add_scalar("Loss/train", loss_item, train_steps)
                writer.add_scalar("Metric/train", metric_item, train_steps)

        model.eval()
        with torch.no_grad():
            for (x, y) in eval_loader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss_item = loss_fn(y_hat, y)

                eval_loss = update_ewma(
                    obs=loss_item, prev=eval_loss, alpha=smoothing_alpha
                )
                eval_steps += 1
                metric_item = (
                    metric_fn(y_hat, y) if metric_fn is not None else 1.0
                )
                if print_every and eval_steps % print_every == 0:
                    print(
                        f"Step: {eval_steps} | Validation Loss: {loss_item:.5f}"
                    )
                    print(
                        f"Step: {eval_steps} | Validation Metric: {metric_item:.5f}"
                    )
                if write_every and eval_steps % write_every == 0:
                    writer.add_scalar("Loss/eval", loss_item, eval_steps)
                    writer.add_scalar("Metric/eval", metric_item, eval_steps)
        if check_every and (epoch_ix + 1) % check_every == 0:
            torch.save(
                {
                    "epoch": epoch_ix + 1,
                    "model_state": model.state_dict(),
                    "optim_state": opt.state_dict(),
                    "eval_loss": eval_loss,
                },
                SAVE_PATH,
            )
