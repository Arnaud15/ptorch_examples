import os
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.constants import DATA_DIR
from src.utils import update_ewma, write_lr


def training_loop(
    name: str,  # identifier for the current run
    model: nn.Module,
    opt: optim.Optimizer,
    scheduler: Any,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    loss_fn: Callable,
    device: Any,
    n_epochs: int,
    print_every: int = 0,
    write_every: int = 0,
    plot_every: int = 0,
    check_every: int = 0,
    smoothing_alpha: float = 0.9,
    metric_fn: Optional[Callable] = None,
):
    assert n_epochs > 0, f"got a nonpositive number of epochs {n_epochs}"
    assert (
        smoothing_alpha > 0.0 and smoothing_alpha <= 1.0
    ), f"got smoothing alpha={smoothing_alpha}"
    assert print_every >= 0
    assert write_every >= 0
    assert plot_every >= 0
    assert check_every >= 0
    if write_every or plot_every:
        logs_path = os.path.join(DATA_DIR, "logs")
        if not os.path.isdir(logs_path):
            os.mkdir(logs_path)
        run_path = os.path.join(logs_path, name)
        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        writer = SummaryWriter(log_dir=run_path)
    if check_every:
        save_path = os.path.join(DATA_DIR, "checkpoints")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    train_loss = None
    eval_loss = None
    train_steps = 0
    eval_steps = 0
    for epoch_ix in range(n_epochs):
        print(f"Start of epoch {epoch_ix + 1}")
        model.train()
        for (x, y) in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss_item = loss_fn(y_hat, y)
            opt.zero_grad()
            loss_item.backward()
            opt.step()

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

            if plot_every and train_steps % plot_every == 0:
                writer.add_images("Plots/train", x[:15], train_steps)

            if check_every and train_steps % check_every == 0:
                torch.save(
                    {
                        "epoch": epoch_ix + 1,
                        "model_state": model.state_dict(),
                        "optim_state": opt.state_dict(),
                        "eval_loss": eval_loss,
                    },
                    os.path.join(save_path, f"{name}-{train_steps}.pt"),
                )

        # one step per epoch
        if scheduler is not None:
            scheduler.step()
            if write_every:
                write_lr(scheduler, writer, epoch_ix + 1)

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
            if write_every:
                writer.add_scalar("Loss/eval", loss_item, eval_steps)
                writer.add_scalar("Metric/eval", metric_item, eval_steps)

            if print_every:
                print(f"Step: {eval_steps} | Validation Loss: {loss_item:.5f}")
                print(
                    f"Step: {eval_steps} | Validation Metric: {metric_item:.5f}"
                )
            if plot_every:
                writer.add_images("Plots/eval", x[:15], eval_steps)
    return model, opt
