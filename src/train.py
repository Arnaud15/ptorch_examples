"""Training loop for image classification models."""

import os
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.args import TrainingArgs
from src.constants import DATA_DIR
from src.utils import mixup, set_learning_rate, update_ewma, write_lr


def training_loop(
    name: str,  # identifier for the training run
    args: TrainingArgs,
    model: nn.Module,
    opt: optim.Optimizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    loss_fn: Callable,
    device: Any,
    metric_fn: Optional[Callable] = None,
):
    print(f"Training starts for {name}")
    if args.warmup_epochs:
        set_learning_rate(opt, args.learning_rate / args.warmup_epochs)
        warm_scheduler = optim.lr_scheduler.LambdaLR(
            opt, lambda epoch_ix: (epoch_ix + 1)
        )  # 0 indexed epochs
        inner_train(
            name=name,
            n_epochs=args.num_epochs,
            args=args,
            model=model,
            opt=opt,
            scheduler=warm_scheduler,
            train_loader=train_loader,
            eval_loader=eval_loader,
            loss_fn=loss_fn,
            metric_fn=metric_fn,
            device=device,
        )
    if not args.cosine_lr:
        base_scheduler = optim.lr_scheduler.StepLR(
            opt, step_size=args.decay_interval, gamma=args.decay_gamma
        )
    else:
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, args.num_epochs, 0
        )
    print("Post-warmup begins")
    set_learning_rate(opt, args.learning_rate)
    inner_train(
        name=name,
        n_epochs=args.num_epochs,
        args=args,
        model=model,
        opt=opt,
        scheduler=base_scheduler,
        train_loader=train_loader,
        eval_loader=eval_loader,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        device=device,
    )


def inner_train(
    name: str,  # identifier for the training run
    n_epochs: int,
    args: TrainingArgs,
    model: nn.Module,
    opt: optim.Optimizer,
    scheduler: Any,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    loss_fn: Callable,
    device: Any,
    metric_fn: Optional[Callable] = None,
):
    assert n_epochs > 0, f"got a nonpositive number of epochs {n_epochs}"
    assert (
        args.smoothing_alpha > 0.0 and args.smoothing_alpha <= 1.0
    ), f"got smoothing alpha={args.smoothing_alpha}"
    assert args.print_every >= 0
    assert args.write_every >= 0
    assert args.plot_every >= 0
    assert args.check_every >= 0
    if args.mixup_alpha is not None:
        assert args.mixup_alpha > 0
    if args.write_every or args.plot_every:
        logs_path = os.path.join(DATA_DIR, "logs")
        if not os.path.isdir(logs_path):
            os.mkdir(logs_path)
        run_path = os.path.join(logs_path, name)
        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        writer = SummaryWriter(log_dir=run_path)
    if args.check_every:
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
            y_0 = y
            if args.mixup_alpha is not None:
                x_0 = x
                (x, y) = mixup(x, y, args.mixup_alpha, args.num_classes)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss = update_ewma(
                obs=loss.item(), prev=train_loss, alpha=args.smoothing_alpha
            )
            train_steps += 1

            metric = (
                metric_fn(y_hat, y_0).item() if metric_fn is not None else 1.0
            )
            if args.print_every and train_steps % args.print_every == 0:
                print(
                    f"Step: {train_steps} | Training Loss: {loss.item():.5f}"
                )
                print(f"Step: {train_steps} | Training Metric: {metric:.5f}")
            if args.write_every and train_steps % args.write_every == 0:
                writer.add_scalar("Loss/train", loss.item(), train_steps)
                writer.add_scalar("Metric/train", metric, train_steps)

            if args.plot_every and train_steps % args.plot_every == 0:
                writer.add_images("Plots/train", x[:15], train_steps)
                if args.mixup_alpha is not None:
                    writer.add_images("Plots/train_mix", x_0[:15], train_steps)

            if args.check_every and train_steps % args.check_every == 0:
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
            if args.write_every:
                write_lr(scheduler, writer, epoch_ix + 1)

        model.eval()
        with torch.no_grad():
            for (x, y) in eval_loader:
                x = x.to(device)
                y = y.to(device)
                y_0 = y
                if args.mixup_alpha is not None:
                    x_0 = x
                    (x, y) = mixup(x, y, args.mixup_alpha, args.num_classes)
                y_hat = model(x)
                loss = loss_fn(y_hat, y)

                eval_loss = update_ewma(
                    obs=loss.item(), prev=eval_loss, alpha=args.smoothing_alpha
                )
                eval_steps += 1
                metric = (
                    metric_fn(y_hat, y_0).item()
                    if metric_fn is not None
                    else 1.0
                )
            if args.write_every:
                writer.add_scalar("Loss/eval", loss.item(), eval_steps)
                writer.add_scalar("Metric/eval", metric, eval_steps)

            if args.print_every:
                print(
                    f"Step: {eval_steps} | Validation Loss: {loss.item():.5f}"
                )
                print(f"Step: {eval_steps} | Validation Metric: {metric:.5f}")
            if args.plot_every:
                writer.add_images("Plots/eval", x[:15], eval_steps)
                if args.mixup_alpha is not None:
                    writer.add_images("Plots/eval_mix", x_0[:15], train_steps)
    return model, opt
