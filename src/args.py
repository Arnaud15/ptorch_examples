"""Frozen dataclasses for experiment parameters."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ResNetExpArgs:
    """ResNet experiment"""

    dataset_name: str
    batch_size: int
    learning_rate: float = 0.1
    num_epochs: int = 120
    momentum: float = 0.9
    weight_decay: float = 0.0
    cosine_lr: bool = False
    warmup_epochs: int = 0
    decay_interval: int = 30
    decay_gamma: float = 0.1
    mixup_alpha: Optional[float] = None
    lean_stem: bool = False
    smart_downsampling: bool = False
    use_gpu: bool = True


@dataclass(frozen=True)
class TrainingArgs:
    """Training loop arguments for an image classification task"""

    batch_size: int
    num_classes: int
    num_epochs: int = 120
    cosine_lr: bool = False
    warmup_epochs: int = 0
    decay_interval: int = 30
    decay_gamma: float = 0.1
    mixup_alpha: Optional[float] = None
    print_every: int = 0
    write_every: int = 0
    plot_every: int = 0
    check_every: int = 0
    smoothing_alpha: float = 0.9


def to_exp_name(args) -> str:
    """Utility to get a pretty string name from args."""
    return "_".join(str(val) for val in vars(args).values())
