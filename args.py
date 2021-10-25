"""Frozen classes for experiment parameters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResNetTrainingArgs:
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
    mixup_alpha: float = 0.5
    mixup_enabled: bool = True
    lean_stem: bool = False
    smart_downsampling: bool = False
    use_gpu: bool = True


def to_exp_name(args: ResNetTrainingArgs) -> str:
    """Util to get experiment names."""
    return str(args).replace(" ", "")

