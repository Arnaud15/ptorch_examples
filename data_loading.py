from typing import Callable, Optional, Tuple

import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

DATA_DIR = "data"


def get_image_data_loader(
    name: str,
    train: bool,
    val_share: float,
    batch_size: int,
    shuffle: bool = True,
    single_batch: bool = False,
    transform: Optional[Callable] = T.ToTensor(),
) -> Tuple[DataLoader, DataLoader]:
    """Get a cv dataset"""
    assert batch_size > 0, f"received incorrect batch size: {batch_size}."
    if train:
        assert (
            val_share >= 0.0 and val_share < 1.0
        ), f"The validation set should map to a share s in [0, 1.0) of the training data, received {val_share}"
    # Get datasets
    if name == "mnist":
        dataset = MNIST(
            root=DATA_DIR, train=train, transform=transform, download=True
        )
    elif name == "fmnist":
        dataset = FashionMNIST(
            root=DATA_DIR, train=train, transform=transform, download=True
        )
    elif name == "cifar10":
        dataset = CIFAR10(
            root=DATA_DIR, train=train, transform=transform, download=True
        )
    else:
        raise ValueError("supported datasets are 'mnist', 'fmnist', 'cifar10'")
    if single_batch:
        train_dataset = Subset(dataset, indices=list(range(batch_size)))
        val_dataset = train_dataset
    else:
        n = len(dataset)
        n_train = int(n * (1.0 - val_share)) if train else n
        n_val = n - n_train
        if n_val:
            train_dataset, val_dataset = random_split(
                dataset, [n_train, n_val]
            )
        else:
            val_dataset = None
            train_dataset = dataset
    # Get dataloaders and return
    print(
        f"Dataset lengths: train-{len(train_dataset)}, val-{0 if val_dataset is None else len(val_dataset)}"
    )
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            prefetch_factor=2,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            prefetch_factor=2,
        )
        if val_dataset is not None
        else val_dataset,
    )
