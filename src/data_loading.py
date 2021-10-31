from typing import Callable, Optional, Set, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, ImageNet

from src.constants import DATA_DIR


def subset_classes(dataset: Dataset, classes_retained: Set[int]) -> Dataset:
    """Returns a subset of the data, where observations all come
    from one of the `classes_retained`. 

    All integers in `classes_retained` must therefore be in
    [0, num classes).

    Assumes a map-style dataset (implements __getitem__) with
    tuple observations (x, y), y an integer representing x's class label."""
    n = len(dataset)
    indices_retained = [
        ix for ix in range(n) if dataset[ix][1] in classes_retained
    ]
    return Subset(dataset, indices_retained)


def transforms_image_net(
    is_image: bool = True,
    crop: bool = True,
    crop_size: Optional[int] = None,
    flip: bool = True,
    colors: bool = True,
    standardize: bool = True,
    means: Optional[Tuple[float]] = None,
    stds: Optional[Tuple[float]] = None,
) -> Callable:
    """ImageNet image transforms."""
    transforms = []
    if crop:
        transforms.append(
            T.RandomResizedCrop(
                size=224 if crop_size is None else crop_size,
                scale=(0.08, 1.0),
                ratio=(3 / 4, 4 / 3),
            )
        )
    if flip:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    if colors:
        transforms.append(
            T.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4)
        )
    if standardize:
        means = (123.68, 116.779, 103.939) if means is None else means
        stds = (58.393, 51.12, 57.375) if stds is None else stds
        transforms.append(T.Normalize(means, stds,))
    # TODO: PCA colors Augment
    if is_image:
        transforms.append(T.ToTensor())
    return T.Compose(transforms)


def get_image_data_loader(
    name: str,
    train: bool,
    val_share: float,
    batch_size: int,
    shuffle: bool = True,
    single_batch: bool = False,
    transform: Optional[Callable] = T.ToTensor(),
) -> Tuple[DataLoader, Optional[DataLoader]]:
    assert batch_size > 0, f"received incorrect batch size: {batch_size}."
    if train:
        assert (
            val_share >= 0.0 and val_share < 1.0
        ), f"The validation set should map to a share s in [0, 1.0) of the training data, received {val_share}"
    data = get_image_dataset(name=name, train=train, transform=transform)
    return get_data_loader(
        dataset=data,
        val_share=val_share,
        batch_size=batch_size,
        shuffle=shuffle,
        single_batch=single_batch,
    )


def get_image_dataset(
    name: str, train: bool, transform: Optional[Callable] = T.ToTensor(),
) -> Dataset:
    """Get a CV dataset, select the train or test split and apply desired transforms."""
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
    elif name == "imagenet":
        dataset = ImageNet(
            root=DATA_DIR,
            train=train,
            transform=transforms_image_net()
            if transform is None
            else transform,
            download=True,
        )
    else:
        raise ValueError("supported datasets are 'mnist', 'fmnist', 'cifar10'")
    return dataset


def get_data_loader(
    dataset: Dataset,
    val_share: float,
    batch_size: int,
    shuffle: bool = True,
    single_batch: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Get a dataloader from an input dataset."""
    if single_batch:
        train_dataset = Subset(dataset, indices=list(range(batch_size)))
        val_dataset = train_dataset
    else:
        n = len(dataset)
        n_train = int(n * (1.0 - val_share))
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
