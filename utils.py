from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F


def update_ewma(obs: float, prev: Optional[float], alpha: float) -> float:
    if prev is None:
        return obs
    else:
        return alpha * obs + (1.0 - alpha) * prev


def accuracy(preds, targets):
    return (torch.max(preds, 1)[1] == targets).float().mean()


plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
