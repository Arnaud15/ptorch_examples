import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ContrastiveLearner(nn.Module):
    def __init__(self, encoder: nn.Module, projection: nn.Module):
        super(ContrastiveLearner, self).__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z


def encode_simclr(x, model, transform):
    t = transform(x)
    h, z = model(t)
    return t, h, z


def info_nce(z1, z2, temp):
    assert z1.size() == z2.size()
    assert z2.dim() == 2
    n = z1.size(0)
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    features = torch.cat([z1, z2])
    sims = torch.matmul(features, torch.t(features))
    sims = sims / temp
    mask = torch.eye(2 * n, dtype=torch.bool)
    negative = sims[~mask].view(2 * n, 2 * n - 1)
    idxs = torch.cat([(torch.arange(n) + n).unsqueeze(1), torch.arange(n).unsqueeze(1)])
    positive = torch.gather(sims, 1, idxs).squeeze(1)
    assert positive.size() == (2 * n,)
    out = torch.logsumexp(negative, dim=1) - positive
    return torch.mean(out)


def step(x, model, transform):
    t1, h1, z1 = encode_simclr(x, model, transform)
    t2, h2, z2 = encode_simclr(x, model, transform)
    loss = info_nce(z2, z2, temp=1.0)
    return torch.cat([t1, t2, z1]), loss
