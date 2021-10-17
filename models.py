from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, layer_dims: List[int]):
        assert (
            len(layer_dims) >= 2
        ), f"at least input and output dims, got {layer_dims}"
        super(MLP, self).__init__()
        self.layer_dims = layer_dims
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(layer_dims[ix - 1], layer_dims[ix])
                for ix in range(1, len(layer_dims))
            ]
        )
        self.n_layers = len(layer_dims) - 1

    def forward(self, x):
        if len(x.shape) > 1:
            x = torch.flatten(x, start_dim=1)
        else:
            x = torch.flatten(x)
        assert x.shape[-1] == self.layer_dims[0], (x.shape, self.layer_dims)
        for layer_ix, layer in enumerate(self.linear_layers):
            x = layer(x)
            if layer_ix < self.n_layers - 1:
                x = nn.ReLU()(x)
        return x


class BottleneckProjection(nn.Module):
    def __init__(self, expansion: int, in_channels: int, out_channels: int, downsample: bool):
        super(BottleneckProjection, self).__init__()
        self.expansion = expansion
        self.projection = nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=2 if downsample else 1, padding=0)
        self.bn0 = nn.BatchNorm2d(out_channels * self.expansion)
        self.compress = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.expand = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x):
        residual = x
        x = self.compress(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.expand(x)
        x = self.bn3(x)
        x += self.bn0(self.projection(residual))
        x = F.relu(x)
        return x


class BottleneckPlain(nn.Module):
    def __init__(self, expansion: int, in_channels: int, out_channels: int, downsample: bool):
        super(BottleneckPlain, self).__init__()
        assert out_channels * expansion == in_channels
        self.expansion = expansion
        self.compress = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.expand = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x):
        residual = x
        x = self.compress(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        x = self.expand(x)
        x = self.bn3(x)
        x += residual
        x = F.relu(x)
        return x

