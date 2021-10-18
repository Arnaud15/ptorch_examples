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


def conv_bn(in_channels: int, out_channels: int, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels),
    )


class Resnet(nn.Module):
    def __init__(
        self,
        img_channels: int,
        n_classes: int,
        extra_blocks_per_layer: List[int],
        resnet_channels: int = 64,
        stem_conv_size: int = 7,
        stem_pool_size: int = 3,
        expansion: int = 4,
    ):
        super(Resnet, self).__init__()
        self.stem = ResnetStem(
            in_channels=img_channels,
            out_channels=resnet_channels,
            conv_size=stem_conv_size,
            pool_size=stem_pool_size,
        )
        layers = []
        in_channels = resnet_channels
        for n_blocks in extra_blocks_per_layer:
            layers.append(
                ResnetLayer(
                    n_extra_blocks=n_blocks,
                    expansion=expansion,
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                )
            )
            in_channels *= 2
        self.layers = nn.ModuleList(layers)
        self.head = ResnetHead(in_channels=in_channels, n_classes=n_classes)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)
        return x


class ResnetHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super(ResnetHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.projection_head = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.projection_head(x)
        return x


class ResnetStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_size: int,
        pool_size: int,
    ):
        super(ResnetStem, self).__init__()
        self.conv_layer = conv_bn(
            in_channels,
            out_channels,
            kernel_size=conv_size,
            stride=2,
            padding=conv_size // 2,
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size=pool_size, stride=2, padding=pool_size // 2
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.max_pool(x)
        return x


class ResnetLayer(nn.Module):
    def __init__(
        self,
        n_extra_blocks: int,
        expansion: int,
        in_channels: int,
        out_channels: int,
    ):
        super(ResnetLayer, self).__init__()
        assert in_channels <= out_channels
        downsample = in_channels < out_channels
        self.input_block = BottleneckProjection(
            expansion=expansion,
            in_channels=in_channels,
            out_channels=out_channels,
            downsample=downsample,
        )
        self.later_blocks = nn.ModuleList(
            [
                BottleneckPlain(
                    expansion=expansion,
                    in_channels=out_channels * expansion,
                    out_channels=out_channels,
                )
                for _ in range(n_extra_blocks)
            ]
        )

    def forward(self, x):
        x = self.input_block(x)
        for block in self.later_blocks:
            x = block(x)
        return x


class BottleneckProjection(nn.Module):
    def __init__(
        self,
        expansion: int,
        in_channels: int,
        out_channels: int,
        downsample: bool,
    ):
        super(BottleneckProjection, self).__init__()
        self.expansion = expansion
        self.projection = conv_bn(in_channels, out_channels * expansion)
        self.compress = conv_bn(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2 if downsample else 1,
            padding=0,
        )
        self.conv = conv_bn(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.expand = conv_bn(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        residual = x
        x = self.compress(x)
        x = self.conv(x)
        x = self.expand(x)
        x += self.projection(residual)
        x = F.relu(x)
        return x


class BottleneckPlain(nn.Module):
    def __init__(self, expansion: int, in_channels: int, out_channels: int):
        super(BottleneckPlain, self).__init__()
        assert in_channels == out_channels * expansion
        self.expansion = expansion
        self.compress = conv_bn(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv = conv_bn(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.expand = conv_bn(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        residual = x
        x = self.compress(x)
        x = self.conv(x)
        x = self.expand(x)
        x += residual
        x = F.relu(x)
        return x
