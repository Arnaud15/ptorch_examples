from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def conv_bn(
    in_channels: int, out_channels: int, kernel_size: int, *args, **kwargs
):
    """2D convolution followed by BatchNorm, used everywhere in the ResNet."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs),
        nn.BatchNorm2d(out_channels),
    )


class ResNet(nn.Module):
    def __init__(
        self,
        img_channels: int,
        n_classes: int,
        extra_blocks_per_layer: List[int],
        resnet_channels: List[int],
        stem_channels: int,
        slender_stem: bool,
        better_downsampling: bool,
        resblock_kernel_size: int = 3,
        stem_conv_size: int = 7,
        stem_pool_size: int = 3,
        stem_downsample: bool = True,
        expansion: int = 4,
    ):
        super(ResNet, self).__init__()
        assert len(extra_blocks_per_layer) > 0
        assert len(extra_blocks_per_layer) == len(resnet_channels)
        self.stem = ResnetStem(
            slender_stem=slender_stem,
            in_channels=img_channels,
            out_channels=stem_channels,
            conv_size=stem_conv_size,
            pool_size=stem_pool_size,
            downsample=stem_downsample,
        )
        layers = []
        in_channels = stem_channels
        for (n_blocks, channel_size) in zip(
            extra_blocks_per_layer, resnet_channels
        ):
            layers.append(
                ResnetLayer(
                    n_extra_blocks=n_blocks,
                    expansion=expansion,
                    in_channels=in_channels,
                    out_channels=channel_size,
                    kernel_size=resblock_kernel_size,
                    better_downsampling=better_downsampling,
                )
            )
            in_channels = channel_size * expansion
        self.layers = nn.ModuleList(layers)
        self.head = ResnetHead(in_channels=in_channels, n_classes=n_classes)
        self.apply(xavier_init)
        self.zero_out_blocks()

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)
        return x

    def get_params(self):
        """Return to parameter groups: with and without weight decay"""
        no_decay = []
        with_decay = []
        for name, params in self.named_parameters():
            if "bias" in name:
                no_decay.append(params)
            else:
                with_decay.append(params)
        assert len(list(self.parameters())) == len(no_decay) + len(with_decay)
        return no_decay, with_decay

    def zero_out_blocks(self):
        """Zero out blocks by setting the scale of batch norm layers at the end of each residual block to 0."""
        for res_layer in self.layers:
            res_layer.input_block.expand.apply(zero_scale_init_bn)
            for block in res_layer.later_blocks:
                block.expand.apply(zero_scale_init_bn)


def xavier_init(m):
    """Use the uniform Xavier initialization for linear and convolution layers."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def zero_scale_init_bn(m):
    """Useful to have skip through connections"""
    if isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.weight)


class ResnetHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super(ResnetHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.projection_head = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.projection_head(x)
        return x


class ResnetStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_size: int,
        pool_size: int,
        downsample: bool,
        slender_stem: bool,
    ):
        super(ResnetStem, self).__init__()
        if slender_stem:
            # typically convs 3x3
            self.conv_layer = nn.Sequential(
                conv_bn(
                    in_channels,
                    out_channels,
                    kernel_size=conv_size,
                    stride=2 if downsample else 1,
                    padding=conv_size // 2,
                ),
                *[
                    conv_bn(
                        out_channels,
                        out_channels,
                        kernel_size=conv_size,
                        stride=1,
                        padding=conv_size // 2,
                    )
                    for _ in range(2)
                ],
            )
        else:
            # typically a conv 7x7
            self.conv_layer = conv_bn(
                in_channels,
                out_channels,
                kernel_size=conv_size,
                stride=2 if downsample else 1,
                padding=conv_size // 2,
            )
        self.max_pool = nn.MaxPool2d(
            kernel_size=pool_size,
            stride=2 if downsample else 1,
            padding=pool_size // 2,
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
        better_downsampling: bool,
        kernel_size: int = 3,
    ):
        super(ResnetLayer, self).__init__()
        downsample = in_channels != out_channels
        self.input_block = BottleneckProjection(
            expansion=expansion,
            in_channels=in_channels,
            out_channels=out_channels,
            downsample=downsample,
            kernel_size=kernel_size,
            better_downsampling=better_downsampling,
        )
        self.later_blocks = nn.ModuleList(
            [
                BottleneckPlain(
                    expansion=expansion,
                    in_channels=out_channels * expansion,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
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
        better_downsampling: bool,
        kernel_size: int = 3,
    ):
        super(BottleneckProjection, self).__init__()
        self.expansion = expansion
        if downsample:
            if better_downsampling:
                self.projection = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                    conv_bn(
                        in_channels,
                        out_channels * expansion,
                        kernel_size=1,
                        stride=1,
                    ),
                )
                compress_stride = 1
                conv_stride = 2
            else:
                self.projection = conv_bn(
                    in_channels,
                    out_channels * expansion,
                    kernel_size=1,
                    stride=2,
                )
                compress_stride = 2
                conv_stride = 1
        else:
            self.projection = conv_bn(
                in_channels, out_channels * expansion, kernel_size=1, stride=1,
            )
            compress_stride = 1
            conv_stride = 1
        self.compress = conv_bn(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=compress_stride,
            padding=0,
        )
        self.conv = conv_bn(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            padding=kernel_size // 2,
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
    def __init__(
        self,
        expansion: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super(BottleneckPlain, self).__init__()
        assert in_channels == out_channels * expansion
        self.expansion = expansion
        self.compress = conv_bn(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv = conv_bn(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
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
