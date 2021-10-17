from typing import List

import torch
import torch.nn as nn


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
