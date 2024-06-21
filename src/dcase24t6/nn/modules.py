#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from dcase24t6.nn.functional import drop_path

_DATA_FORMATS = ("channels_last", "channels_first")


class CustomLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        if data_format not in _DATA_FORMATS:
            raise ValueError(
                f"Invalid argument {data_format=}. (expected one of {_DATA_FORMATS})"
            )

        super().__init__()
        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            raise ValueError(f"Invalid argument {self.data_format=}.")


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class PositionalEncoding(nn.Module):
    # BASED ON PYTORCH TUTORIAL : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(
        self,
        emb_size: int,
        dropout_p: float,
        maxlen: int = 5000,
    ) -> None:
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer("pos_embedding", pos_embedding)
        self.pos_embedding: Tensor

    def forward(self, token_embedding: Tensor) -> Tensor:
        pos_embedding_value = self.pos_embedding[: token_embedding.size(0), :]
        output = self.dropout(token_embedding + pos_embedding_value)
        return output


class MultipleProjections(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_projections: int,
        loss_scale: float = 3e-6,
    ) -> None:
        super().__init__()

        self.num_projections = num_projections
        self.out_features = out_features
        self.loss_scale = loss_scale

        self.projection_layers = nn.ModuleList(
            [
                nn.Linear(in_features=in_features, out_features=out_features)
                for _ in range(num_projections)
            ]
        )

        self.router = nn.Linear(in_features=in_features, out_features=num_projections)

    def compute_loss(self, layer: torch.Tensor) -> torch.Tensor:
        a = torch.zeros(self.num_projections)
        for i in range(self.num_projections):
            a[i] = (layer == i).sum()
        variation = a - a.mean()
        loss = (variation**2).mean()
        return loss * self.loss_scale

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq, d_model)
        """
        assert x.ndim == 3

        B, C, D = x.shape

        x = x.reshape(B * C, -1)
        layer = self.router(x).argmax(dim=1)
        loss = self.compute_loss(layer=layer)

        output = torch.zeros(B * C, self.out_features, device="cuda")
        for i in range(self.num_projections):
            output[layer == i] = self.projection_layers[i](x[layer == i])

        output = output.reshape(B, C, -1)
        return output, loss


# Testing
if __name__ == "__main__":
    a = torch.rand(64, 22, 256, device="cuda")
    layer = MultipleProjections(256, 512, num_projections=4).to("cuda")
    output, loss = layer(a)
    print(loss)
    pass
