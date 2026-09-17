"""Dependency-light PatchTST-style encoder with a custom volatility head.

This is a clean research implementation of the architectural pattern: overlapping patches are
projected into tokens, encoded with a Transformer, pooled, and mapped to normalized log-average
variance targets. It does not load or copy pretrained weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PatchTSTStyleConfig:
    context_length: int
    patch_length: int
    patch_stride: int
    num_hidden_layers: int
    d_model: int
    num_attention_heads: int
    ffn_dim: int
    dropout: float
    num_targets: int

    def validate(self) -> None:
        if self.context_length < self.patch_length:
            raise ValueError("context_length must be at least patch_length")
        if self.patch_length <= 0 or self.patch_stride <= 0:
            raise ValueError("patch dimensions must be positive")
        if self.d_model % self.num_attention_heads != 0:
            raise ValueError("d_model must be divisible by num_attention_heads")
        if min(self.num_hidden_layers, self.ffn_dim, self.num_targets) <= 0:
            raise ValueError("Layer count, feed-forward width, and target count must be positive")


def _sinusoidal_encoding(length: int, width: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width)
    )
    encoding = torch.zeros(length, width, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * divisor)
    encoding[:, 1::2] = torch.cos(positions * divisor[: encoding[:, 1::2].shape[1]])
    return encoding


class PatchTSTStyleRegressor(nn.Module):
    """Univariate patch Transformer for direct multi-horizon regression."""

    def __init__(self, config: PatchTSTStyleConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        patch_count = 1 + (config.context_length - config.patch_length) // config.patch_stride
        self.patch_projection = nn.Linear(config.patch_length, config.d_model)
        self.register_buffer(
            "positional_encoding",
            _sinusoidal_encoding(patch_count, config.d_model),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_attention_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_hidden_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_targets),
        )

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        if contexts.ndim == 3 and contexts.shape[-1] == 1:
            contexts = contexts.squeeze(-1)
        if contexts.ndim != 2 or contexts.shape[1] != self.config.context_length:
            raise ValueError(
                f"Expected contexts shaped [batch, {self.config.context_length}], got "
                f"{tuple(contexts.shape)}"
            )
        patches = contexts.unfold(
            dimension=1,
            size=self.config.patch_length,
            step=self.config.patch_stride,
        )
        tokens = self.patch_projection(patches)
        tokens = tokens + self.positional_encoding[: tokens.shape[1]].unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.head(encoded.mean(dim=1))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
