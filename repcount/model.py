"""A small causal temporal convolutional network (TCN) over pose features.

For every frame it predicts, from that frame and the ones before it (the receptive field: 63 frames
with the default dilations, about 4 seconds at 15 fps):
    active  - is a repetition in progress (logit)
    phase   - where in the repetition we are, as (cos, sin) of an angle that goes once around per rep
    action  - which exercise (logits)
Counting turns the phase into completed laps (see count.py).
"""

from __future__ import annotations

import torch
from torch import nn

from .data import CLASSES
from .features import FEATURES

DILATIONS = (1, 2, 4, 8, 16)
KERNEL = 3


def receptive_field(dilations=DILATIONS) -> int:
    return 1 + (KERNEL - 1) * sum(dilations)


class CausalConv(nn.Conv1d):
    def __init__(self, channels: int, dilation: int):
        super().__init__(channels, channels, KERNEL, dilation=dilation)
        self.left = (KERNEL - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(nn.functional.pad(x, (self.left, 0)))


class Block(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = CausalConv(channels, dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.conv(torch.relu(x)))
        y = self.mix(torch.relu(y))
        return x + self.drop(y)


class RepTCN(nn.Module):
    def __init__(self, channels: int = 64, dropout: float = 0.1, dilations=DILATIONS):
        super().__init__()
        self.dilations = tuple(dilations)
        self.inp = nn.Conv1d(FEATURES, channels, 1)
        self.blocks = nn.Sequential(*[Block(channels, d, dropout) for d in self.dilations])
        self.head = nn.Conv1d(channels, 3 + len(CLASSES), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, FEATURES, frames] -> [batch, 3 + classes, frames]: active, cos, sin, action logits."""
        return self.head(torch.relu(self.blocks(self.inp(x))))


def split_outputs(y: torch.Tensor):
    active = y[:, 0]
    phase = y[:, 1:3]
    action = y[:, 3:]
    return active, phase, action
