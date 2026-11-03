"""MS-TCN (Multi-Stage Temporal Convolutional Network, Farha & Gall, CVPR 2019) cho phân đoạn bước theo frame.

Stage 1: đặc trưng (D) → logits (C) qua L lớp conv giãn có residual (dilation 1, 2, 4, …, 2^(L−1)).
Stage 2..S: tinh chỉnh từ softmax của stage trước. Loss = Σ_stage [CE + λ · MSE(Δ log-prob theo thời gian) bị chặn τ²],
thành phần làm mượt giảm phân mảnh đoạn (over-segmentation). Cần torch (CPU đủ chạy; GPU nhanh hơn nhiều).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidual(nn.Module):
    def __init__(self, dilation: int, ch: int, dropout: float):
        super().__init__()
        self.conv_dilated = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(ch, ch, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.conv_1x1(F.relu(self.conv_dilated(x))))
        return (x + out) * mask


class SingleStage(nn.Module):
    def __init__(self, layers: int, ch: int, dim_in: int, n_classes: int, dropout: float):
        super().__init__()
        self.conv_in = nn.Conv1d(dim_in, ch, 1)
        self.layers = nn.ModuleList([DilatedResidual(2 ** i, ch, dropout) for i in range(layers)])
        self.conv_out = nn.Conv1d(ch, n_classes, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out, mask)
        return self.conv_out(out) * mask


class MSTCN(nn.Module):
    """x: (B, D, T), mask: (B, 1, T) → logits (S, B, C, T)."""

    def __init__(self, dim_in: int, n_classes: int, stages: int = 4, layers: int = 10, channels: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        self.stage1 = SingleStage(layers, channels, dim_in, n_classes, dropout)
        self.stages = nn.ModuleList([SingleStage(layers, channels, n_classes, n_classes, dropout)
                                     for _ in range(stages - 1)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.stage1(x, mask)
        outs = [out]
        for st in self.stages:
            out = st(F.softmax(out, dim=1) * mask, mask)
            outs.append(out)
        return torch.stack(outs, dim=0)


def mstcn_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, lam: float = 0.15,
               tau: float = 4.0, ignore_index: int = -100) -> torch.Tensor:
    """logits (S, B, C, T), y (B, T) với ignore_index cho frame không nhãn/đệm, mask (B, 1, T)."""
    loss = logits.new_zeros(())
    C = logits.shape[2]
    for p in logits:
        loss = loss + F.cross_entropy(p.transpose(2, 1).reshape(-1, C), y.reshape(-1), ignore_index=ignore_index)
        if p.shape[-1] > 1:
            d = F.mse_loss(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1),
                           reduction="none")
            m = mask[:, :, 1:]
            loss = loss + lam * (torch.clamp(d, 0, tau ** 2) * m).sum() / (m.sum() * C).clamp(min=1)
    return loss
