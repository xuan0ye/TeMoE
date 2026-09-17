from __future__ import annotations

import math

import torch
from torch import nn

try:  # mamba_ssm needs a CUDA build; keep it optional so the pipeline always runs.
    from mamba_ssm import Mamba  # type: ignore

    _HAS_MAMBA = True
except Exception:  # pragma: no cover - depends on runtime environment
    Mamba = None  # type: ignore
    _HAS_MAMBA = False

if _HAS_MAMBA:
    # mamba_ssm's Triton kernels touch the CUDA driver context in a way that can
    # break cuDNN's RNN handle creation later in the *same process* (observed as
    # "CUDNN_STATUS_NOT_INITIALIZED" the moment a GRU fallback module is moved to
    # cuda). The GRU fallback below is a small model with no need for cuDNN's
    # fused kernel, so disable cuDNN globally rather than fight the interaction.
    torch.backends.cudnn.enabled = False


def has_mamba() -> bool:
    """Whether the real selective-scan Mamba kernel is importable."""
    return _HAS_MAMBA


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalSSM(nn.Module):
    """Linear-time temporal encoder.

    Uses the selective-scan ``Mamba`` block when available (linear complexity in
    sequence length, the source of the inference-acceleration claim). Falls back
    to a bidirectional GRU so the model still trains anywhere ``mamba_ssm`` is not
    installed, keeping experiments reproducible on any A800/CPU environment.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, use_mamba: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._use_mamba = bool(use_mamba and _HAS_MAMBA)
        if self._use_mamba:
            self.core = Mamba(d_model=d_model)
            self.proj: nn.Module = nn.Identity()
        else:
            self.core = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.norm(x)
        if self._use_mamba:
            hidden = self.core(hidden)
        else:
            hidden, _ = self.core(hidden)
            hidden = self.proj(hidden)
        return residual + self.dropout(hidden)

    @property
    def backend(self) -> str:
        return "mamba" if self._use_mamba else "bi-gru-fallback"
