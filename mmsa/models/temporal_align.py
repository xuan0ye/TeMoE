from __future__ import annotations

import torch
from torch import nn

from mmsa.models.sequence_backbones import TemporalSSM


class TimeCrossAttention(nn.Module):
    """Time-based cross-attention: one modality queries another along the time axis."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attended, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return query + attended


class TemporalAlignment(nn.Module):
    """Time-oriented module that aligns audio and vision representations.

    Combines the strengths of Mamba (efficient temporal modeling) and
    time-based cross-attention (fine-grained inter-modal alignment):
    1. Each modality passes through its own Mamba/SSM encoder.
    2. Bidirectional time cross-attention (audio<->vision) aligns them.
    3. Each aligned stream is time-pooled and fused into one audio-visual vector.

    Pooling before fusion keeps the module valid for both aligned datasets (equal
    audio/vision length) and unaligned datasets (different lengths, e.g. SIMS).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_mamba: bool = True,
    ) -> None:
        super().__init__()
        self.audio_ssm = TemporalSSM(d_model, dropout, use_mamba)
        self.vision_ssm = TemporalSSM(d_model, dropout, use_mamba)
        self.audio_to_vision = TimeCrossAttention(d_model, n_heads, dropout)
        self.vision_to_audio = TimeCrossAttention(d_model, n_heads, dropout)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        audio: torch.Tensor,
        vision: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        audio_seq = self.audio_ssm(audio)
        vision_seq = self.vision_ssm(vision)
        audio_aligned = self.audio_to_vision(audio_seq, vision_seq, key_padding_mask=vision_mask)
        vision_aligned = self.vision_to_audio(vision_seq, audio_seq, key_padding_mask=audio_mask)
        audio_vec = audio_aligned.mean(dim=1)
        vision_vec = vision_aligned.mean(dim=1)
        return self.fuse(torch.cat([audio_vec, vision_vec], dim=-1))

    @property
    def backend(self) -> str:
        return self.audio_ssm.backend


class GatedConvTemporalAlignment(nn.Module):
    """Lightweight gated-Conv1d temporal alignment (Rao et al. 2026, "TEXT").

    Used as a comparison/ablation variant for our Mamba+cross-attention hybrid:
    TEXT reports that replacing this block with plain Mamba or plain temporal
    cross-attention *hurts* their model, i.e. it claims to outperform both of the
    ingredients our hybrid combines. Re-implemented here (not the original
    checkpoint/code) so we can test that claim directly against our own hybrid
    on the same data/training pipeline, rather than only citing their numbers.

    For each modality pair (audio a, vision v), one direction is:
        left = a + Linear(Conv1d(LN(a)) elementwise* SiLU(LN(v)))
    and symmetrically for `right` (roles of a/v swapped). The two gated streams
    are time-pooled and concatenated, matching the output contract of
    ``TemporalAlignment`` (a single [B, d_model] audio-visual summary).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, kernel_size: int = 3) -> None:
        super().__init__()
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.conv_a = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2)
        self.gate_act = nn.SiLU()
        self.lin_a = nn.Linear(d_model, d_model)
        self.lin_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        audio: torch.Tensor,
        vision: torch.Tensor,
        audio_mask: torch.Tensor | None = None,  # unused: matches TemporalAlignment's signature
        vision_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = min(audio.size(1), vision.size(1))
        a, v = audio[:, :t], vision[:, :t]
        a_n, v_n = self.norm_a(a), self.norm_v(v)
        a_conv = self.conv_a(a_n.transpose(1, 2)).transpose(1, 2)
        v_conv = self.conv_v(v_n.transpose(1, 2)).transpose(1, 2)
        left = a + self.dropout(self.lin_a(a_conv * self.gate_act(v_n)))
        right = v + self.dropout(self.lin_v(v_conv * self.gate_act(a_n)))
        pooled = torch.cat([left.mean(dim=1), right.mean(dim=1)], dim=-1)
        return self.fuse(pooled)

    @property
    def backend(self) -> str:
        return "gated-conv (TEXT-style)"
