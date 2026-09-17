from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mmsa.models.sequence_backbones import SinusoidalPositionalEncoding, TemporalSSM
from mmsa.models.temporal_align import GatedConvTemporalAlignment, TemporalAlignment
from mmsa.models.text_moe import TextGuidedSparseMoE


class TeMoE(nn.Module):
    """MLLM-Augmented Temporal-Aligned Mixture-of-Experts for MSA.

    Pipeline:
    1. Per-modality projection + temporal positional encoding.
    2. TemporalAlignment: Mamba + time cross-attention aligns audio/vision.
    3. Text branch encoded by its own SSM; text summary acts as the anchor.
    4. TextGuidedSparseMoE fuses aligned AV + text + MLLM interpretation.
    5. Regression head (sentiment intensity) + optional classification head.

    Plan B (person-centric clue extraction): when `interp_channels` is used,
    the MLLM interpretation arrives as three separate clue channels --
    face (speaker expression), content (what is happening), and bg
    (background). The background channel is gated by a learned scalar
    (``bg_gate``) conditioned on text+face+content, so background cues are
    suppressed by default and admitted only when informative. A
    cross-modal consistency module computes cosine similarities between the
    text context and the content/face channels and returns a consistency
    auxiliary loss; the consistency vector is fused into the MoE token.
    """

    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        text_dim: int,
        interp_dim: int = 0,
        channel_dim: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        use_mamba: bool = True,
        num_classes: int = 1,
        use_interpretation: bool = True,
        temporal_variant: str = "hybrid",
        # Plan B knobs (from model config):
        use_channels: bool = True,
        use_face: bool = True,
        use_bg_gate: bool = True,
        use_consistency: bool = True,
        cons_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

        self.temporal_variant = str(temporal_variant or "hybrid").lower()
        if self.temporal_variant == "gated_conv":
            self.temporal_align: nn.Module = GatedConvTemporalAlignment(d_model, dropout)
        else:
            self.temporal_align = TemporalAlignment(d_model, n_heads, dropout, use_mamba)
        self.text_ssm = TemporalSSM(d_model, dropout, use_mamba)

        self.use_interpretation = bool(use_interpretation and interp_dim > 0)
        self._interp_dim = interp_dim
        # Plan B channel dim: dimension of each structured clue embedding.
        # Legacy caches have a single vector (interp_dim) and no channels.
        self._channel_dim = channel_dim or interp_dim
        self.use_channels = bool(use_channels)
        self.use_face = bool(use_face)
        self.use_bg_gate = bool(use_bg_gate)
        self.use_consistency = bool(use_consistency)
        self.cons_loss_weight = float(cons_loss_weight)

        # Plan B: per-channel projections (used when the dataset provides
        # interp_face / interp_content / interp_bg / interp_bg_flag).
        self.face_proj = nn.Linear(self._channel_dim, d_model)
        self.content_proj = nn.Linear(self._channel_dim, d_model)
        self.bg_proj = nn.Linear(self._channel_dim, d_model)
        if self.use_interpretation:
            self.interp_proj = nn.Sequential(nn.Linear(interp_dim, d_model), nn.GELU())
            n_inputs = 3
        else:
            self.interp_proj = None
            n_inputs = 2

        # Plan B: background gate -- scalar in (0,1) conditioned on
        # text context + face/content/bg clues; background is suppressed by
        # default (gate near 0) and admitted only when informative.
        if self.use_bg_gate and self.use_interpretation:
            self.bg_gate_net = nn.Sequential(
                nn.Linear(4 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.bg_gate_net = None

        # Plan B: cross-modal consistency -- maps the 3 cosine similarities
        # (text-content, face-text, face-content) into the fusion token.
        # Only meaningful on the channel path; the legacy single-vector path
        # has no separate channels to compare, so keep n_inputs consistent.
        if self.use_channels and self.use_consistency and self.use_interpretation:
            self.cons_proj = nn.Sequential(nn.Linear(3, d_model), nn.GELU())
            n_inputs += 1
        else:
            self.cons_proj = None

        self.moe = TextGuidedSparseMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            n_inputs=n_inputs,
        )
        self.reg_head = nn.Linear(d_model, 1)
        self.cls_head = nn.Linear(d_model, num_classes) if num_classes > 1 else None
        # populated on forward for analysis: per-sample bg gate value and
        # consistency similarities
        self.last_bg_gate: torch.Tensor | None = None
        self.last_cons_sims: torch.Tensor | None = None

    def forward(
        self,
        audio: torch.Tensor,
        vision: torch.Tensor,
        text: torch.Tensor,
        interpretation: torch.Tensor | None = None,
        interp_face: torch.Tensor | None = None,
        interp_content: torch.Tensor | None = None,
        interp_bg: torch.Tensor | None = None,
        interp_bg_flag: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        audio_h = self.pos_encoding(self.audio_proj(audio))
        vision_h = self.pos_encoding(self.vision_proj(vision))
        text_h = self.pos_encoding(self.text_proj(text))

        av_summary = self.temporal_align(audio_h, vision_h)

        text_seq = self.text_ssm(text_h)
        text_context = text_seq.mean(dim=1)

        moe_inputs = [av_summary, text_context]
        cons_loss = torch.zeros((), device=audio.device, dtype=av_summary.dtype)
        if self.use_interpretation:
            if self.use_channels:
                # Plan B path: separate clue channels. Missing channels are
                # zero-filled so the channel-path input count stays fixed.
                batch = audio.size(0)
                device = audio.device
                dtype = av_summary.dtype
                face = (interp_face if interp_face is not None else torch.zeros(
                    batch, self._channel_dim, device=device, dtype=dtype
                )).to(device)
                content = (interp_content if interp_content is not None else torch.zeros(
                    batch, self._channel_dim, device=device, dtype=dtype
                )).to(device)
                bg = (interp_bg if interp_bg is not None else torch.zeros(
                    batch, self._channel_dim, device=device, dtype=dtype
                )).to(device)

                e_face = self.face_proj(face) if self.use_face else torch.zeros_like(
                    self.face_proj(face)
                )
                e_content = self.content_proj(content)
                e_bg = self.bg_proj(bg)

                # Background gate: scalar in (0,1); background suppressed by default.
                bg_gate = torch.zeros(batch, 1, device=device, dtype=dtype)
                if self.bg_gate_net is not None:
                    bg_gate_input = torch.cat([text_context, e_face, e_content, e_bg], dim=-1)
                    bg_gate = torch.sigmoid(self.bg_gate_net(bg_gate_input))
                e_gated = e_face + e_content + bg_gate * e_bg
                self.last_bg_gate = bg_gate.detach()

                # Cross-modal consistency: cosine similarities among text,
                # content, and face clues (face terms degenerate to the
                # text-content similarity when the face channel is dropped).
                # The consistency vector joins the MoE token; the (negative)
                # mean similarity is a cheap auxiliary loss that pulls
                # agreeing channels together.
                if self.use_face:
                    cons_sims = torch.stack(
                        [
                            F.cosine_similarity(text_context, e_content, dim=-1),
                            F.cosine_similarity(e_face, text_context, dim=-1),
                            F.cosine_similarity(e_face, e_content, dim=-1),
                        ],
                        dim=-1,
                    )
                else:
                    sim_tc = F.cosine_similarity(text_context, e_content, dim=-1)
                    cons_sims = torch.stack([sim_tc, sim_tc, sim_tc], dim=-1)
                self.last_cons_sims = cons_sims.detach()
                if self.cons_proj is not None:
                    moe_inputs.append(self.cons_proj(cons_sims))
                if self.use_consistency:
                    cons_loss = self.cons_loss_weight * (1.0 - cons_sims.mean())
                # e_gated is already d_model-dimensional (sum of channel
                # projections); feed it directly to the MoE.
                moe_inputs.append(e_gated)
            else:
                # Legacy single-vector path (use_channels=False).
                if interpretation is None:
                    interpretation = torch.zeros(
                        audio.size(0), self._interp_dim, device=audio.device, dtype=av_summary.dtype
                    )
                moe_inputs.append(self.interp_proj(interpretation))

        fused, aux_loss = self.moe(moe_inputs, text_context)

        outputs: dict[str, torch.Tensor] = {
            "reg": self.reg_head(fused).squeeze(-1),
            "aux_loss": aux_loss,
            "cons_loss": cons_loss,
            "embedding": fused,
        }
        if self.cls_head is not None:
            outputs["cls"] = self.cls_head(fused)
        return outputs

    @property
    def temporal_backend(self) -> str:
        return self.temporal_align.backend
