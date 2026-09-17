from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class _Expert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextGuidedSparseMoE(nn.Module):
    """Text-centric sparse Mixture-of-Experts fusion.

    Modality summaries (aligned audio-visual, text, MLLM interpretation) are
    concatenated and projected into a fused token. A router **conditioned on the
    text context** picks the top-k experts per sample (sparse compute -> the
    second source of the acceleration claim). A Switch-style load-balancing
    auxiliary loss keeps expert utilization even.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden: int | None = None,
        dropout: float = 0.1,
        n_inputs: int = 3,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")
        expert_hidden = expert_hidden or d_model * 2
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_proj = nn.Linear(n_inputs * d_model, d_model)
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(_Expert(d_model, expert_hidden, dropout) for _ in range(num_experts))
        self.norm = nn.LayerNorm(d_model)
        # Populated on every forward call (detached) so analysis scripts
        # (mmsa.analysis.expert_routing) can inspect per-sample routing
        # decisions without threading extra return values through every
        # caller (TeMoE.forward, baselines, train/eval loops).
        self.last_topk_idx: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None
        self.last_probs: torch.Tensor | None = None

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
        text_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.input_proj(torch.cat(list(inputs), dim=-1))

        gate_logits = self.router(text_context)
        topk_val, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        gate = torch.softmax(topk_val, dim=-1)

        probs = torch.softmax(gate_logits, dim=-1)
        importance = probs.mean(dim=0)
        dispatch = torch.zeros_like(gate_logits).scatter(1, topk_idx, 1.0)
        load = dispatch.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(importance * load)

        self.last_topk_idx = topk_idx.detach()
        self.last_gate = gate.detach()
        self.last_probs = probs.detach()

        output = torch.zeros_like(fused)
        for slot in range(self.top_k):
            slot_expert = topk_idx[:, slot]
            slot_weight = gate[:, slot].unsqueeze(-1)
            for expert_id in range(self.num_experts):
                selected = slot_expert == expert_id
                if selected.any():
                    output[selected] = output[selected] + slot_weight[selected] * self.experts[expert_id](fused[selected])

        return self.norm(fused + output), aux_loss
