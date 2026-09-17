from __future__ import annotations

from typing import Any

from torch import nn

from mmsa.models.baselines import (
    ALMTBaseline,
    CorMulTBaseline,
    CrossModalTransformer,
    KuDABaseline,
    LateFusionBaseline,
    LMFBaseline,
    MAGBertBaseline,
    MISABaseline,
    SelfMMBaseline,
)
from mmsa.models.model import TeMoE

ARCHITECTURES = (
    "temoe",
    "late_fusion",
    "cross_modal",
    "lmf",
    "almt",
    "kuda",
    "misa",
    "selfmm",
    "magbert",
    "cormult",
)


def build_model(
    arch: str,
    dims: dict[str, int],
    model_cfg: dict[str, Any],
    use_interpretation: bool,
) -> nn.Module:
    # Plan B: per-channel clue dims. The dataset exposes interp_face_dim etc.
    # when a structured cache is present; legacy caches fall back to the
    # single interp_dim for all channels (harmless: channel path is only
    # taken when the batch actually carries channel tensors).
    channel_dim = dims.get("interp_face_dim") or dims.get("interp_dim") or 0
    # A structured (Plan B) cache exposes per-channel dims; a legacy single-vector
    # cache does not.  Taking the channel path without channels is silently
    # destructive: model.forward ignores `interpretation` there and fills the
    # channels with zeros, so the model sees only constant per-batch biases.  That
    # is how every MOSEI explanation run before this fix came out identical to its
    # no-explanation twin, while the baselines (which have no channel path) gained
    # 0.036--0.124 MAE as expected.
    has_channels = bool(
        dims.get("interp_face_dim") or dims.get("interp_content_dim") or dims.get("interp_bg_dim")
    )
    common = dict(
        audio_dim=dims["audio_dim"],
        vision_dim=dims["vision_dim"],
        text_dim=dims["text_dim"],
        interp_dim=channel_dim or dims["interp_dim"],
        d_model=int(model_cfg.get("d_model", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_interpretation=use_interpretation,
    )
    arch = (arch or "temoe").lower()
    if arch == "temoe":
        return TeMoE(
            **common,
            n_heads=int(model_cfg.get("n_heads", 4)),
            num_experts=int(model_cfg.get("num_experts", 8)),
            top_k=int(model_cfg.get("top_k", 2)),
            use_mamba=bool(model_cfg.get("use_mamba", True)),
            temporal_variant=str(model_cfg.get("temporal_variant", "hybrid")),
            use_channels=bool(model_cfg.get("use_channels", True)) and has_channels,
            use_face=bool(model_cfg.get("use_face", True)),
            use_bg_gate=bool(model_cfg.get("use_bg_gate", True)),
            use_consistency=bool(model_cfg.get("use_consistency", True)),
            cons_loss_weight=float(model_cfg.get("cons_loss_weight", 0.1)),
        )
    if arch == "late_fusion":
        return LateFusionBaseline(**common)
    if arch == "cross_modal":
        return CrossModalTransformer(**common, n_heads=int(model_cfg.get("n_heads", 4)))
    if arch == "lmf":
        return LMFBaseline(**common, rank=int(model_cfg.get("rank", 4)))
    if arch == "almt":
        return ALMTBaseline(
            **common,
            n_heads=int(model_cfg.get("n_heads", 4)),
            num_hyper_tokens=int(model_cfg.get("num_hyper_tokens", 4)),
        )
    if arch == "kuda":
        return KuDABaseline(**common)
    if arch == "misa":
        return MISABaseline(**common)
    if arch == "selfmm":
        return SelfMMBaseline(**common)
    if arch == "magbert":
        return MAGBertBaseline(**common, n_heads=int(model_cfg.get("n_heads", 4)))
    if arch == "cormult":
        return CorMulTBaseline(**common, n_heads=int(model_cfg.get("n_heads", 4)))
    raise ValueError(f"Unknown arch '{arch}'. Expected one of {ARCHITECTURES}")
