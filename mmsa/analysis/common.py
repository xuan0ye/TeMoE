"""Shared helpers for mmsa.analysis scripts.

Every analysis script (expert_routing, modality_attribution, qualitative_case)
needs to rebuild one specific ablation variant's model architecture and load
its already-trained checkpoint -- no retraining, pure post-hoc inspection.
Centralizing that logic keeps the three scripts consistent with each other
and with mmsa.training.ablation's own variant definitions.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import torch

from mmsa.data.mmsa_pkl_dataset import MMSASequenceDataset, load_mmsa_splits
from mmsa.models.builder import build_model
from mmsa.training.ablation import PLAN_B_VARIANTS, VARIANTS, _apply_patch
from mmsa.training.train import load_config


# Plan B added channel-path parameters (face/content/bg projections, the
# background gate, and the consistency projection) that are constructed
# unconditionally in TeMoE.__init__. A checkpoint trained before that rework
# therefore cannot contain them. When the rebuilt variant runs the
# single-vector path (use_channels=False) those layers are never read, so
# leaving them at random init is harmless -- but any *other* mismatch means the
# checkpoint and the architecture genuinely disagree and must fail loudly.
_CHANNEL_ONLY_PREFIXES = ("face_proj.", "content_proj.", "bg_proj.", "bg_gate_net.", "cons_proj.")


def _load_state(model: torch.nn.Module, state: dict[str, torch.Tensor], ckpt: Path) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    tolerated = sorted(key for key in missing if key.startswith(_CHANNEL_ONLY_PREFIXES))
    real_missing = sorted(set(missing) - set(tolerated))
    if real_missing or unexpected:
        raise RuntimeError(
            f"checkpoint {ckpt} does not match the rebuilt architecture: "
            f"missing={real_missing} unexpected={sorted(unexpected)}"
        )
    if tolerated:
        print(
            f"[info] {ckpt.name}: {len(tolerated)} Plan-B channel parameter(s) absent from this "
            "pre-Plan-B checkpoint and left at random init (use_channels=False, so they are never read)"
        )


def load_variant(    config_path: str,
    variant_name: str,
    ablation_dir: str,
    seed: int,
    device: torch.device,
    ckpt_variant: str | None = None,
) -> tuple[torch.nn.Module, MMSASequenceDataset, bool]:
    """Rebuild `variant_name` (per mmsa.training.ablation.VARIANTS) and load its
    seed_{seed} checkpoint from `ablation_dir`. Returns (model, test_dataset,
    use_interpretation).

    `ckpt_variant` decouples the architecture/config used to rebuild the model
    from the checkpoint sub-directory name. This matters here because Plan B
    variants live in a separate table (``PLAN_B_VARIANTS``, selected by
    ``ablation --plan-b``) while the pre-Plan-B single-vector checkpoints are
    still on disk under the classic variant names: pass
    ``variant_name="single_channel"`` (single-vector interpretation path, which
    matches the pre-Plan-B architecture) together with ``ckpt_variant="full"``
    to analyse a legacy checkpoint with the correct weight shapes."""
    base_cfg = load_config(config_path)
    # Plan B variants are declared in their own table (ablation --plan-b); both
    # tables are merged here so analysis scripts can address any variant that
    # exists on disk, whichever table it was trained from.
    variant_patches = {**dict(VARIANTS), **dict(PLAN_B_VARIANTS)}
    if variant_name not in variant_patches:
        raise ValueError(f"Unknown ablation variant '{variant_name}'; known: {sorted(variant_patches)}")
    config = _apply_patch(copy.deepcopy(base_cfg), variant_patches[variant_name], variant_name)
    disabled = set(config.get("disabled_modalities", []) or [])

    interp_dir = base_cfg["data"].get("interpretation_dir") or os.environ.get("TEMOE_INTERP_DIR")
    test_set = load_mmsa_splits(
        base_cfg["data"]["pkl"], interpretation_dir=interp_dir, disabled_modalities=disabled
    )["test"]
    dims = test_set.dims
    use_interp = bool(config["model"].get("use_interpretation", True)) and dims["interp_dim"] > 0
    model = build_model("temoe", dims, config["model"], use_interp).to(device)

    ckpt = Path(ablation_dir) / (ckpt_variant or variant_name) / f"seed_{seed}" / "temoe_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint for variant '{variant_name}' not found at {ckpt}. "
            "Run mmsa.training.ablation with this seed first (the ablation sweep "
            "saves a checkpoint per variant/seed as a side effect of training)."
        )
    state = torch.load(ckpt, map_location="cpu")
    _load_state(model, state, ckpt)
    model.eval()
    return model, test_set, use_interp


def load_baseline_arch(
    config_path: str,
    arch: str,
    multiseed_dir: str,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, MMSASequenceDataset, bool]:
    """Same as `load_variant` but for a plain baseline arch trained via
    mmsa.training.multiseed (late_fusion / cross_modal / lmf / almt / kuda /
    misa / selfmm / magbert / cormult), which has no disabled_modalities and
    no temporal/MoE patch."""
    base_cfg = load_config(config_path)
    config: dict[str, Any] = copy.deepcopy(base_cfg)
    config["model"]["arch"] = arch

    interp_dir = base_cfg["data"].get("interpretation_dir") or os.environ.get("TEMOE_INTERP_DIR")
    test_set = load_mmsa_splits(base_cfg["data"]["pkl"], interpretation_dir=interp_dir, disabled_modalities=set())[
        "test"
    ]
    dims = test_set.dims
    use_interp = bool(config["model"].get("use_interpretation", True)) and dims["interp_dim"] > 0
    model = build_model(arch, dims, config["model"], use_interp).to(device)

    ckpt = Path(multiseed_dir) / arch / f"seed_{seed}" / "temoe_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint for arch '{arch}' not found at {ckpt}. "
            "Run mmsa.training.multiseed --arch {arch} with this seed first."
        )
    state = torch.load(ckpt, map_location="cpu")
    _load_state(model, state, ckpt)
    model.eval()
    return model, test_set, use_interp
