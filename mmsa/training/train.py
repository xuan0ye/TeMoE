from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from mmsa.data.mmsa_pkl_dataset import load_mmsa_splits
from mmsa.models.builder import build_model
from mmsa.training.metrics import msa_regression_metrics


def train(config: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    disabled = set(config.get("disabled_modalities", []) or [])
    # env override lets orchestration scripts point at a cache without editing the yaml
    interp_dir = data_cfg.get("interpretation_dir") or os.environ.get("TEMOE_INTERP_DIR")
    splits = load_mmsa_splits(data_cfg["pkl"], interpretation_dir=interp_dir, disabled_modalities=disabled)
    if "train" not in splits or "test" not in splits:
        raise ValueError("pkl must contain at least train and test splits")
    valid = splits.get("valid", splits["test"])

    dims = splits["train"].dims
    use_interpretation = bool(model_cfg.get("use_interpretation", True)) and dims["interp_dim"] > 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch = str(model_cfg.get("arch", "temoe"))
    model = build_model(arch, dims, model_cfg, use_interpretation).to(device)

    train_loader = DataLoader(splits["train"], batch_size=int(train_cfg.get("batch_size", 32)), shuffle=True)
    valid_loader = DataLoader(valid, batch_size=int(train_cfg.get("eval_batch_size", 64)))
    test_loader = DataLoader(splits["test"], batch_size=int(train_cfg.get("eval_batch_size", 64)))

    # TEMOE_LR lets a single architecture be retrained at a different learning rate
    # without editing the shared config -- used to recover architectures that
    # diverge at the config default (ALMT/CorMulT on CH-SIMS).
    lr_override = os.environ.get("TEMOE_LR")
    if lr_override:
        train_cfg = {**train_cfg, "learning_rate": float(lr_override)}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    reg_loss_fn = nn.L1Loss()
    aux_weight = float(train_cfg.get("aux_loss_weight", 0.01))
    cons_weight = float(model_cfg.get("cons_loss_weight", 0.1))
    label_range = tuple(train_cfg.get("label_range", [-3.0, 3.0]))
    epochs = int(train_cfg.get("epochs", 40))
    patience = int(train_cfg.get("patience", 8))

    output = Path(train_cfg.get("output_dir", "outputs/mmsa"))
    output.mkdir(parents=True, exist_ok=True)

    best_valid_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_valid_report: dict[str, float] = {}
    epochs_no_improve = 0

    nonfinite_batches = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = _forward(model, batch, device, use_interpretation)
            loss = reg_loss_fn(outputs["reg"], batch["label"].to(device))
            loss = loss + aux_weight * outputs["aux_loss"]
            if "cons_loss" in outputs and cons_weight > 0:
                loss = loss + cons_weight * outputs["cons_loss"]
            if not torch.isfinite(loss):
                # A non-finite loss must not be back-propagated: without this guard a
                # single divergent batch turns every weight into NaN and the run only
                # fails later inside evaluate() (observed for ALMT/CorMulT on CH-SIMS,
                # whose unaligned variable-length sequences are numerically harder).
                nonfinite_batches += 1
                print(
                    f"[warn] non-finite loss ({nonfinite_batches} so far this run); skipping batch -- "
                    "if this keeps climbing the architecture is diverging on this dataset and needs a smaller lr"
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            running += float(loss.item())

        # per-epoch hook (Self-MM re-estimates its pseudo-label blend weights here)
        hook = getattr(model, "on_epoch_end", None)
        if callable(hook):
            hook()

        valid_report = evaluate(model, valid_loader, device, use_interpretation, label_range)
        print({"epoch": epoch, "train_loss": round(running / max(1, len(train_loader)), 4), **_round(valid_report)})

        if valid_report["mae"] < best_valid_mae - 1e-5:
            best_valid_mae = valid_report["mae"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_valid_report = valid_report
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, output / "temoe_model.pt")

    test_report = evaluate(model, test_loader, device, use_interpretation, label_range)
    report = {
        "dataset": data_cfg.get("name", Path(data_cfg["pkl"]).stem),
        "arch": arch,
        "temporal_backend": getattr(model, "temporal_backend", arch),
        "use_interpretation": use_interpretation,
        "use_channels": bool(getattr(model, "use_channels", False)),
        "disabled_modalities": sorted(disabled),
        "params": sum(p.numel() for p in model.parameters()),
        "best_valid": best_valid_report,
        "test": test_report,
    }
    with (output / "temoe_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({"test": _round(test_report)}, ensure_ascii=False))
    return report


def _forward(model: nn.Module, batch: dict, device: torch.device, use_interpretation: bool) -> dict:
    interpretation = batch.get("interpretation")
    if use_interpretation and interpretation is not None:
        interpretation = interpretation.to(device)
    else:
        interpretation = None
    # Plan B: per-channel clue tensors -- ONLY forwarded to the TeMoE channel
    # path (model exposes `use_channels`). Baselines (LateFusion/LMF/MulT/...)
    # have the legacy signature (audio, vision, text, interpretation=None) and
    # must never receive the channel kwargs, or they raise TypeError.
    channel_kwargs: dict[str, torch.Tensor] = {}
    if getattr(model, "use_channels", False):
        for key in ("interp_face", "interp_content", "interp_bg", "interp_bg_flag"):
            value = batch.get(key)
            if value is not None:
                channel_kwargs[key] = value.to(device)
    # Self-MM builds its unimodal pseudo-labels from the ground-truth label, so it is
    # the one baseline that must see `y`; every other architecture stays label-free.
    if getattr(model, "needs_labels", False) and "label" in batch:
        channel_kwargs["labels"] = batch["label"].to(device)
    return model(
        audio=batch["audio"].to(device),
        vision=batch["vision"].to(device),
        text=batch["text"].to(device),
        interpretation=interpretation,
        **channel_kwargs,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_interpretation: bool,
    label_range: tuple[float, float],
) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    nonfinite_preds = 0
    label_mid = float(np.mean(label_range)) if label_range else 0.0
    for batch in loader:
        outputs = _forward(model, batch, device, use_interpretation)
        pred = outputs["reg"].cpu().numpy()
        bad = ~np.isfinite(pred)
        if bad.any():
            nonfinite_preds += int(bad.sum())
            # a divergent sample must not crash the whole run (msa_regression_metrics
            # feeds the predictions to sklearn, which raises on NaN); substitute the
            # label midpoint so the sample is scored as neutral rather than aborting.
            pred = np.where(bad, label_mid, pred)
        preds.append(pred)
        trues.append(batch["label"].numpy())
    metrics = msa_regression_metrics(np.concatenate(preds), np.concatenate(trues), label_range=label_range)
    if nonfinite_preds:
        # surfaced in the run report so a diverged run can never be mistaken for a
        # merely weak one: any non-zero value here makes the metrics unusable for
        # comparison (the substituted predictions are not real model outputs).
        metrics["nonfinite_preds"] = float(nonfinite_preds)
        print(
            f"[warn] {nonfinite_preds} non-finite predictions replaced with the label midpoint -- "
            "these metrics are NOT usable for comparison; treat this run as diverged"
        )
    return metrics


def _round(report: dict[str, float]) -> dict[str, float]:
    return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in report.items()}


def load_config(path: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if overrides:
        config.update(overrides)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TeMoE multimodal sentiment model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--arch",
        default=None,
        help="temoe | late_fusion | cross_modal | lmf | almt | kuda | misa | selfmm | magbert | cormult",
    )
    parser.add_argument("--disable", nargs="*", default=None, help="ablation: modalities to zero out")
    parser.add_argument("--no-mamba", action="store_true")
    parser.add_argument("--no-interpretation", action="store_true")
    parser.add_argument(
        "--temporal-variant",
        default=None,
        choices=["hybrid", "gated_conv"],
        help="temoe only: 'hybrid' (Mamba+cross-attention, ours) or "
        "'gated_conv' (TEXT/Rao et al. 2026-style gated Conv1d comparison block)",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.arch:
        config["model"]["arch"] = args.arch
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.disable is not None:
        config["disabled_modalities"] = args.disable
    if args.no_mamba:
        config["model"]["use_mamba"] = False
    if args.no_interpretation:
        config["model"]["use_interpretation"] = False
    if args.temporal_variant is not None:
        config["model"]["temporal_variant"] = args.temporal_variant
    if args.top_k is not None:
        config["model"]["top_k"] = args.top_k
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    train(config)


if __name__ == "__main__":
    main()
