r"""Integrated-Gradients-based per-modality attribution for a single test clip.

This complements the modality-\emph{ablation} study
(\S~sec:ablation -- whole-dataset, retrain-per-variant, 5-seed) with a second,
independent, per-\emph{sample} line of evidence: for one fixed trained
checkpoint, how much does the model's own gradient attribute the regression
output to each modality's raw input (and the cached MLLM interpretation
vector)? We use Integrated Gradients (Sundararajan et al. 2017, "Axiomatic
Attribution for Deep Networks") with a *zero* baseline -- deliberately the
same "zeroed-out modality" convention already used throughout
mmsa/data/mmsa_pkl_dataset.py's `disabled_modalities` ablations, so the two
analyses (retrain-based ablation vs.\ gradient-based attribution) are
directly comparable rather than using two different notions of "absence".

No captum dependency: the trapezoidal Integrated Gradients estimator is
~30 lines of plain autograd (see `integrated_gradients` below), which keeps
this analysis reproducible with only the same requirements.txt already used
for training.

Usage (reuses the sample already picked by mmsa.analysis.qualitative_case,
or pass --sample-index directly):

    python -m mmsa.analysis.modality_attribution \
        --config mmsa/configs/mosi.yaml \
        --ablation-dir outputs/mmsa/mosi/ablation --seed 42 \
        --from-case-study outputs/mmsa/mosi/analysis/qualitative_case.json \
        --output outputs/mmsa/mosi/analysis/modality_attribution.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mmsa.analysis.common import load_variant

_MODALITY_KEYS = ("audio", "vision", "text", "interpretation")


def integrated_gradients(
    model: torch.nn.Module,
    item: dict[str, torch.Tensor],
    use_interp: bool,
    device: torch.device,
    steps: int = 32,
) -> dict[str, float]:
    """Zero-baseline Integrated Gradients attribution, one modality at a time.

    Returns each modality's share of total |attribution| (sums to ~1.0), so
    the four numbers are directly comparable to each other regardless of the
    very different raw scales of audio/vision/text/interpretation features.
    """
    inputs: dict[str, torch.Tensor | None] = {
        "audio": item["audio"].unsqueeze(0).to(device),
        "vision": item["vision"].unsqueeze(0).to(device),
        "text": item["text"].unsqueeze(0).to(device),
    }
    interp = item.get("interpretation")
    inputs["interpretation"] = interp.unsqueeze(0).to(device) if (use_interp and interp is not None) else None

    baselines = {name: (torch.zeros_like(tensor) if tensor is not None else None) for name, tensor in inputs.items()}
    grad_accum = {name: (torch.zeros_like(tensor) if tensor is not None else None) for name, tensor in inputs.items()}

    for step in range(1, steps + 1):
        alpha = step / steps
        scaled: dict[str, torch.Tensor | None] = {}
        for name, tensor in inputs.items():
            if tensor is None:
                scaled[name] = None
                continue
            interpolated = baselines[name] + alpha * (tensor - baselines[name])
            scaled[name] = interpolated.detach().clone().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        outputs = model(
            audio=scaled["audio"], vision=scaled["vision"], text=scaled["text"], interpretation=scaled["interpretation"]
        )
        outputs["reg"].sum().backward()

        for name, tensor in scaled.items():
            if tensor is None or tensor.grad is None:
                continue
            grad_accum[name] = grad_accum[name] + tensor.grad.detach()

    magnitude: dict[str, float] = {}
    for name, tensor in inputs.items():
        if tensor is None:
            magnitude[name] = 0.0
            continue
        avg_grad = grad_accum[name] / steps
        ig = (tensor - baselines[name]) * avg_grad
        magnitude[name] = float(ig.abs().sum().item())

    total = sum(magnitude.values()) or 1.0
    return {name: round(value / total, 4) for name, value in magnitude.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated-Gradients modality attribution for one test clip.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation-dir", required=True)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument(
        "--from-case-study",
        default=None,
        help="read sample_index from a mmsa.analysis.qualitative_case JSON output, for a consistent example",
    )
    parser.add_argument("--steps", type=int, default=32, help="Integrated-Gradients interpolation steps")
    parser.add_argument("--output", default="outputs/mmsa/analysis/modality_attribution.json")
    args = parser.parse_args()

    sample_index = args.sample_index
    if sample_index is None and args.from_case_study:
        payload = json.loads(Path(args.from_case_study).read_text(encoding="utf-8"))
        sample_index = int(payload["sample_index"])
    if sample_index is None:
        raise ValueError("Pass --sample-index or --from-case-study to select a test clip.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, test_set, use_interp = load_variant(args.config, args.variant, args.ablation_dir, args.seed, device)
    item = test_set[sample_index]

    attribution = integrated_gradients(model, item, use_interp, device, steps=args.steps)
    result = {
        "variant": args.variant,
        "seed": args.seed,
        "sample_index": sample_index,
        "label": float(item["label"].item()),
        "method": "integrated_gradients_zero_baseline",
        "steps": args.steps,
        "attribution_share": attribution,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
