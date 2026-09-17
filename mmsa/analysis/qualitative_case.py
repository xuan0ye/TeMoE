"""Qualitative single-sample case study, in the style of TEXT's Table 3
(Rao et al. 2026, "A Text-Routed Sparse Mixture-of-Experts Model with
Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis").

TEXT picks one MOSI test clip where uni-modal predictions disagree with the
ground truth and shows how adding modalities / explanations / architectural
components changes the predicted score and its deviation from the label.
This script reproduces that presentation for TeMoE using checkpoints already
produced by `mmsa.training.ablation` (per-variant, per-seed) and, optionally,
`mmsa.training.multiseed` (per-arch baselines), so it requires no new
training run -- only that those checkpoints exist on disk for the requested
seed.

Usage (after an ablation run has produced checkpoints under
outputs/mmsa/mosi/ablation/<variant>/seed_42/temoe_model.pt):

    python -m mmsa.analysis.qualitative_case \
        --config mmsa/configs/mosi.yaml \
        --ablation-dir outputs/mmsa/mosi/ablation \
        --baseline-dir outputs/mmsa/mosi/multiseed \
        --seed 42 \
        --output outputs/mmsa/mosi/analysis/qualitative_case.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mmsa.analysis.common import load_baseline_arch, load_variant

DEFAULT_ABLATION_VARIANTS = [
    "full",
    "no_interpretation",
    "text_only",
    "audio_only",
    "vision_only",
    "no_audio",
    "no_vision",
    "no_text",
    "no_mamba",
    "ta_gated_conv_text",
    "dense_moe",
]
DEFAULT_BASELINE_ARCHS = [
    "late_fusion",
    "cross_modal",
    "lmf",
    "almt",
    "kuda",
    "misa",
    "selfmm",
    "magbert",
    "cormult",
]


def build_case_study(
    config_path: str,
    ablation_dir: str,
    baseline_dir: str | None,
    seed: int,
    variants: list[str],
    baseline_archs: list[str],
    sample_index: int | None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions_by_variant: dict[str, np.ndarray] = {}
    label_by_variant: dict[str, np.ndarray] = {}

    for name in variants:
        model, test_set, use_interp = load_variant(config_path, name, ablation_dir, seed, device)
        preds, labels = _predict_all(model, test_set, device, use_interp)
        predictions_by_variant[name] = preds
        label_by_variant[name] = labels

    for arch in baseline_archs:
        if baseline_dir is None:
            continue
        try:
            model, test_set, use_interp = load_baseline_arch(config_path, arch, baseline_dir, seed, device)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue
        preds, labels = _predict_all(model, test_set, device, use_interp)
        predictions_by_variant[f"baseline:{arch}"] = preds
        label_by_variant[f"baseline:{arch}"] = labels

    if sample_index is None:
        sample_index = _auto_select_index(predictions_by_variant, label_by_variant)

    label = float(next(iter(label_by_variant.values()))[sample_index])
    rows = []
    for name, preds in predictions_by_variant.items():
        score = float(preds[sample_index])
        rows.append(
            {
                "variant": name,
                "score": round(score, 3),
                "label": round(label, 3),
                "deviation": round(abs(score - label), 3),
            }
        )

    return {
        "seed": seed,
        "sample_index": int(sample_index),
        "label": round(label, 3),
        "rows": rows,
    }


@torch.no_grad()
def _predict_all(model: torch.nn.Module, dataset, device: torch.device, use_interp: bool) -> tuple[np.ndarray, np.ndarray]:
    preds, labels = [], []
    for index in range(len(dataset)):
        item = dataset[index]
        interpretation = item.get("interpretation")
        if use_interp and interpretation is not None:
            interpretation = interpretation.unsqueeze(0).to(device)
        else:
            interpretation = None
        outputs = model(
            audio=item["audio"].unsqueeze(0).to(device),
            vision=item["vision"].unsqueeze(0).to(device),
            text=item["text"].unsqueeze(0).to(device),
            interpretation=interpretation,
        )
        preds.append(float(outputs["reg"].item()))
        labels.append(float(item["label"].item()))
    return np.asarray(preds, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def _auto_select_index(predictions_by_variant: dict[str, np.ndarray], label_by_variant: dict[str, np.ndarray]) -> int:
    """Pick the test sample where uni-modal predictions are most misleading but
    `full` is closest to the label -- i.e. the strongest illustration that
    combining modalities (+ explanation) rescues an otherwise-wrong prediction,
    mirroring the motivating example in TEXT's Fig. 1 / Table 3.
    """
    if "full" not in predictions_by_variant:
        raise ValueError("Auto-selection requires the 'full' variant among --variants.")
    uni_names = [n for n in ("text_only", "audio_only", "vision_only") if n in predictions_by_variant]
    if not uni_names:
        raise ValueError(
            "Auto-selection requires at least one of text_only/audio_only/vision_only among "
            "--variants (run the extended mmsa.training.ablation sweep first), or pass --sample-index."
        )
    labels = label_by_variant["full"]
    full_err = np.abs(predictions_by_variant["full"] - labels)
    uni_err = np.max(np.stack([np.abs(predictions_by_variant[n] - labels) for n in uni_names]), axis=0)
    rescue = uni_err - full_err
    return int(np.argmax(rescue))


def main() -> None:
    parser = argparse.ArgumentParser(description="TEXT-Table-3-style qualitative case study for TeMoE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation-dir", required=True, help="output-dir used for mmsa.training.ablation")
    parser.add_argument("--baseline-dir", default=None, help="output-dir used for mmsa.training.multiseed (optional)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_ABLATION_VARIANTS)
    parser.add_argument("--baseline-archs", nargs="+", default=DEFAULT_BASELINE_ARCHS)
    parser.add_argument("--sample-index", type=int, default=None, help="skip auto-selection, use this test index")
    parser.add_argument("--output", default="outputs/mmsa/analysis/qualitative_case.json")
    args = parser.parse_args()

    result = build_case_study(
        args.config,
        args.ablation_dir,
        args.baseline_dir,
        args.seed,
        args.variants,
        args.baseline_archs,
        args.sample_index,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        f"Sample index {result['sample_index']} (label = {result['label']}):",
        "",
        "| Variant | Score | Deviation |",
        "| --- | --- | --- |",
    ]
    for row in result["rows"]:
        md_lines.append(f"| {row['variant']} | {row['score']} | {row['deviation']} |")
    md_path = output_path.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nwrote {output_path} and {md_path}")


if __name__ == "__main__":
    main()
