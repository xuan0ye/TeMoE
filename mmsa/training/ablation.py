from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from mmsa.training.multiseed import _aggregate
from mmsa.training.train import load_config, train

# Each variant is (name, config-patch). Patches are applied on top of the base config.
# no_audio/no_vision/no_text keep two modalities (bi-modal); text_only/audio_only/vision_only
# drop to a single modality. Together with `full` (all three), this reproduces the full
# uni-/bi-modal combination grid TEXT (Rao et al. 2026) reports in its Table 2, so the
# modality-importance ablation (\S~\ref{sec:ablation}) can be read off directly rather than
# only inferred from single-modality removals.
VARIANTS: list[tuple[str, dict]] = [
    ("full", {}),
    ("no_mamba", {"model": {"use_mamba": False}}),
    ("ta_gated_conv_text", {"model": {"temporal_variant": "gated_conv"}}),
    ("no_interpretation", {"model": {"use_interpretation": False}}),
    ("dense_moe", {"model": {"top_k": None}}),  # top_k -> num_experts (dense) resolved below
    ("no_audio", {"disabled_modalities": ["audio"]}),
    ("no_vision", {"disabled_modalities": ["vision"]}),
    ("no_text", {"disabled_modalities": ["text"]}),
    ("text_only", {"disabled_modalities": ["audio", "vision"]}),
    ("audio_only", {"disabled_modalities": ["vision", "text"]}),
    ("vision_only", {"disabled_modalities": ["audio", "text"]}),
    # Responds to the "why not fix expert collapse with stronger load balancing?"
    # question: the same architecture trained with a 10x larger balancing term.
    ("high_aux", {"training": {"aux_loss_weight": 0.1}}),
]

# Plan B (person-centric clue extraction): structured face/content/bg channels.
#   full_pb       -- face + content + gated bg + cross-modal consistency
#   no_face       -- content + gated bg + consistency (face channel dropped)
#   no_cons       -- face + content + gated bg (consistency module dropped)
#   no_bg_gate    -- face + content, background always suppressed
#   single_channel -- legacy single-vector interpretation (pre-Plan-B TeMoE)
PLAN_B_VARIANTS: list[tuple[str, dict]] = [
    ("full_pb", {"model": {"use_channels": True, "use_face": True, "use_bg_gate": True, "use_consistency": True}}),
    ("no_face", {"model": {"use_channels": True, "use_face": False, "use_bg_gate": True, "use_consistency": True}}),
    ("no_cons", {"model": {"use_channels": True, "use_face": True, "use_bg_gate": True, "use_consistency": False}}),
    ("no_bg_gate", {"model": {"use_channels": True, "use_face": True, "use_bg_gate": False, "use_consistency": True}}),
    ("single_channel", {"model": {"use_channels": False, "use_face": False, "use_bg_gate": False, "use_consistency": False}}),
]

_TABLE_METRICS = ["mae", "corr", "acc7", "acc2_has0", "f1_has0", "acc2_non0", "f1_non0"]
_REFERENCE_VARIANT = "full"


def run_ablation(
    config_path: str,
    output_dir: str,
    seeds: list[int] | None = None,
    plan_b: bool = False,
    only: list[str] | None = None,
) -> dict:
    """Run every ablation variant across `seeds` and aggregate mean/std per variant.

    A single seed=42 run is *not* reliable for close comparisons (e.g. our hybrid
    Mamba+cross-attention block vs. TEXT's gated-conv block): the real Mamba CUDA/
    Triton kernels are not bit-deterministic across process runs even with a fixed
    seed, and on a small dataset with early stopping that noise can rival or exceed
    the gap between variants. Multi-seed averaging is required before drawing any
    "variant A beats variant B" conclusion.
    """
    base = load_config(config_path)
    seeds = seeds or [int(base.get("seed", 42))]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    variants = PLAN_B_VARIANTS if plan_b else VARIANTS
    if only:
        # selective runs (e.g. --only full high_aux) avoid retraining variants that
        # already have results while keeping the same reference row for the t-tests
        wanted = {v.lower() for v in only}
        variants = [(n, p) for (n, p) in variants if n.lower() in wanted]
        if not variants:
            raise SystemExit(f"--only {only} matched no variant in the selected variant list")
    results: dict[str, dict] = {}
    for name, patch in variants:
        per_seed: list[dict] = []
        params = None
        temporal_backend = None
        for seed in seeds:
            config = _apply_patch(copy.deepcopy(base), patch, name)
            config["seed"] = seed
            config["training"]["output_dir"] = str(output / name / f"seed_{seed}")
            print(f"\n===== ablation variant: {name} seed={seed} =====")
            report = train(config)
            per_seed.append({"seed": seed, **report["test"]})
            params = report["params"]
            temporal_backend = report["temporal_backend"]

        aggregate = _aggregate(per_seed)
        results[name] = {
            "seeds": seeds,
            "per_seed": per_seed,
            "mean": aggregate["mean"],
            "std": aggregate["std"],
            "params": params,
            "temporal_backend": temporal_backend,
        }

    _write_summary(results, output / "ablation_summary.json", output / "ablation_summary.md")
    return results


def _apply_patch(config: dict, patch: dict, name: str) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict):
            config.setdefault(key, {}).update(value)
        else:
            config[key] = value
    if name == "dense_moe":
        config["model"]["top_k"] = int(config["model"].get("num_experts", 8))
    return config


def _write_summary(results: dict, json_path: Path, md_path: Path) -> None:
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    n_seeds = len(next(iter(results.values()))["seeds"]) if results else 0
    header = "| variant | params | temporal_backend | " + " | ".join(_TABLE_METRICS) + " |"
    divider = "| " + " | ".join(["---"] * (len(_TABLE_METRICS) + 3)) + " |"
    lines = [f"n_seeds={n_seeds}", "", header, divider]
    for name, payload in results.items():
        mean, std = payload["mean"], payload["std"]
        cells = [f"{mean.get(col, float('nan')):.4f}±{std.get(col, float('nan')):.4f}" for col in _TABLE_METRICS]
        lines.append(f"| {name} | {payload['params']} | {payload['temporal_backend']} | " + " | ".join(cells) + " |")

    lines.extend(_significance_lines(results))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _significance_lines(results: dict) -> list[str]:
    if _REFERENCE_VARIANT not in results or len(results[_REFERENCE_VARIANT]["seeds"]) < 2:
        return ["", "(paired significance test skipped: need >=2 seeds and a `full` reference row)"]
    try:
        from scipy import stats
    except Exception:
        return ["", "(scipy not available; paired significance test skipped)"]

    ref_per_seed = {row["seed"]: row for row in results[_REFERENCE_VARIANT]["per_seed"]}
    lines = ["", f"Paired t-test vs `{_REFERENCE_VARIANT}` (matched by seed), p<0.05 marked with *:"]
    for name, payload in results.items():
        if name == _REFERENCE_VARIANT:
            continue
        other_per_seed = {row["seed"]: row for row in payload["per_seed"]}
        shared_seeds = sorted(set(ref_per_seed) & set(other_per_seed))
        if len(shared_seeds) < 2:
            lines.append(f"- {name}: skipped (fewer than 2 shared seeds with `{_REFERENCE_VARIANT}`)")
            continue
        cells = []
        for metric in _TABLE_METRICS:
            a = [ref_per_seed[s][metric] for s in shared_seeds if _is_number(ref_per_seed[s].get(metric))]
            b = [other_per_seed[s][metric] for s in shared_seeds if _is_number(other_per_seed[s].get(metric))]
            if len(a) != len(b) or len(a) < 2:
                continue
            result = stats.ttest_rel(a, b)
            mark = "*" if result.pvalue < 0.05 else ""
            cells.append(f"{metric}: p={result.pvalue:.3f}{mark}")
        lines.append(f"- {name}: " + (", ".join(cells) if cells else "no comparable metrics"))
    return lines


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and value == value  # excludes NaN


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TeMoE ablation suite across multiple seeds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/mmsa/ablation")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="run only these variants (e.g. --only full high_aux); default runs every variant",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 1, 2, 3, 4],
        help="run every variant with each of these seeds and report mean±std (default: 5 seeds, matches multiseed.py)",
    )
    parser.add_argument(
        "--plan-b",
        action="store_true",
        help="run the Plan B variants (full_pb / no_face / no_cons / no_bg_gate / single_channel) "
        "instead of the classic variant list",
    )
    args = parser.parse_args()
    run_ablation(args.config, args.output_dir, args.seeds, plan_b=args.plan_b, only=args.only)


if __name__ == "__main__":
    main()
