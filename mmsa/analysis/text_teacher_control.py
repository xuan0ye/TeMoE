"""Compare a 1.5B text teacher with raw text and the 7B transcript-only cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmsa.analysis.control_caches import load, paired_stats, per_seed_mae


ARCHS = ("temoe", "lmf", "late_fusion")


def _load_control(path: Path, arch: str, condition: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("arch") != arch or data.get("condition") != condition:
        raise RuntimeError(f"Unexpected result identity in {path}")
    if len(data.get("per_seed") or []) != 5:
        raise RuntimeError(f"Expected five seeds in {path}")
    return {
        "mae": float(data["mean"]["mae"]),
        "mae_std": float(data["std"]["mae"]),
        "corr": float(data["mean"]["corr"]),
        "corr_std": float(data["std"]["corr"]),
        "n": len(data["per_seed"]),
        "manifest": data.get("manifest"),
        "path": str(path),
    }


def _clean(value: dict) -> dict:
    return {
        key: (None if isinstance(item, float) and (item != item or abs(item) == float("inf")) else item)
        for key, item in value.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/mmsa")
    parser.add_argument("--dataset", default="mosi")
    parser.add_argument("--latency", default="outputs/mmsa/mosi/efficiency/text_teacher_latency.json")
    parser.add_argument("--output", default="outputs/mmsa/official/text_teacher_control.json")
    args = parser.parse_args()

    dataset_root = Path(args.root) / args.dataset
    rows: dict[str, dict] = {}
    tests: dict[str, dict] = {}
    for arch in ARCHS:
        teacher = _load_control(
            dataset_root / "control_text_teacher_1p5b" / arch / f"multiseed_{arch}.json",
            arch,
            "text_teacher_1p5b",
        )
        rawtext = _load_control(
            dataset_root / "control_rawtext" / arch / f"multiseed_{arch}.json",
            arch,
            "raw_transcript_direct_embedding",
        )
        row = {"teacher_1p5b": teacher, "rawtext": rawtext}
        for source, target in (("none", "none"), ("textonly", "qwen7b_textonly"), ("matched", "matched")):
            result, why = load(dataset_root, arch, source)
            if result is None:
                raise FileNotFoundError(f"{arch}/{source}: {why}")
            row[target] = result
        rows[arch] = row
        seed_values = {name: per_seed_mae(entry["path"]) for name, entry in row.items()}
        tests[arch] = {
            "teacher_1p5b_vs_none": _clean(paired_stats(seed_values["teacher_1p5b"], seed_values["none"])),
            "teacher_1p5b_vs_rawtext": _clean(paired_stats(seed_values["teacher_1p5b"], seed_values["rawtext"])),
            "teacher_1p5b_vs_qwen7b_textonly": _clean(
                paired_stats(seed_values["teacher_1p5b"], seed_values["qwen7b_textonly"])
            ),
            "teacher_1p5b_vs_matched": _clean(paired_stats(seed_values["teacher_1p5b"], seed_values["matched"])),
            "qwen7b_textonly_vs_matched": _clean(
                paired_stats(seed_values["qwen7b_textonly"], seed_values["matched"])
            ),
        }

    latency = json.loads(Path(args.latency).read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "dataset": args.dataset,
        "question": (
            "Can a cheaper 1.5B text-only teacher preserve the benefit of the 7B VL "
            "model used without video?"
        ),
        "difference_convention": "first condition MAE minus second condition MAE; negative is better",
        "rows": rows,
        "paired_tests": tests,
        "latency": latency,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== Text-teacher control (MOSI, 5 seeds, mean +/- std) ===")
    print(f"{'arch':<14}{'none':>18}{'raw':>18}{'1.5B teacher':>22}{'7B text-only':>22}{'matched':>18}")
    for arch in ARCHS:
        row = rows[arch]
        cell = lambda key: f"{row[key]['mae']:.4f}+/-{row[key]['mae_std']:.4f}"
        print(
            f"{arch:<14}{cell('none'):>18}{cell('rawtext'):>18}{cell('teacher_1p5b'):>22}"
            f"{cell('qwen7b_textonly'):>22}{cell('matched'):>18}"
        )
        for name, stat in tests[arch].items():
            print(
                f"  {name:<36} diff={stat.get('diff')!s:<12} "
                f"95%CI=[{stat.get('lo')!s}, {stat.get('hi')!s}] p={stat.get('p')!s}"
            )
    print(
        f"[latency] 1.5B={latency['text_1p5b']['median_s']:.4f}s "
        f"7B={latency['vl_7b_textonly']['median_s']:.4f}s "
        f"speedup={latency['median_speedup_1p5b_over_7b']:.2f}x"
    )
    print(f"[done] wrote {output}")


if __name__ == "__main__":
    main()
