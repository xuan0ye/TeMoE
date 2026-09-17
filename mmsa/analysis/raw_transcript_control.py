"""Compare direct transcript embeddings against Qwen transcript-only and matched caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmsa.analysis.control_caches import load, paired_stats, per_seed_mae


ARCHS = ("temoe", "lmf", "late_fusion")


def _load_rawtext(root: Path, arch: str) -> dict:
    path = root / "control_rawtext" / arch / f"multiseed_{arch}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("arch") != arch or data.get("condition") != "raw_transcript_direct_embedding":
        raise RuntimeError(f"Unexpected raw-transcript result identity in {path}")
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


def _clean_stats(value: dict) -> dict:
    return {
        key: (None if isinstance(item, float) and (item != item or abs(item) == float("inf")) else item)
        for key, item in value.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/mmsa")
    parser.add_argument("--dataset", default="mosi")
    parser.add_argument(
        "--output",
        default="outputs/mmsa/official/raw_transcript_control.json",
    )
    args = parser.parse_args()

    dataset_root = Path(args.root) / args.dataset
    rows: dict[str, dict] = {}
    tests: dict[str, dict] = {}
    for arch in ARCHS:
        rawtext = _load_rawtext(dataset_root, arch)
        row = {"rawtext": rawtext}
        for condition in ("none", "textonly", "matched"):
            result, why = load(dataset_root, arch, condition)
            if result is None:
                raise FileNotFoundError(f"{arch}/{condition}: {why}")
            row[condition] = result
        rows[arch] = row

        seed_values = {
            condition: per_seed_mae(entry["path"])
            for condition, entry in row.items()
        }
        tests[arch] = {
            "rawtext_vs_none": _clean_stats(
                paired_stats(seed_values["rawtext"], seed_values["none"])
            ),
            "rawtext_vs_qwen_textonly": _clean_stats(
                paired_stats(seed_values["rawtext"], seed_values["textonly"])
            ),
            "rawtext_vs_matched": _clean_stats(
                paired_stats(seed_values["rawtext"], seed_values["matched"])
            ),
            "qwen_textonly_vs_matched": _clean_stats(
                paired_stats(seed_values["textonly"], seed_values["matched"])
            ),
        }

    report = {
        "schema_version": 1,
        "dataset": args.dataset,
        "question": (
            "Does Qwen transcript rewriting add value beyond directly encoding the same raw "
            "transcript with the identical frozen sentence encoder?"
        ),
        "difference_convention": "first condition MAE minus second condition MAE; negative is better",
        "rows": rows,
        "paired_tests": tests,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== Raw transcript embedding control (5 seeds, mean +/- std) ===")
    print(f"{'arch':<14}{'none':>18}{'raw transcript':>22}{'Qwen text-only':>22}{'matched':>18}")
    for arch in ARCHS:
        row = rows[arch]
        cell = lambda key: f"{row[key]['mae']:.4f}+/-{row[key]['mae_std']:.4f}"
        print(
            f"{arch:<14}{cell('none'):>18}{cell('rawtext'):>22}"
            f"{cell('textonly'):>22}{cell('matched'):>18}"
        )

    print("\n=== Paired MAE differences (first minus second) ===")
    for arch in ARCHS:
        print(f"\n[{arch}]")
        for name, stat in tests[arch].items():
            print(
                f"  {name:<28} diff={stat.get('diff')!s:<12} "
                f"95%CI=[{stat.get('lo')!s}, {stat.get('hi')!s}] p={stat.get('p')!s}"
            )
    print(f"\n[done] wrote {output}")


if __name__ == "__main__":
    main()
