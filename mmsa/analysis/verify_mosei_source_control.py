"""Verify the released MOSEI source-control report from its paired seed MAEs.

No dataset, GPU, video, checkpoint, or cached embedding is needed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mmsa.analysis.equivalence_tests import holm_adjust, tost

ARCHS = ("temoe", "lmf", "late_fusion")
SEEDS = (1, 2, 3, 4, 42)


def score_map(values: dict[str, float]) -> dict[int, float]:
    scores = {int(seed): float(value) for seed, value in values.items()}
    if tuple(sorted(scores)) != SEEDS:
        raise ValueError(f"Expected five paired seeds {SEEDS}; found {tuple(sorted(scores))}")
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("Nonfinite test MAE in released report")
    return scores


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError(f"{label}: recomputed {actual} differs from released {expected}")


def verify(report: dict) -> dict[str, dict[str, dict]]:
    if report.get("dataset") != "mosei":
        raise ValueError("Expected the MOSEI report")
    if set(report.get("architectures", {})) != set(ARCHS):
        raise ValueError("Expected TeMoE, LMF, and Late Fusion")
    if tuple(sorted(report.get("seeds", ()))) != SEEDS:
        raise ValueError("Unexpected training seeds")
    close(float(report["equivalence_margin_mae"]), 0.06, "declared margin")

    scores = {}
    for arch in ARCHS:
        row = report["architectures"][arch]
        scores[arch] = {
            condition: score_map(row["per_seed_mae"][condition])
            for condition in ("none", "textonly", "matched")
        }
        for condition, values in scores[arch].items():
            close(
                sum(values.values()) / len(values),
                float(row["mean_mae"][condition]),
                f"{arch}/{condition} mean MAE",
            )

    results = {}
    for margin in (0.06, 0.01):
        entries = [
            (arch, tost(scores[arch]["textonly"], scores[arch]["matched"], margin))
            for arch in ARCHS
        ]
        holm_adjust(entries)
        results[f"{margin:.2f}"] = dict(entries)

    for arch in ARCHS:
        released = report["architectures"][arch]["textonly_minus_matched"]
        computed = results["0.06"][arch]
        for field in (
            "difference_first_minus_second", "sd", "se", "tost_p",
            "holm_adjusted_tost_p",
        ):
            close(float(computed[field]), float(released[field]), f"{arch}/{field}")
        for field in ("ci90",):
            for i, (actual, expected) in enumerate(zip(computed[field], released[field])):
                close(float(actual), float(expected), f"{arch}/{field}[{i}]")
        if computed["equivalent_after_holm"] != released["equivalent_after_holm"]:
            raise ValueError(f"{arch}: equivalence decision differs from released report")

    if not all(results["0.01"][arch]["equivalent_after_holm"] for arch in ARCHS):
        raise ValueError("The reported +/-0.01 sensitivity claim did not reproduce")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="results/mosei/analysis/mosei_source_control.json",
        help="Published aggregate JSON, relative to the repository root",
    )
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    results = verify(report)
    for margin in ("0.06", "0.01"):
        print(f"TOST margin +/-{margin} MAE (five paired seeds; Holm across three consumers)")
        for arch in ARCHS:
            row = results[margin][arch]
            print(
                f"  {arch:11s} difference={row['difference_first_minus_second']:+.6f} "
                f"Holm p={row['holm_adjusted_tost_p']:.8g} "
                f"equivalent={row['equivalent_after_holm']}"
            )
    print("MOSEI released source-control report verified")


if __name__ == "__main__":
    main()