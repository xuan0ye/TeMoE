from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

_TABLE_METRICS = ["mae", "corr", "acc7", "acc2_has0", "f1_has0", "acc2_non0", "f1_non0"]
_HIGHER_IS_BETTER = {"corr", "acc7", "acc5", "acc2_has0", "f1_has0", "acc2_non0", "f1_non0"}


def summarize(results_dir: str, reference_arch: str, output_dir: str) -> dict:
    runs = _load_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"No multiseed_*.json found under {results_dir}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _write_main_table(runs, output / "main_table.md", output / "main_table.csv")
    significance = _significance(runs, reference_arch)
    with (output / "significance.json").open("w", encoding="utf-8") as file:
        json.dump(significance, file, ensure_ascii=False, indent=2)

    _write_significance_note(runs, significance, reference_arch, output / "main_table.md")
    return {"archs": list(runs.keys()), "significance": significance}


def _load_runs(results_dir: str) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for path in sorted(glob.glob(str(Path(results_dir) / "**" / "multiseed_*.json"), recursive=True)):
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        runs[payload["arch"]] = payload
    return runs


def _write_main_table(runs: dict[str, dict], md_path: Path, csv_path: Path) -> None:
    header = ["arch"] + _TABLE_METRICS
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    csv_rows = [header]
    for arch, payload in runs.items():
        mean = payload["mean"]
        std = payload["std"]
        md_cells = [arch]
        csv_cells = [arch]
        for metric in _TABLE_METRICS:
            m = mean.get(metric, float("nan"))
            s = std.get(metric, float("nan"))
            md_cells.append(f"{m:.3f}±{s:.3f}")
            csv_cells.append(f"{m:.4f}")
        md_lines.append("| " + " | ".join(md_cells) + " |")
        csv_rows.append(csv_cells)

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(csv_rows)


def _significance(runs: dict[str, dict], reference_arch: str) -> dict:
    if reference_arch not in runs:
        return {"note": f"reference arch '{reference_arch}' not found; skipped significance."}
    try:
        from scipy import stats
    except Exception:
        return {"note": "scipy not available; significance skipped."}

    ref = _per_seed_map(runs[reference_arch])
    out: dict[str, dict] = {}
    for arch, payload in runs.items():
        if arch == reference_arch:
            continue
        other = _per_seed_map(payload)
        metric_stats: dict[str, dict] = {}
        for metric in _TABLE_METRICS:
            a = [row[metric] for row in ref if _num(row.get(metric))]
            b = [row[metric] for row in other if _num(row.get(metric))]
            if len(a) == len(b) and len(a) >= 2:
                result = stats.ttest_rel(a, b)
                metric_stats[metric] = {
                    "t": float(result.statistic),
                    "p": float(result.pvalue),
                    "significant_0.05": bool(result.pvalue < 0.05),
                }
        out[arch] = metric_stats
    return out


def _per_seed_map(payload: dict) -> list[dict]:
    return payload.get("per_seed", [])


def _num(value) -> bool:
    return isinstance(value, (int, float)) and value == value  # excludes NaN


def _write_significance_note(runs: dict, significance: dict, reference_arch: str, md_path: Path) -> None:
    lines = ["", f"Paired t-test vs `{reference_arch}` (across seeds), p<0.05 marked:"]
    if "note" in significance:
        lines.append(f"- {significance['note']}")
    else:
        for arch, metrics in significance.items():
            sig = [m for m, s in metrics.items() if s.get("significant_0.05")]
            lines.append(f"- {arch}: significant on {sig if sig else 'none'}")
    with md_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multiseed runs into a main comparison table + significance.")
    parser.add_argument("--results-dir", required=True, help="dir containing multiseed_*.json (searched recursively)")
    parser.add_argument("--reference-arch", default="temoe")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(args.results_dir) / "summary")
    result = summarize(args.results_dir, args.reference_arch, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
