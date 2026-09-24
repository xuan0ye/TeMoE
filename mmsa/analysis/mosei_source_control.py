"""Report the five-seed MOSEI transcript-only versus matched-cache control."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mmsa.analysis.control_caches import per_seed_mae
from mmsa.analysis.equivalence_tests import holm_adjust, t_cdf, t_ppf, tost

SEEDS = {1, 2, 3, 4, 42}


def find_run(root: Path, arch: str, condition: str) -> Path:
    folders = {
        "matched": ("multiseed_interp", "multiseed"),
        "textonly": ("control_textonly",),
        "none": ("multiseed_nointerp",),
    }[condition]
    for folder in folders:
        path = root / folder / arch / f"multiseed_{arch}.json"
        if path.is_file():
            return path
    raise FileNotFoundError(f"{condition}/{arch}: expected a completed multiseed JSON under {root}")


def read_run(path: Path) -> dict[int, float]:
    result = per_seed_mae(path)
    if set(result) != SEEDS:
        raise ValueError(f"{path}: expected seeds {sorted(SEEDS)}, found {sorted(result)}")
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError(f"{path}: nonfinite MAE")
    return result


def paired(first: dict[int, float], second: dict[int, float]) -> dict:
    result = tost(first, second, 0.06)
    mean, se = result["difference_first_minus_second"], result["se"]
    critical = t_ppf(0.975, 4)
    statistic = 0.0 if se == 0 else mean / se
    p = (0.0 if mean != 0 else 1.0) if se == 0 else 2 * (1 - t_cdf(abs(statistic), 4))
    result["ci95"] = [mean - critical * se, mean + critical * se]
    result["difference_p_two_sided"] = p
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/mmsa/mosei")
    ap.add_argument("--archs", nargs="+", default=["temoe"])
    ap.add_argument("--output", default="outputs/mmsa/mosei/analysis/mosei_source_control.json")
    ap.add_argument("--markdown", default="outputs/mmsa/mosei/analysis/mosei_source_control.md")
    args = ap.parse_args()
    root = Path(args.root)
    rows = {}
    equivalence = []
    for arch in args.archs:
        paths = {condition: find_run(root, arch, condition) for condition in ("matched", "textonly", "none")}
        values = {condition: read_run(path) for condition, path in paths.items()}
        source = paired(values["textonly"], values["matched"])
        gain = paired(values["none"], values["textonly"])
        rows[arch] = {
            "paths": {name: str(path) for name, path in paths.items()},
            "per_seed_mae": {name: {str(k): v for k, v in scores.items()} for name, scores in values.items()},
            "mean_mae": {name: sum(scores.values()) / 5 for name, scores in values.items()},
            "textonly_minus_matched": source,
            "none_minus_textonly": gain,
        }
        equivalence.append((arch, source))
    holm_adjust(equivalence)

    report = {
        "dataset": "mosei",
        "seeds": sorted(SEEDS),
        "equivalence_margin_mae": 0.06,
        "equivalence_family": args.archs,
        "comparison": "Paired test MAE, retraining seeds held fixed; text-only uses same 7B teacher and MiniLM encoder as matched cache.",
        "architectures": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# MOSEI rationale-source control", "",
        "Five paired training seeds; lower MAE is better. Difference is transcript-only minus video+transcript.",
        "Equivalence uses the declared ±0.06 MAE margin, a 90% t CI and Holm correction across architectures in this run.",
        "",
        "| Architecture | None MAE | Transcript-only MAE | Video+transcript MAE | Difference | 95% CI | 90% CI | Holm TOST p | Equivalent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for arch, row in rows.items():
        m = row["mean_mae"]
        s = row["textonly_minus_matched"]
        ci95, ci90 = s["ci95"], s["ci90"]
        lines.append(
            f"| {arch} | {m['none']:.4f} | {m['textonly']:.4f} | {m['matched']:.4f} | "
            f"{s['difference_first_minus_second']:+.4f} | "
            f"[{ci95[0]:+.4f}, {ci95[1]:+.4f}] | [{ci90[0]:+.4f}, {ci90[1]:+.4f}] | "
            f"{s['holm_adjusted_tost_p']:.4g} | {'yes' if s['equivalent_after_holm'] else 'no'} |"
        )
    lines += ["", "This comparison is conditional on the fixed MOSEI test split and fixed cache generation.", ""]
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved {output} and {markdown}")


if __name__ == "__main__":
    main()
