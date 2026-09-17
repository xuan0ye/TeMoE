"""Paired TOST equivalence tests for claims currently phrased as 'matches' or 'neutral'."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


SEEDS = (42, 1, 2, 3, 4)
ARCHS = ("temoe", "lmf", "late_fusion")
LABEL_SPAN = {"mosi": 6.0, "mosei": 6.0, "sims": 2.0}
PRIMARY_FRACTION = 0.01


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else d
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise RuntimeError("Incomplete beta did not converge")


def _betai(a: float, b: float, x: float) -> float:
    if x in (0.0, 1.0):
        return x
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(value: float, df: int) -> float:
    if value == 0.0:
        return 0.5
    tail = 0.5 * _betai(df / 2.0, 0.5, df / (df + value * value))
    return 1.0 - tail if value > 0 else tail


def t_ppf(probability: float, df: int) -> float:
    low, high = -50.0, 50.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if t_cdf(middle, df) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def tost(first: dict[int, float], second: dict[int, float], margin: float) -> dict:
    shared = sorted(set(first) & set(second))
    differences = np.asarray([first[seed] - second[seed] for seed in shared], dtype=np.float64)
    n = len(differences)
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    if se == 0.0:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
    else:
        p_lower = 1.0 - t_cdf((mean + margin) / se, df)
        p_upper = t_cdf((mean - margin) / se, df)
    critical = t_ppf(0.95, df)
    low, high = mean - critical * se, mean + critical * se
    p_equivalence = max(p_lower, p_upper)
    return {
        "n": n,
        "seeds": shared,
        "difference_first_minus_second": mean,
        "sd": sd,
        "se": se,
        "margin": margin,
        "lower_test_p": p_lower,
        "upper_test_p": p_upper,
        "tost_p": p_equivalence,
        "ci90": [low, high],
        "equivalent_at_alpha_0p05": bool(p_equivalence < 0.05 and low > -margin and high < margin),
    }


def per_seed(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["seed"]): float(row["mae"]) for row in data["per_seed"]}


def summary_seed(summary: dict, condition: str) -> dict[int, float]:
    entry = summary[condition]
    return {int(row["seed"]): float(row["mae"]) for row in entry["per_seed"]}


def holm_adjust(entries: list[tuple[str, dict]]) -> None:
    ordered = sorted(entries, key=lambda item: item[1]["tost_p"])
    running = 0.0
    total = len(ordered)
    for index, (_, result) in enumerate(ordered):
        adjusted = min(1.0, (total - index) * result["tost_p"])
        running = max(running, adjusted)
        result["holm_adjusted_tost_p"] = running
        result["equivalent_after_holm"] = bool(
            result["equivalent_at_alpha_0p05"] and running < 0.05
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/mmsa")
    parser.add_argument("--output", default="outputs/mmsa/official/equivalence_tests.json")
    parser.add_argument("--markdown", default="outputs/mmsa/official/equivalence_tests.md")
    args = parser.parse_args()
    root = Path(args.root)
    report = {
        "schema_version": 1,
        "method": "paired two-one-sided t-tests (TOST), alpha=0.05, 90% t confidence interval",
        "margin_policy": (
            "Practical equivalence is 1% of the dataset label span: 0.06 MAE for "
            "MOSI/MOSEI and 0.02 MAE for CH-SIMS."
        ),
        "primary_margin_fraction_of_label_span": PRIMARY_FRACTION,
        "families": {},
        "sensitivity_mosi": {},
    }

    controls = json.loads((root / "official/control_caches.json").read_text(encoding="utf-8"))
    modality_families = {
        "mosi_qwen7b_textonly_vs_matched": [],
        "mosi_cliponly_vs_none": [],
    }
    for arch in ARCHS:
        paths = {name: root.parent.parent / controls[arch][name]["path"] for name in ("matched", "videonly", "textonly", "none")}
        values = {name: per_seed(path) for name, path in paths.items()}
        text_result = tost(values["textonly"], values["matched"], 0.06)
        clip_result = tost(values["videonly"], values["none"], 0.06)
        modality_families["mosi_qwen7b_textonly_vs_matched"].append((arch, text_result))
        modality_families["mosi_cliponly_vs_none"].append((arch, clip_result))
        report["sensitivity_mosi"][arch] = {
            "qwen7b_textonly_vs_matched": {
                f"{margin:.2f}": tost(values["textonly"], values["matched"], margin)["equivalent_at_alpha_0p05"]
                for margin in (0.02, 0.03, 0.04, 0.05, 0.06)
            },
            "cliponly_vs_none": {
                f"{margin:.2f}": tost(values["videonly"], values["none"], margin)["equivalent_at_alpha_0p05"]
                for margin in (0.02, 0.03, 0.04, 0.05, 0.06)
            },
        }
    for family, entries in modality_families.items():
        holm_adjust(entries)
        report["families"][family] = {name: result for name, result in entries}

    shuffled = per_seed(root / "mosi/multiseed_shuffled/temoe/multiseed_temoe.json")
    none = per_seed(root / controls["temoe"]["none"]["path"].replace("outputs/mmsa/", ""))
    mismatch_result = tost(shuffled, none, 0.06)
    mismatch_result["holm_adjusted_tost_p"] = mismatch_result["tost_p"]
    mismatch_result["equivalent_after_holm"] = mismatch_result["equivalent_at_alpha_0p05"]
    report["families"]["mosi_mispaired_vs_none"] = {"temoe": mismatch_result}

    for dataset in ("mosi", "sims", "mosei"):
        summary = json.loads((root / f"{dataset}/ablation/ablation_summary.json").read_text(encoding="utf-8"))
        margin = LABEL_SPAN[dataset] * PRIMARY_FRACTION
        entries = []
        for variant in ("no_mamba", "ta_gated_conv_text", "dense_moe"):
            result = tost(summary_seed(summary, variant), summary_seed(summary, "full"), margin)
            entries.append((variant, result))
        holm_adjust(entries)
        report["families"][f"{dataset}_architecture_choices_vs_full"] = {
            name: result for name, result in entries
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Paired equivalence tests",
        "",
        report["margin_policy"],
        "",
        "| family | comparison | mean difference | 90% CI | margin | TOST p | Holm p | equivalent |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for family, entries in report["families"].items():
        for name, result in entries.items():
            low, high = result["ci90"]
            lines.append(
                f"| {family} | {name} | {result['difference_first_minus_second']:+.4f} | "
                f"[{low:+.4f}, {high:+.4f}] | +/-{result['margin']:.3f} | "
                f"{result['tost_p']:.4g} | {result['holm_adjusted_tost_p']:.4g} | "
                f"{'yes' if result['equivalent_after_holm'] else 'no'} |"
            )
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[done] wrote {output} and {markdown}")


if __name__ == "__main__":
    main()
