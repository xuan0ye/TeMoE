"""Summarize Plan B results into paper-ready tables + a decision note.

Run after server results are copied back:
    python -m mmsa.analysis.summarize_planb --results-dir outputs/mmsa \
        --out-dir outputs/mmsa/planb_paper

Reads, per dataset (mosi/sims/mosei):
  - planb_ablation/ablation_summary.json   (5 variants x 5 seeds mean/std/significance)
  - planb_analysis/bg_gate_report.md
  - planb_analysis/consistency_report.md
  - planb_analysis/case_study.md

Writes:
  - paper_table.md    : main Plan B ablation table (variant x dataset x metric)
  - decision.md       : whether the method story holds (full_pb vs variants)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATASETS = ("mosi", "sims", "mosei")
VARIANTS = ("full_pb", "no_face", "no_cons", "no_bg_gate", "single_channel")
METRICS = ("mae", "corr", "acc7", "f1_has0")


def _load_ablation(base: Path, dataset: str) -> dict | None:
    path = base / dataset / "planb_ablation" / "ablation_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(mean: dict, std: dict, metric: str) -> str:
    m = mean.get(metric, float("nan"))
    s = std.get(metric, float("nan"))
    return f"{m:.4f}±{s:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/mmsa")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    base = Path(args.results_dir)
    out = Path(args.out_dir or (base / "planb_paper"))
    out.mkdir(parents=True, exist_ok=True)

    table_lines = [
        "# Plan B ablation (mean±std, 5 seeds)",
        "",
        "| dataset | variant | MAE↓ | Corr↑ | Acc-7↑ | F1↑ | params |",
        "|---|---|---|---|---|---|---|",
    ]
    decision_lines = [
        "# Plan B decision note",
        "",
        "Generated from ablation_summary.json + analysis reports.",
        "",
    ]

    any_data = False
    for ds in DATASETS:
        ab = _load_ablation(base, ds)
        if not ab:
            decision_lines.append(f"## {ds}: NO ablation data yet")
            continue
        any_data = True
        decision_lines.append(f"## {ds}")
        for v in VARIANTS:
            payload = ab.get(v)
            if not payload:
                decision_lines.append(f"- {v}: missing")
                continue
            mean, std = payload.get("mean", {}), payload.get("std", {})
            row = f"| {ds} | {v} | " + " | ".join(_fmt(mean, std, m) for m in METRICS)
            row += f" | {payload.get('params', '')} |"
            table_lines.append(row)

        # paired t-test vs full_pb: re-derive from per-seed if present
        ref = ab.get("full_pb", {}).get("per_seed", [])
        ref_by_seed = {r["seed"]: r for r in ref}
        if len(ref) >= 2:
            try:
                from scipy import stats as _stats
            except Exception:
                _stats = None
            if _stats is not None:
                for v in VARIANTS:
                    if v == "full_pb":
                        continue
                    other = {r["seed"]: r for r in ab.get(v, {}).get("per_seed", [])}
                    shared = sorted(set(ref_by_seed) & set(other))
                    if len(shared) < 2:
                        continue
                    sigs = []
                    for m in METRICS:
                        a = [ref_by_seed[s][m] for s in shared if m in ref_by_seed[s]]
                        b = [other[s][m] for s in shared if m in other[s]]
                        if len(a) == len(b) and len(a) >= 2:
                            p = _stats.ttest_rel(a, b).pvalue
                            if p < 0.05:
                                sigs.append(f"{m}:p={p:.3f}")
                    decision_lines.append(
                        f"- {v} vs full_pb: " + (", ".join(sigs) if sigs else "no significant difference")
                    )

        # analysis reports
        for name in ("bg_gate_report", "consistency_report", "case_study"):
            p = base / ds / "planb_analysis" / f"{name}.md"
            if p.exists():
                decision_lines.append(f"  - {name}: present ({p.stat().st_size} bytes)")
            else:
                decision_lines.append(f"  - {name}: MISSING")

    if not any_data:
        decision_lines.append("")
        decision_lines.append("**No Plan B ablation data found under --results-dir. "
                              "Copy back outputs/mmsa/<ds>/planb_ablation and planb_analysis first.**")
        print("\n".join(decision_lines))
        (out / "decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
        sys.exit(1)

    (out / "paper_table.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    (out / "decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    print("\n".join(decision_lines))
    print(f"\nwrote: {out / 'paper_table.md'} and {out / 'decision.md'}")


if __name__ == "__main__":
    main()