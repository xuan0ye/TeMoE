"""Audit which condition every trained run actually used, and rebuild the crossing table.

Directory names in ``outputs/mmsa`` are a *convention*, and a fragile one: the
first MOSI sweep wrote explanation-augmented and explanation-free runs into the
same ``multiseed/`` folder, so the folder name alone cannot tell you what a
number means.  Every run report, however, records the truth:

    {"arch": "temoe", "use_interpretation": true, "params": 1245705, ...}

This script reads those reports (not the folder names), regroups runs by
(architecture, condition), recomputes the with/without crossing table, and
checks the result against the numbers printed in the paper.  It also reports
anomalies -- mixed folders, duplicate runs, incomplete seed sets, and diverged
configurations -- so that a partial MOSEI sweep can be trusted at a glance.

Usage:
  python -m mmsa.analysis.audit_conditions --root outputs/mmsa \
      --output outputs/mmsa/condition_audit.json --markdown outputs/mmsa/condition_audit.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ARCH_ORDER = ("late_fusion", "lmf", "cross_modal", "misa", "kuda", "magbert", "selfmm", "almt", "cormult", "temoe")
DISPLAY = {
    "late_fusion": "Late Fusion", "lmf": "LMF", "cross_modal": "Cross-Modal",
    "misa": "MISA-lite", "kuda": "KuDA-lite", "magbert": "MAG-BERT-lite",
    "selfmm": "Self-MM-lite", "almt": "ALMT-lite", "cormult": "CorMulT-lite",
    "temoe": "TeMoE (ours)",
}

# The numbers currently printed in TeMoE_ICASSP_paper.tex, Table 1.  Kept here so
# that a regeneration of the results can be diffed against the submitted table.
PAPER_TABLE = {
    "mosi": {
        "late_fusion": (0.927, 0.702), "lmf": (0.932, 0.711), "cross_modal": (0.909, 0.711),
        "misa": (0.937, 0.722), "kuda": (0.918, 0.698), "magbert": (0.923, 0.717),
        "selfmm": (0.941, 0.714), "almt": (0.927, 0.683),
        "cormult": (0.918, 0.702), "temoe": (0.924, 0.703),
    },
    "mosei": {
        "late_fusion": (0.565, 0.519), "lmf": (0.567, 0.519), "cross_modal": (0.676, 0.552),
        "misa": (0.558, 0.515), "kuda": (0.563, 0.518), "magbert": (0.560, 0.524),
        "selfmm": (0.568, 0.517), "almt": (0.575, 0.521),
        "cormult": (0.610, 0.550), "temoe": (0.556, 0.524),
    },
    "sims": {
        "late_fusion": (0.438, 0.387), "lmf": (0.432, 0.382),
        "misa": (0.436, 0.391), "kuda": (0.432, 0.389), "magbert": (0.445, 0.385),
        "selfmm": (0.438, 0.391), "temoe": (0.454, 0.396),
    },
}

_SEED = re.compile(r"seed[_-]?(\d+)")

# Only the multiseed sweeps are candidates for the crossing table.  Ablation and
# Plan-B directories reuse arch names ("full" is TeMoE, and the mispaired-cache
# control is TeMoE too), so grouping by arch alone silently merges them -- the
# first run of this audit did exactly that and produced a wrong TeMoE row.
CROSSING_DIRS = {"multiseed", "multiseed_interp", "multiseed_nointerp"}
CONTROL_LABELS = {
    "multiseed_shuffled": "mispaired cache (control)",
    "ablation_aux": "load-balancing sweep, no explanation (control)",
}
# CH-SIMS never got a `multiseed_nointerp/temoe` sweep, but the ablation's
# `no_interpretation` variant *is* TeMoE with its explanation branch removed --
# the same condition.  It is admitted only as a labelled fallback, never
# silently, so the provenance of that one number stays visible.
FALLBACK_KEYS = {"ablation/no_interpretation": ("temoe", False)}
# A configuration whose correlation with the label collapses to zero has not
# learned anything; CH-SIMS cross_modal and the two reproduced transformers do
# exactly this under the shared budget, and the paper excludes them there.
_DIVERGED_CORR = 0.05


def _iter_runs(dataset_root: Path) -> list[dict]:
    runs: list[dict] = []
    for report in sorted(dataset_root.rglob("*_report.json")):
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "use_interpretation" not in payload or "arch" not in payload:
            continue
        match = _SEED.search(report.parent.name)
        test = payload.get("test") or {}
        runs.append({
            "arch": str(payload["arch"]),
            "seed": int(match.group(1)) if match else None,
            "use_interp": bool(payload["use_interpretation"]),
            "params": payload.get("params"),
            "backend": payload.get("temporal_backend"),
            # Recorded since the channel-path bug: a TeMoE run built with
            # use_channels=True on a single-vector cache silently discards the
            # explanation (model.forward reads only the channel tensors there), so
            # such a run looks like an explanation-free control while claiming to
            # be an explanation run.  Old reports predate the field -> None.
            "use_channels": payload.get("use_channels"),
            "use_interp_bypass": payload.get("use_interp_bypass"),
            "dir": str(report.parent.parent.relative_to(dataset_root)),
            "mae": test.get("mae"),
            "corr": test.get("corr"),
            "acc7": test.get("acc7"),
            "diverged": bool(test.get("corr") is not None and abs(float(test["corr"])) < _DIVERGED_CORR),
        })
    return runs


def _group_key(arch: str, use_interp: bool) -> str:
    return f"{arch}|{'interp' if use_interp else 'nointerp'}"


def _prefer(group_a: list[dict], group_b: list[dict]) -> list[dict]:
    """Choose between duplicate groups for one (arch, condition): most seeds wins."""
    if len(group_a) != len(group_b):
        return group_a if len(group_a) > len(group_b) else group_b
    a_dir, b_dir = group_a[0]["dir"], group_b[0]["dir"]
    if len(a_dir) != len(b_dir):
        return group_a if len(a_dir) < len(b_dir) else group_b
    return group_a if a_dir <= b_dir else group_b


def _summarise(chosen: list[dict]) -> dict:
    mae = [r["mae"] for r in chosen if isinstance(r["mae"], (int, float))]
    corr = [r["corr"] for r in chosen if isinstance(r["corr"], (int, float))]
    acc7 = [r["acc7"] for r in chosen if isinstance(r["acc7"], (int, float))]
    return {
        "n_seeds": len(chosen),
        "seeds": sorted(r["seed"] for r in chosen if r["seed"] is not None),
        # per-seed MAE, so downstream paired statistics use exactly the same run
        # selection as the table the paper prints
        "per_seed_mae": {str(r["seed"]): r["mae"] for r in chosen if r["seed"] is not None},
        "mae_mean": statistics.fmean(mae) if mae else None,
        "mae_std": statistics.stdev(mae) if len(mae) > 1 else 0.0,
        "corr_mean": statistics.fmean(corr) if corr else None,
        "acc7_mean": statistics.fmean(acc7) if acc7 else None,
        "diverged": all(r["diverged"] for r in chosen),
    }


def audit(dataset: str, dataset_root: Path) -> dict:
    runs = _iter_runs(dataset_root)
    crossing_runs: list[dict] = []
    fallback: dict[str, list[dict]] = {}
    controls: dict[str, list[dict]] = {}
    excluded: dict[str, int] = {}
    for run in runs:
        top = Path(run["dir"]).parts[0] if run["dir"] else ""
        rel = (run["dir"] or "").replace("\\", "/")
        if rel in FALLBACK_KEYS:
            fallback.setdefault(_group_key(*FALLBACK_KEYS[rel]), []).append(run)
        elif top in CROSSING_DIRS:
            crossing_runs.append(run)
        elif top in CONTROL_LABELS:
            controls.setdefault(top, []).append(run)
        else:
            excluded[top] = excluded.get(top, 0) + 1

    runs = crossing_runs
    by_dir: dict[str, list[dict]] = {}
    for run in runs:
        by_dir.setdefault(run["dir"], []).append(run)

    # group per (arch, condition) across the whole dataset, then resolve duplicates
    candidates: dict[str, dict[str, list[dict]]] = {}
    for run in runs:
        key = _group_key(run["arch"], run["use_interp"])
        candidates.setdefault(key, {}).setdefault(run["dir"], []).append(run)

    resolved: dict[str, dict] = {}
    duplicates: list[dict] = []
    for key, dirs in candidates.items():
        groups = list(dirs.values())
        chosen = groups[0]
        for other in groups[1:]:
            chosen = _prefer(chosen, other)
        if len(groups) > 1:
            duplicates.append({
                "key": key,
                "dirs": sorted(dirs.keys()),
                "chosen": chosen[0]["dir"],
                "identical_mae": len({round(float(g[0]["mae"]), 4) for g in groups if g[0].get("mae") is not None}) == 1,
            })
        resolved[key] = {"arch": chosen[0]["arch"], "use_interp": chosen[0]["use_interp"],
                         "dir": chosen[0]["dir"], "source": "multiseed sweep", **_summarise(chosen)}

    for key, group in fallback.items():
        if key in resolved:
            continue
        resolved[key] = {"arch": group[0]["arch"], "use_interp": group[0]["use_interp"],
                         "dir": group[0]["dir"], "source": "ablation no_interpretation variant (fallback)",
                         **_summarise(group)}

    mixed = sorted(name for name, group in by_dir.items()
                   if {r["use_interp"] for r in group} == {True, False})

    control_rows = []
    for top, group in sorted(controls.items()):
        for arch, subgroup in sorted({a: [r for r in group if r["arch"] == a]
                                      for a in {r["arch"] for r in group}}.items()):
            control_rows.append({"control": CONTROL_LABELS[top], "dir": f"{top}/{arch}",
                                 "arch": arch, "use_interp": subgroup[0]["use_interp"],
                                 **_summarise(subgroup)})

    expected = max((v["n_seeds"] for v in resolved.values()), default=0)
    incomplete = sorted(f"{v['arch']} ({'interp' if v['use_interp'] else 'nointerp'}): "
                        f"{v['n_seeds']}/{expected} seeds in {v['dir']}"
                        for v in resolved.values() if v["n_seeds"] < expected)

    return {
        "dataset": dataset,
        "n_runs": len(runs),
        "expected_seeds": expected,
        "conditions": {f"{arch}|{cond}": resolved.get(f"{arch}|{cond}")
                       for arch in ARCH_ORDER for cond in ("nointerp", "interp")},
        "mixed_folders": mixed,
        "duplicate_groups": duplicates,
        "incomplete": incomplete,
        "controls": control_rows,
        "excluded_runs_by_dir": dict(sorted(excluded.items())),
    }


def _cell(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def crossing_markdown(report: dict) -> list[str]:
    conds = report["conditions"]
    lines = ["| architecture | MAE w/o expl. | MAE w/ expl. | delta | Corr w/o | Corr w/ | seeds |",
             "|---|---|---|---|---|---|---|"]
    fallback_used = False
    for arch in ARCH_ORDER:
        without = conds.get(f"{arch}|nointerp")
        with_ = conds.get(f"{arch}|interp")
        if not without or not with_:
            missing = "without" if not without else "with"
            lines.append(f"| {DISPLAY.get(arch, arch)} | -- | -- | -- | -- | -- | _{missing} run missing_ |")
            continue
        is_fallback = "fallback" in str(without.get("source", ""))
        fallback_used = fallback_used or is_fallback
        name = DISPLAY.get(arch, arch) + (r"$^\dagger$" if is_fallback else "")
        if without.get("diverged") or with_.get("diverged"):
            lines.append(f"| {name} | {_cell(without['mae_mean'])} | {_cell(with_['mae_mean'])} | "
                         f"diverged | -- | -- | {with_['n_seeds']} |")
            continue
        delta = with_["mae_mean"] - without["mae_mean"]
        lines.append("| {} | {} | {} | {:+.3f} | {} | {} | {} |".format(
            name, _cell(without["mae_mean"]), _cell(with_["mae_mean"]),
            delta, _cell(without["corr_mean"]), _cell(with_["corr_mean"]), with_["n_seeds"]))
    lines.append("")
    if fallback_used:
        lines += [r"$\dagger$ the explanation-free number comes from the `ablation/no_interpretation`",
                  "variant -- TeMoE with its explanation branch removed, i.e. the same condition --",
                  "because no dedicated multiseed explanation-free sweep exists for that dataset.", ""]
    return lines


def control_markdown(report: dict) -> list[str]:
    """Controls kept out of the crossing table but still worth reporting."""
    rows = report.get("controls") or []
    if not rows:
        return []
    lines = ["**Control runs (not part of the crossing table)**", "",
             "| control | run | condition | MAE | Corr | seeds |", "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| {} | `{}` | {} | {} | {} | {} |".format(
            row["control"], row["dir"], "w/ expl." if row["use_interp"] else "w/o expl.",
            _cell(row["mae_mean"]), _cell(row["corr_mean"]), row["n_seeds"]))
    return lines + [""]


def paper_check(report: dict) -> list[str]:
    """Diff the audited crossing table against the values printed in the paper."""
    dataset = report["dataset"]
    expected = PAPER_TABLE.get(dataset)
    if not expected:
        return []
    lines = [f"| architecture | paper w/o | audit w/o | paper w/ | audit w/ | verdict |", "|---|---|---|---|---|---|"]
    ok = True
    for arch, (paper_without, paper_with) in expected.items():
        without = report["conditions"].get(f"{arch}|nointerp")
        with_ = report["conditions"].get(f"{arch}|interp")
        if not without or not with_:
            lines.append(f"| {DISPLAY.get(arch, arch)} | {paper_without:.3f} | missing | {paper_with:.3f} | missing | FAIL |")
            ok = False
            continue
        note = "ok (fallback source)" if "fallback" in str(without.get("source", "")) else "ok"
        got_without = round(float(without["mae_mean"]), 3)
        got_with = round(float(with_["mae_mean"]), 3)
        match = abs(got_without - paper_without) <= 0.001 and abs(got_with - paper_with) <= 0.001
        ok = ok and match
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
            DISPLAY.get(arch, arch), paper_without, got_without, paper_with, got_with,
            note if match else "MISMATCH"))
    lines += ["", f"**Paper-vs-raw-data verification: {'PASS' if ok else 'FAIL'}**", ""]
    report["paper_check_passed"] = ok
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit run conditions and rebuild the crossing table.")
    parser.add_argument("--root", default="outputs/mmsa")
    parser.add_argument("--datasets", nargs="+", default=["mosi", "sims", "mosei"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    reports = {}
    md: list[str] = ["# Condition audit (from raw run reports)", "",
                     "Condition, not folder name, is read from each run's `use_interpretation`",
                     "flag in `<arch>_report.json`, so a mixed folder cannot mis-attribute a",
                     "number. `delta` is the MAE change from adding the cached explanation.", ""]
    for dataset in args.datasets:
        data_root = root / dataset
        if not data_root.is_dir():
            continue
        report = audit(dataset, data_root)
        reports[dataset] = report
        md += [f"## {dataset.upper()}  ({report['n_runs']} runs, {report['expected_seeds']} seeds)", ""]
        md += crossing_markdown(report)
        anomalies = []
        if dataset in PAPER_TABLE:
            md += ["**Verification against the submitted table**", ""] + paper_check(report)
        if report["mixed_folders"]:
            anomalies.append("- **mixed folders** (both conditions inside one directory): "
                             + ", ".join(f"`{name}`" for name in report["mixed_folders"]))
        for dup in report["duplicate_groups"]:
            note = "identical values" if dup.get("identical_mae") else "VALUES DIFFER -- pick deliberately"
            anomalies.append(f"- duplicate run for `{dup['key']}` in {', '.join('`' + d + '`' for d in dup['dirs'])} "
                             f"(used `{dup['chosen']}`, {note})")
        for item in report["incomplete"]:
            anomalies.append(f"- incomplete: {item}")
        # invariant: on the single-vector caches used throughout this paper the
        # channel path must never be taken, or the explanation is silently lost
        flags = [r for r in _iter_runs(data_root)]
        channel_on = sorted(f"{r['arch']} ({r['dir']})" for r in flags if r.get("use_channels") is True)
        unknown = sum(1 for r in flags if r.get("use_channels") is None)
        if channel_on:
            anomalies.append("- **CHANNEL PATH ON** -- the explanation is discarded in these runs, "
                             "they must be rerun: " + ", ".join(channel_on))
        if unknown:
            recorded = len(flags) - unknown
            anomalies.append(f"- `use_channels` coverage: {recorded}/{len(flags)} runs record the flag "
                             f"(all recorded values are {sorted({r.get('use_channels') for r in flags if r.get('use_channels') is not None})}); "
                             f"{unknown} predate the field, i.e. were written before the channel-path fix")
        excluded = report.get("excluded_runs_by_dir") or {}
        if excluded:
            anomalies.append("- out of scope for the crossing table: " + ", ".join(
                f"`{name}` ({count} runs)" for name, count in excluded.items()))
        if anomalies:
            md += ["**Anomalies and scope**", ""] + anomalies + [""]
        md += control_markdown(report)
    md += ["**How to read this.** A row appears only when both conditions were run for that",
           "architecture, so a partially finished sweep shows explicit `missing` cells instead of",
           "silently mixing conditions -- which is precisely the error this audit exists to catch.", ""]

    text = "\n".join(md) + "\n"
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    if args.markdown:
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    if not args.output and not args.markdown:
        print(text)


if __name__ == "__main__":
    main()