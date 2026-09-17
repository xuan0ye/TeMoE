"""Turn the official MMSA result CSVs into a paper-ready table.

MMSA writes results/<mode>/<dataset>.csv with one row per model and cells like

    "(np.float64(71.75), np.float64(10.67))"

Two traps this script exists to absorb:

1. The cells are not plain floats: they are (mean, std) tuples printed with numpy
   reprs. `csv` keeps them intact (the field is quoted, so the inner comma is
   safe) but they still need a regex.

2. MMSA multiplies the regression metrics by 100. The LMF/MOSI row comes out as
   MAE 105.29, Corr 55.06. A correlation of 55 is impossible -- Corr is bounded by
   1 -- so that column is certainly * 100. MAE sits in the same row under the same
   formatting, and an MAE of 105 is impossible for labels in [-3, 3] while Acc-2
   is a healthy 71.75%, so it is scaled the same way. Nothing here guesses
   silently: raw and divided values are BOTH printed and each row is flagged.

Usage:
    python -m mmsa.analysis.official_mmsa_report --root official_runs
    python -m mmsa.analysis.official_mmsa_report --root official_runs \
        --markdown official_runs/official_baselines.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

CELL = re.compile(
    r"\(np\.float64\((?P<mean>[-+0-9.eE]+)\),\s*np\.float64\((?P<std>[-+0-9.eE]+)\)\)"
)
SCALED = ("MAE", "Corr", "Loss")


def parse_cell(cell):
    m = CELL.search(cell)
    if m:
        return float(m.group("mean")), float(m.group("std"))
    try:
        return float(cell), None
    except (TypeError, ValueError):
        return None, None


def looks_scaled(name, mean):
    if mean is None or name not in SCALED:
        return False
    if name == "Corr":
        return abs(mean) > 1.5
    if name == "MAE":
        return abs(mean) > 3.0
    return False


def read_csv(path):
    rows = []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        for raw in csv.DictReader(fh):
            model = (raw.get("Model") or "").strip()
            if not model:
                continue
            entry = {"model": model, "csv": str(path), "metrics": {}}
            for name, cell in raw.items():
                if name is None or name == "Model" or cell is None:
                    continue
                mean, std = parse_cell(cell)
                if mean is None:
                    continue
                scaled = looks_scaled(name, mean)
                entry["metrics"][name] = {
                    "raw": mean,
                    "raw_std": std,
                    "value": mean / 100.0 if scaled else mean,
                    "std": std / 100.0 if (scaled and std is not None) else std,
                    "scale_100": scaled,
                }
            rows.append(entry)
    return rows


def summary_rows(rows, dataset, tag):
    out = []
    for r in rows:
        m = r["metrics"]

        def g(key):
            return m.get(key, {}).get("value")

        def s(key):
            return m.get(key, {}).get("std")

        out.append({
            "tag": tag,
            "dataset": dataset,
            "model": r["model"],
            "mae": g("MAE"), "mae_std": s("MAE"),
            "corr": g("Corr"), "corr_std": s("Corr"),
            "acc7": g("Mult_acc_7"), "acc2_has0": g("Has0_acc_2"),
            "f1_has0": g("Has0_F1_score"),
            "scaled": sorted(k for k, v in m.items() if v["scale_100"]),
            "csv": r["csv"],
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="official_runs")
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    csvs = sorted(p for p in root.rglob("*.csv") if p.is_file())
    if not csvs:
        print("no CSVs under " + str(root) + "/ -- has the sweep produced results yet?")
        return

    collected = []
    print("=== " + str(len(csvs)) + " result CSV(s) under " + str(root) + " ===")
    for path in csvs:
        rel = path.relative_to(root)
        tag = rel.parts[0] if len(rel.parts) > 1 else "?"
        rows = read_csv(path)
        print("\n--- " + str(path) + "   (tag=" + tag + ")")
        if not rows:
            print("      (no model rows)")
            continue
        for r in rows:
            m = r["metrics"]
            scaled = [k for k, v in m.items() if v["scale_100"]]
            print("    " + r["model"])
            for key in ("Has0_acc_2", "Mult_acc_7", "MAE", "Corr"):
                if key not in m:
                    continue
                v = m[key]
                note = "  <-- divided by 100" if v["scale_100"] else ""
                sd = v["std"]
                sd_s = "None" if sd is None else str(round(sd, 4))
                print("        {0:12} raw={1:>10.2f}  -> {2:.4f}  (std {3}){4}".format(
                    key, v["raw"], v["value"], sd_s, note))
            if scaled:
                print("        (MMSA printed " + str(scaled) + " at 100x; values above are the real ones)")
        collected += summary_rows(rows, path.stem, tag)

    if not collected:
        return

    print("\n=== paper-ready rows ===")
    print("{0:<10} {1:<7} {2:<10} {3:>8} {4:>7} {5:>7} {6:>7}".format(
        "tag", "dataset", "model", "MAE", "+-", "Corr", "Acc-7"))
    for r in sorted(collected, key=lambda x: (x["dataset"], x["mae"] or 9e9)):
        mae = "-" if r["mae"] is None else "{0:.4f}".format(r["mae"])
        sd = "-" if r["mae_std"] is None else "{0:.4f}".format(r["mae_std"])
        corr = "-" if r["corr"] is None else "{0:.4f}".format(r["corr"])
        a7 = "-" if r["acc7"] is None else "{0:.2f}".format(r["acc7"])
        print("{0:<10} {1:<7} {2:<10} {3:>8} {4:>7} {5:>7} {6:>7}".format(
            r["tag"], r["dataset"], r["model"], mae, sd, corr, a7))

    print("\n=== sanity ===")
    bad = [r for r in collected if r["corr"] is not None and not (-1.01 <= r["corr"] <= 1.01)]
    tail = "  <-- scaling assumption is WRONG, investigate" if bad else "  (ok)"
    print("  correlations outside [-1, 1] after unscaling : " + str(len(bad)) + tail)
    print("  rows with no MAE at all                      : " + str(len([r for r in collected if r["mae"] is None])))

    if args.json:
        Path(args.json).write_text(json.dumps(collected, indent=2), encoding="utf-8")
        print("\nwrote " + args.json)
    if args.markdown:
        lines = ["| pkl | dataset | model | MAE | Corr | Acc-7 |", "|---|---|---|---|---|---|"]
        for r in sorted(collected, key=lambda x: (x["dataset"], x["mae"] or 9e9)):
            mae = "-" if r["mae"] is None else "{0:.4f} +- {1:.4f}".format(r["mae"], r["mae_std"])
            corr = "-" if r["corr"] is None else "{0:.4f}".format(r["corr"])
            a7 = "-" if r["acc7"] is None else "{0:.2f}".format(r["acc7"])
            lines.append("| {0} | {1} | {2} | {3} | {4} | {5} |".format(
                r["tag"], r["dataset"], r["model"], mae, corr, a7))
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("wrote " + args.markdown)


if __name__ == "__main__":
    main()