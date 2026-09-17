"""Cross-check the official MMSA CSVs against MMSA's own per-seed logs.

Why this exists: MMSA writes two different representations of the same run.

    logs/<model>-<dataset>.log   "Result for seed 4: {...'MAE': 0.9466, 'Corr': 0.666}"
    results/<mode>/<dataset>.csv "lmf,...,(np.float64(105.29), np.float64(18.29)),
                                  ...(np.float64(55.06), np.float64(22.55))..."

The log values are raw (MAE ~0.95). The CSV values are 100x the 5-seed mean
(105.29). Nothing in the CSV says so, and copying 105.29 into a paper table would
be an obvious error, while blindly dividing everything by 100 could equally hide a
genuinely diverged run. So: recompute mean/std from the per-seed log lines and
compare. If csv/100 == log mean, the scaling is PROVEN rather than assumed, and
the per-seed spread tells us whether one seed is dragging the mean.

Usage:
    python -m mmsa.analysis.official_logs_check --root official_runs
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics as st
from pathlib import Path

RESULT = re.compile(r"Result for seed\s+(\d+):\s*(\{.*\})")
KV = re.compile(r"'([A-Za-z0-9_]+)':\s*(?:np\.float64\()?([-+0-9.eE]+)")
CELL = re.compile(r"\(np\.float64\(([-+0-9.eE]+)\)")
METRICS = ("MAE", "Corr", "Mult_acc_7", "Has0_acc_2")


def per_seed(log):
    out = {}
    for line in log.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        m = RESULT.search(line)
        if not m:
            continue
        out[int(m.group(1))] = {k: float(v) for k, v in KV.findall(m.group(2))}
    return out


def csv_first(path):
    """{model: {metric: mean}} from a single MMSA csv, keeping the raw printout."""
    out = {}
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        for raw in csv.DictReader(fh):
            model = (raw.get("Model") or "").strip()
            if not model:
                continue
            row = {}
            for name, cell in raw.items():
                if not name or name == "Model" or cell is None:
                    continue
                m = CELL.search(cell)
                if m:
                    row[name] = float(m.group(1))
            out[model] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="official_runs")
    args = ap.parse_args()
    root = Path(args.root)

    logdirs = sorted(p for p in root.rglob("logs") if p.is_dir())
    if not logdirs:
        print("no logs/ directory under " + str(root))
        return

    verdicts = []
    for logdir in logdirs:
        tag = logdir.relative_to(root).parts[0]
        for log in sorted(logdir.glob("*.log")):
            stem = log.stem
            if "-" not in stem:
                continue
            model, dataset = stem.rsplit("-", 1)
            seeds = per_seed(log)
            print("\n=== " + str(log) + "   (" + str(len(seeds)) + " seed result lines)")
            if not seeds:
                print("      no 'Result for seed' lines -- the run never reached test")
                continue
            for name in METRICS:
                vals = [s[name] for s in seeds.values() if name in s]
                if not vals:
                    continue
                mean = st.fmean(vals)
                sd = st.stdev(vals) if len(vals) > 1 else 0.0
                flag = ""
                if name in ("MAE", "Corr") and sd > 0.5 * abs(mean):
                    flag = "   <-- huge spread, check the seeds below"
                print("      {0:12} per-seed {1}".format(
                    name, " ".join("{0:.4f}".format(v) for v in vals)))
                print("      {0:12} mean {1:.4f}  std {2:.4f}{3}".format("", mean, sd, flag))

            # rev 8 moved results to <root>/<tag>/results/<dataset>/<model>/<mode>/.
            # Search both that layout and the older flat one, and when several copies
            # exist prefer the one that sits under .../<dataset>/<model>/ -- otherwise
            # we could compare this model's log against another model's csv.
            cands = []
            for base in (logdir.parent / "results", root):
                if base.is_dir():
                    cands += [p for p in base.rglob(dataset + ".csv")]
            pref = [p for p in cands
                    if (os.sep + dataset + os.sep + model + os.sep) in str(p)]
            csvpath = (pref or cands or [None])[0]
            if csvpath is None:
                print("      (no csv found for " + dataset + ")")
                continue
            rows = csv_first(csvpath)
            row = None
            for key in rows:
                if key.lower() == model.lower():
                    row = rows[key]
                    break
            if row is None:
                print("      (" + model + " not in " + str(csvpath) + ": " + str(sorted(rows)) + ")")
                continue
            print("      --- csv " + str(csvpath))
            for name in ("MAE", "Corr"):
                if name not in row:
                    continue
                raw = row[name]
                vals = [s[name] for s in seeds.values() if name in s]
                logmean = st.fmean(vals)
                ok = abs(raw / 100.0 - logmean) < 0.005
                verdicts.append(ok)
                print("      {0:12} csv {1:>10.2f}   csv/100 {2:.4f}   log mean {3:.4f}   {4}".format(
                    name, raw, raw / 100.0, logmean,
                    "MATCH -> the csv is 100x the mean" if ok else "MISMATCH <-- investigate"))

    print("\n=== verdict ===")
    if not verdicts:
        print("  nothing was compared -- no csv/log pair matched. Check the layout:")
        print("  <root>/<tag>/logs/<model>-<dataset>.log  and  <root>/<tag>/results/<mode>/<dataset>.csv")
    elif all(verdicts):
        print("  every checked column satisfies csv == 100 * log-mean.")
        print("  -> divide MMSA csv values by 100 before they go anywhere near a table.")
    else:
        print("  " + str(verdicts.count(False)) + "/" + str(len(verdicts)) +
              " columns do NOT satisfy csv == 100 * log-mean.")
        print("  -> do not transcribe the csv numbers; use the per-seed log lines above.")


if __name__ == "__main__":
    main()