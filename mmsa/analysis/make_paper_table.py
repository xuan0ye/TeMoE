"""Emit the main-results table body straight from the verified run reports.

Hand-transcribing numbers into a submission table is where integrity bugs come
from -- the previous revision of this paper lost two architectures that way.
This generator reads each run's own ``use_interpretation`` flag (via the audit),
recomputes mean MAE per architecture/condition, and prints the LaTeX row block,
the header, and the caption's delta ranges, so the table cannot drift from the
raw data.  It also refuses to emit a dataset whose two conditions are missing or
whose seed count is short, and says so.

Layout (chosen for the 4-page budget): architecture rows x three datasets x
(without, with) = 7 columns; per-dataset delta ranges live in the caption, so
adding a dataset costs no table height.

Usage:
  python -m mmsa.analysis.make_paper_table --root outputs/mmsa --datasets mosi sims mosei
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mmsa.analysis.audit_conditions import DISPLAY, audit

ORDER = ("late_fusion", "lmf", "cross_modal", "misa", "kuda", "magbert",
         "selfmm", "almt", "cormult", "temoe")
DAGGER = {"misa", "kuda", "magbert", "selfmm", "almt", "cormult"}
DATASET_LABEL = {"mosi": "CMU-MOSI", "sims": "CH-SIMS", "mosei": "CMU-MOSEI"}


def collect(root: Path, datasets: list[str]) -> tuple[dict, list[str]]:
    data, problems = {}, []
    for ds in datasets:
        if not (root / ds).is_dir():
            problems.append(f"{ds}: no results directory under {root}")
            continue
        report = audit(ds, root / ds)
        need = report["expected_seeds"]
        ok, missing = {}, []
        for arch in ORDER:
            w = report["conditions"].get(f"{arch}|nointerp")
            i = report["conditions"].get(f"{arch}|interp")
            # A configuration that ran and diverged numerically is reportable --
            # it belongs in the table as "-- (diverges)", not as missing data.
            if w is not None and i is None and w.get("diverged"):
                ok[arch] = {"w": w, "i": None, "diverged": True}
                continue
            if not w or not i:
                missing.append(arch)
                continue
            if w["n_seeds"] < need or i["n_seeds"] < need:
                missing.append(f"{arch}(seeds {w['n_seeds']}/{i['n_seeds']} of {need})")
                continue
            if w.get("diverged") or i.get("diverged"):
                ok[arch] = {"w": w, "i": i, "diverged": True}
                continue
            ok[arch] = {"w": w, "i": i, "diverged": False}
        if missing:
            problems.append(f"{ds}: incomplete -> {', '.join(missing)}")
        if ok:
            data[ds] = (report, ok)
    return data, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/mmsa")
    ap.add_argument("--datasets", nargs="+", default=["mosi", "sims", "mosei"])
    args = ap.parse_args()
    root = Path(args.root)
    data, problems = collect(root, args.datasets)

    if problems:
        print("% ---- INCOMPLETE DATASETS (do not ship a table built from a subset quietly) ----")
        for p in problems:
            print(f"%   {p}")
        print()

    complete = [ds for ds in args.datasets if ds in data]
    if not complete:
        print("% no dataset has both conditions at full seed count yet")
        return

    ncol = 1 + 2 * len(complete)
    print(f"% ===== Table 1 body: {ncol} columns ({len(complete)} datasets) =====")
    print(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + "l" + "cc" * len(complete) + "@{}}")
    print(r"\toprule")
    print("& " + " & ".join(
        r"\multicolumn{2}{c}{" + DATASET_LABEL[ds] + "}" for ds in complete) + r" \\")
    print(" ".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * k, 3 + 2 * k) for k in range(len(complete))))
    print("Architecture & " + " & ".join(["w/o expl. & w/ expl."] * len(complete)) + r" \\")
    print(r"\midrule")

    # best explained MAE per dataset, for bolding
    best = {}
    for ds in complete:
        _, rows = data[ds]
        vals = [(a, rows[a]["i"]["mae_mean"]) for a in rows if not rows[a]["diverged"]]
        if vals:
            best[ds] = min(vals, key=lambda t: t[1])[0]

    for arch in ORDER:
        if not any(arch in data[ds][1] for ds in complete):
            continue
        cells, name = [], DISPLAY.get(arch, arch) + (r"$^\dagger$" if arch in DAGGER else "")
        if arch == "temoe":
            name = r"\textbf{TeMoE (ours)}"
        for ds in complete:
            _, rows = data[ds]
            if arch not in rows:
                cells.append(r"\multicolumn{2}{c}{--}")
                continue
            entry = rows[arch]
            if entry["diverged"]:
                cells.append(r"\multicolumn{2}{c}{-- (diverges)}")
                continue
            w, i = entry["w"], entry["i"]
            with_txt = f"{i['mae_mean']:.3f}"
            if best.get(ds) == arch:
                with_txt = r"$\mathbf{" + with_txt + "}$"
            cells.append(f"{w['mae_mean']:.3f} & {with_txt}")
        print(f"{name} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular*}")

    print("\n% ===== caption: per-dataset delta ranges, best explained, converging count =====")
    total = 0
    for ds in complete:
        _, rows = data[ds]
        conv = [a for a in rows if not rows[a]["diverged"]]
        total += len(conv)
        d = [rows[a]["i"]["mae_mean"] - rows[a]["w"]["mae_mean"] for a in conv]
        ex = [rows[a]["i"]["mae_mean"] for a in conv]
        un = [rows[a]["w"]["mae_mean"] for a in conv]
        print(f"%   {DATASET_LABEL[ds]}: {len(conv)} converging configs, "
              f"delta {min(d):+.3f} to {max(d):+.3f}, "
              f"explained span {min(ex):.3f}-{max(ex):.3f} (spread {max(ex)-min(ex):.3f}), "
              f"unexplained span {min(un):.3f}-{max(un):.3f}, "
              f"best explained = {min(((rows[a]['i']['mae_mean'], a) for a in conv))[1]}")
    print(f"%   TOTAL converging configurations = {total}")
    seeds = {data[ds][0]['expected_seeds'] for ds in complete}
    print(f"%   seed counts: {sorted(seeds)}  (must be uniform across datasets for a paired t-test claim)")


if __name__ == "__main__":
    main()