"""R3 control-cache comparison: matched vs vision-only vs text-only vs none.

R3 asks the question the paper's pitch depends on: is the gain about the VIDEO, or
would any LLM-generated text do?  Three caches are trained against the same three
architectures on MOSI with the same five seeds:

    matched   data/interpretation/mosi            video + text (the paper's cache)
    videonly  data/interpretation/mosi_videonly   the clip only, transcript withheld
    textonly  data/interpretation/mosi_textonly   the transcript only, clip withheld
    none      (no cache at all)

Every number is printed WITH the file it came from, and every claim is backed by a
paired t-test over the five seeds, because a mean without provenance or a test is
not evidence.

Reading the result:
    vision-only << text-only   -> the visual grounding claim holds
    vision-only ~= text-only   -> it does NOT: any generated rationale helps, and the
                                  paper must say so instead of claiming visual grounding
    text-only ~= matched       -> the transcript alone carries the whole effect

Usage:
    python -m mmsa.analysis.control_caches
    python -m mmsa.analysis.control_caches --dataset mosi --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARCHS = ("temoe", "lmf", "late_fusion")
# ordered candidate directories per condition; the first existing file wins
CANDIDATES = {
    "matched": ["multiseed_interp", "multiseed"],
    "none": ["multiseed_nointerp"],
    "videonly": ["control_videonly"],
    "textonly": ["control_textonly"],
}


def load(dataset_root: Path, arch: str, condition: str):
    for stem in CANDIDATES.get(condition, []):
        path = dataset_root / stem / arch / f"multiseed_{arch}.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                return None, f"{path} (unreadable: {exc})"
            mean = data.get("mean") or {}
            std = data.get("std") or {}
            if "mae" not in mean:
                return None, f"{path} (no mean.mae)"
            return {
                "mae": float(mean["mae"]),
                "mae_std": float(std.get("mae", 0.0)),
                "corr": float(mean.get("corr", float("nan"))),
                "corr_std": float(std.get("corr", 0.0)),
                "acc7": float(mean.get("acc7", float("nan"))),
                "n": len(data.get("per_seed", data.get("runs", [])) or []),
                "path": str(path),
            }, None
    return None, "no file found in " + ", ".join(
        f"{s}/{arch}/multiseed_{arch}.json" for s in CANDIDATES.get(condition, []))


def per_seed_mae(path):
    """Per-seed test MAE, keyed by seed, for paired tests.

    The multiseed json has been written in two shapes over time:
    {"per_seed": [{"seed": 42, "mae": 0.71}, ...]} and {"per_seed_mae": {"42": 0.71}}.
    Both are accepted. Only seeds present in BOTH arms are paired; if that leaves
    fewer than two seeds the caller is told instead of silently zipping different
    seeds together.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    out = {}
    ps = data.get("per_seed")
    if isinstance(ps, list) and ps and isinstance(ps[0], dict):
        for row in ps:
            if "seed" in row and "mae" in row:
                out[int(row["seed"])] = float(row["mae"])
    elif isinstance(data.get("per_seed_mae"), dict):
        for seed, mae in data["per_seed_mae"].items():
            out[int(seed)] = float(mae)
    return out


def paired_stats(a, b):
    """Numeric paired comparison over shared seeds.

    Single implementation of the statistic, shared with the supplement builder so the
    two can never drift apart. Keys: n, seeds, diff, sd, se, lo, hi, t, p.
    """
    shared = sorted(set(a) & set(b))
    out = {"n": len(shared), "seeds": shared}
    if len(shared) < 2:
        return out
    da = [a[s] for s in shared]
    db = [b[s] for s in shared]
    diff = [x - y for x, y in zip(da, db)]
    n = len(diff)
    mean = sum(diff) / n
    var = sum((d - mean) ** 2 for d in diff) / (n - 1) if n > 1 else 0.0
    sd = var ** 0.5
    se = sd / (n ** 0.5) if sd else 0.0
    t_stat = float("inf") if se == 0 else mean / se
    p_val = float("nan")
    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_rel(da, db)
        t_stat, p_val = float(t_stat), float(p_val)
    except Exception:
        pass
    out.update(diff=mean, sd=sd, se=se, lo=mean - 1.96 * se, hi=mean + 1.96 * se,
               t=t_stat, p=p_val)
    return out


def paired_test(a, b, label):
    """Formatted paired t-test line; see paired_stats for the numbers."""
    s = paired_stats(a, b)
    if s["n"] < 2:
        return f"  {label:<32} cannot pair ({s['n']} shared seeds)"
    p_val = s["p"]
    if p_val != p_val:
        verdict = "p unavailable"
    elif p_val < 0.001:
        verdict = "SIGNIFICANT p<0.001"
    elif p_val < 0.05:
        verdict = "SIGNIFICANT p<0.05"
    else:
        verdict = "n.s."
    return (f"  {label:<32} diff {s['diff']:+.4f}  95% CI [{s['lo']:+.4f}, {s['hi']:+.4f}]  "
            f"t={s['t']:+.2f}  p={p_val:.4f}  {verdict}  seeds={s['seeds']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mosi")
    ap.add_argument("--root", default="outputs/mmsa")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    root = Path(args.root) / args.dataset
    if not root.is_dir():
        print("missing dataset root:", root)
        return

    conditions = ("matched", "videonly", "textonly", "none")
    table = {}
    for arch in ARCHS:
        table[arch] = {}
        for cond in conditions:
            got, why = load(root, arch, cond)
            table[arch][cond] = got if got else {"missing": why}

    print(f"=== R3 control caches -- {args.dataset.upper()} (5 seeds, mean MAE) ===")
    head = f"{'architecture':<14}" + "".join(f"{c:>22}" for c in conditions)
    print(head)
    print("-" * len(head))
    for arch in ARCHS:
        cells = []
        for cond in conditions:
            v = table[arch][cond]
            cells.append("-- (missing)" if "missing" in v else f"{v['mae']:.4f} +/- {v['mae_std']:.4f}")
        print(f"{arch:<14}" + "".join(f"{c:>22}" for c in cells))

    print("\n=== provenance (every number above) ===")
    for arch in ARCHS:
        for cond in conditions:
            v = table[arch][cond]
            if "missing" in v:
                print(f"  {arch:<12} {cond:<9} MISSING: {v['missing']}")
            else:
                print(f"  {arch:<12} {cond:<9} n={v['n']:<2} corr={v['corr']:.4f} "
                      f"acc7={v['acc7']:.2f}  <- {v['path']}")

    print("\n=== the verdict R3 exists to produce ===")
    for arch in ARCHS:
        row = table[arch]
        if any("missing" in row[c] for c in conditions):
            print(f"  {arch:<12} incomplete -- wait for R3c to finish")
            continue
        new, vo, to, mt = (row["none"]["mae"], row["videonly"]["mae"],
                           row["textonly"]["mae"], row["matched"]["mae"])
        print(f"  {arch:<12} none {new:.4f} | vision-only {vo:.4f} ({new - vo:+.4f}) | "
              f"text-only {to:.4f} ({new - to:+.4f}) | matched {mt:.4f} ({new - mt:+.4f})")
        gap = vo - to
        if abs(gap) < 0.005:
            print("               -> vision-only and text-only are within 0.005: any generated "
                  "rationale helps equally; do NOT claim visual grounding.")
        elif gap < 0:
            print(f"               -> vision-only beats text-only by {-gap:.4f}: the visual "
                  "grounding claim holds for this architecture.")
        else:
            print(f"               -> text-only beats vision-only by {gap:.4f}: the transcript "
                  "carries the effect; the visual claim does NOT hold here.")

    print("\n=== paired tests over seeds (the evidence, not just the means) ===")
    print("  negative diff = the FIRST arm has the lower (better) MAE")
    for arch in ARCHS:
        row = table[arch]
        if any("missing" in row[c] for c in conditions):
            print(f"  {arch}: incomplete")
            continue
        s = {c: per_seed_mae(row[c]["path"]) for c in conditions}
        empty = [c for c in conditions if not s[c]]
        if empty:
            print(f"  {arch}: no per-seed MAE in {empty} -- cannot pair")
            continue
        print(f"\n  --- {arch} ---")
        print(paired_test(s["videonly"], s["none"], "vision-only vs none"))
        print(paired_test(s["textonly"], s["none"], "text-only vs none"))
        print(paired_test(s["videonly"], s["textonly"], "vision-only vs text-only"))
        print(paired_test(s["matched"], s["textonly"], "matched vs text-only"))

    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()