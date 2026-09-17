"""Paired statistics for the crossed configurations, using the audit's own run selection.

Addresses the reviewer concern that the paper reports 5-seed means with no per-pair
confidence interval, no paired test and no multiple-comparison correction, and that the
mispaired-cache control was never tested against no-explanation.
"""
import json, statistics as st
from math import sqrt
from scipy import stats as sps

j = json.load(open("outputs/mmsa/condition_audit.json", encoding="utf-8"))
ARCHS = ("late_fusion", "lmf", "cross_modal", "misa", "kuda", "magbert", "selfmm", "almt", "cormult", "temoe")
NAME = {"late_fusion": "Late Fusion", "lmf": "LMF", "cross_modal": "Cross-Modal", "misa": "MISA-lite",
        "kuda": "KuDA-lite", "magbert": "MAG-BERT-lite", "selfmm": "Self-MM-lite",
        "almt": "ALMT-lite", "cormult": "CorMulT-lite", "temoe": "TeMoE (ours)"}

def pair(ds, arch):
    c = j[ds]["conditions"]
    w, i = c.get(f"{arch}|nointerp"), c.get(f"{arch}|interp")
    if not w or not i:
        return None
    if w.get("diverged") or i.get("diverged"):
        return None
    a, b = i["per_seed_mae"], w["per_seed_mae"]
    seeds = sorted(set(a) & set(b), key=int)
    if len(seeds) < 5:
        return None
    d = [a[s] - b[s] for s in seeds]
    m, sd, n = st.fmean(d), st.stdev(d), len(d)
    t = m / (sd / sqrt(n)) if sd else float("inf")
    p = 2 * sps.t.sf(abs(t), n - 1)
    half = sps.t.ppf(0.975, n - 1) * sd / sqrt(n)
    return {"ds": ds, "arch": arch, "m": m, "sd": sd, "t": t, "p": p,
            "ci": (m - half, m + half), "worse": sum(1 for x in d if x > 0), "n": n}

rows = [r for ds in ("mosi", "sims", "mosei") for a in ARCHS if (r := pair(ds, a))]
k = len(rows)
rows.sort(key=lambda r: r["p"])
for idx, r in enumerate(rows):
    r["holm"] = r["p"] <= 0.05 / (k - idx) and all(x.get("holm", True) for x in rows[:idx])

allci = all(r["ci"][1] < 0 for r in rows)
print(f"=== {k} crossed architecture--dataset pairs (paired by seed, 5 seeds each) ===")
print(f"  every 95% CI for the mean gain excludes zero : {allci}")
print(f"  significant at 0.05 uncorrected              : {sum(1 for r in rows if r['p'] < 0.05)}/{k}")
print(f"  significant after Holm correction            : {sum(1 for r in rows if r['holm'])}/{k}")
print(f"  seeds where the cached arm is worse          : {sum(r['worse'] for r in rows)}/{sum(r['n'] for r in rows)}")
print(f"  gain range (point estimates)                 : {min(r['m'] for r in rows):+.3f} to {max(r['m'] for r in rows):+.3f}")
print(f"  widest CI half-width                         : {max((r['ci'][1]-r['ci'][0])/2 for r in rows):.4f}")
print()
out = ["| dataset | architecture | mean MAE change | 95% CI | t | p | Holm |",
       "|---|---|---|---|---|---|---|"]
for r in rows:
    out.append("| {} | {} | {:+.3f} | [{:+.3f}, {:+.3f}] | {:.1f} | {:.4f} | {} |".format(
        r["ds"].upper(), NAME[r["arch"]], r["m"], r["ci"][0], r["ci"][1], r["t"], r["p"],
        "yes" if r["holm"] else "no"))
print("\n".join(out))
open("outputs/mmsa/crossing_paired_stats.md", "w", encoding="utf-8", newline="\n").write(
    "# Paired statistics for the crossed configurations\n\n"
    "Paired by seed over the same five seeds; CI is a 95% Student-t interval for the mean MAE\n"
    "change (cached minus no-cache, so negative = the cache helps), and Holm is Holm-Bonferroni\n"
    "over all {} pairs. Diverging configurations (CH-SIMS Cross-Modal/ALMT/CorMulT) are excluded,\n"
    "as in the paper.\n\n".format(k) + "\n".join(out) + "\n")
print("\nwrote outputs/mmsa/crossing_paired_stats.md")

# ---- the control the reviewer says was never tested ----
print()
print("=== MOSI controls ===")
c = j["mosi"]["conditions"]
ctl = {row["arch"]: row for row in j["mosi"]["controls"] if "shuffled" in row["dir"]}
no = c["temoe|nointerp"]["per_seed_mae"]
interp = c["temoe|interp"]["per_seed_mae"]
shuf = ctl["temoe"]["per_seed_mae"] if "temoe" in ctl else None
if shuf:
    seeds = sorted(set(shuf) & set(no) & set(interp), key=int)
    d = [shuf[s] - no[s] for s in seeds]
    m, sd, n = st.fmean(d), st.stdev(d), len(d)
    t = m / (sd / sqrt(n)); p = 2 * sps.t.sf(abs(t), n - 1)
    half = sps.t.ppf(0.975, n - 1) * sd / sqrt(n)
    print(f"  mispaired cache vs no explanation : {m:+.4f}  95% CI [{m-half:+.4f}, {m+half:+.4f}]  t={t:.2f} p={p:.3f}")
    print(f"    -> 95% CI {'includes' if m-half < 0 < m+half else 'EXCLUDES'} 0: "
          f"'statistically indistinguishable' is {'defensible' if m-half < 0 < m+half else 'NOT defensible'}")
    d2 = [interp[s] - no[s] for s in seeds]
    m2, sd2 = st.fmean(d2), st.stdev(d2)
    t2 = m2 / (sd2 / sqrt(len(d2))); p2 = 2 * sps.t.sf(abs(t2), len(d2) - 1)
    print(f"  matched cache  vs no explanation  : {m2:+.4f}  t={t2:.2f} p={p2:.4f}")