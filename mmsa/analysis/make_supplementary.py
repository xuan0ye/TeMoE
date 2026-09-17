"""Build the supplementary material tables from the result JSONs.

The paper says "per-method numbers in supplement" and reports ranges over the
six reproduced baselines; this script expands every range into its individual
rows and adds the full ablation / efficiency / end-to-end / routing detail.

    python -m mmsa.analysis.make_supplementary --results-dir outputs/mmsa \
        --output outputs/mmsa/supplementary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # the crossing table is rebuilt from raw run reports when available
    from mmsa.analysis.audit_conditions import audit as _audit_conditions
    from mmsa.analysis.audit_conditions import control_markdown as _control_markdown
    from mmsa.analysis.audit_conditions import crossing_markdown as _crossing_markdown
    from mmsa.analysis.audit_conditions import paper_check as _paper_check
except Exception:  # pragma: no cover - keep the supplement generation working standalone
    _audit_conditions = None

METRICS = ("mae", "corr", "acc7", "acc2_has0", "f1_has0")


def _fmt(mean: dict, std: dict, key: str) -> str:
    m, s = mean.get(key), std.get(key)
    if m is None:
        return "--"
    return f"{m:.4f}$\\pm${s:.4f}" if s is not None else f"{m:.4f}"


def _multiseed_table(base: Path, dataset: str) -> list[str]:
    """Per-method main results straight from the multiseed aggregates."""
    root = base / dataset / "multiseed"
    if not root.exists():
        return [f"_{dataset}: no multiseed results found_", ""]
    rows = []
    for arch_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        agg = arch_dir / f"multiseed_{arch_dir.name}.json"
        if not agg.exists():
            matches = list(arch_dir.glob("multiseed_*.json"))
            if not matches:
                continue
            agg = matches[0]
        payload = json.loads(agg.read_text(encoding="utf-8"))
        mean, std = payload.get("mean", {}), payload.get("std", {})
        rows.append("| {} | {} |".format(arch_dir.name, " | ".join(_fmt(mean, std, m) for m in METRICS)))
    if not rows:
        return [f"_{dataset}: no aggregated arch results found_", ""]
    head = "| arch | " + " | ".join(m.upper() for m in METRICS) + " |"
    div = "|" + "---|" * (len(METRICS) + 1)
    return [head, div] + rows + [""]


def _crossing_table(base: Path, dataset: str) -> list[str]:
    """With/without-explanation crossing: the effect of the explanation per architecture."""
    arches = ("late_fusion", "cross_modal", "lmf", "misa", "kuda", "magbert",
              "selfmm", "almt", "cormult", "temoe")

    def load(tag: str, arch: str):
        folder = "multiseed" if tag == "orig" else f"multiseed_{tag}"
        path = base / dataset / folder / arch / f"multiseed_{arch}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("mean")

    rows = []
    for arch in arches:
        without = load("nointerp", arch)
        with_ = load("interp", arch) or load("orig", arch)
        if without is None or with_ is None:
            continue
        rows.append("| {} | {:.4f} | {:.4f} | {:+.4f} | {:.4f} | {:.4f} |".format(
            arch, without.get("mae", float("nan")), with_.get("mae", float("nan")),
            with_.get("mae", float("nan")) - without.get("mae", float("nan")),
            without.get("corr", float("nan")), with_.get("corr", float("nan"))))
    if not rows:
        return [f"_{dataset}: crossing runs not found_", ""]
    return ["| architecture | MAE w/o expl. | MAE w/ expl. | delta | Corr w/o | Corr w/ |",
            "|---|---|---|---|---|---|"] + rows + [""]

def _ablation_table(path: Path, title: str) -> list[str]:
    if not path.exists():
        return [f"_{title}: not found ({path})_", ""]
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"**{title}**", "", "| variant | MAE | Corr | Acc-7 | F1 | params | backend |",
             "|---|---|---|---|---|---|---|"]
    for name, entry in payload.items():
        mean, std = entry.get("mean", {}), entry.get("std", {})
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            name, _fmt(mean, std, "mae"), _fmt(mean, std, "corr"), _fmt(mean, std, "acc7"),
            _fmt(mean, std, "f1_has0"), entry.get("params", ""), entry.get("temporal_backend", "")))
    lines.append("")
    return lines


def _significance(path: Path, title: str) -> list[str]:
    if not path.exists():
        return [f"_{title}: not found ({path})_", ""]
    payload = json.loads(path.read_text(encoding="utf-8"))
    sig = payload.get("significance", payload)
    lines = [f"**{title}**", "", "| baseline | metric | t | p | p<0.05 |", "|---|---|---|---|---|"]
    for arch, metrics in sig.items():
        if not isinstance(metrics, dict):
            continue
        for metric, stats in metrics.items():
            if not isinstance(stats, dict) or "p" not in stats:
                continue
            lines.append("| {} | {} | {:.3f} | {:.4g} | {} |".format(
                arch, metric, stats.get("t", float("nan")), stats["p"],
                "yes" if stats.get("significant_0.05") else "no"))
    lines.append("")
    return lines


def _efficiency(base: Path) -> list[str]:
    lines = ["**Latency / throughput / size (MOSI, $T{=}50$, batch 32, one A800)**", "",
             "| config | ms/batch | samples/s | params | GFLOPs |", "|---|---|---|---|---|"]
    acc = base / "mosi" / "efficiency"
    for name in ("mamba", "gru", "gated_conv"):
        for candidate in (acc / f"{name}.json", acc / f"{name}_run1.json"):
            if candidate.exists():
                d = json.loads(candidate.read_text(encoding="utf-8"))
                lines.append("| {} | {} | {} | {} | {} |".format(
                    name, d.get("latency_ms_per_batch"), d.get("throughput_samples_per_s"),
                    d.get("params"), d.get("flops_gflops")))
                break
    # 5-run medians when present
    import statistics
    for tag, pat in (("mamba", "mamba_run{}"), ("gru", "gru_run{}"), ("gated_conv", "gconv_run{}")):
        vals = []
        for i in range(1, 6):
            f = acc / pat.format(i)
            if f.exists():
                vals.append(json.loads(f.read_text(encoding="utf-8"))["latency_ms_per_batch"])
        if len(vals) >= 2:
            lines.append("| {} (5-run median) | {:.3f} | -- | -- | -- |".format(tag, statistics.median(vals)))
    f = acc / "tradeoff.json"
    if f.exists():
        payload = json.loads(f.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
        lines += ["", "**top-$k$ sweep (single run)**", "", "| ms/batch | samples/s |", "|---|---|"]
        for r in rows:
            lines.append("| {} | {} |".format(r.get("latency_ms_per_batch"), r.get("throughput_samples_per_s")))
    f = acc / "end_to_end.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        lines += ["", "**End-to-end deployment (MOSI)**", "",
                  "| quantity | value |", "|---|---|",
                  f"| cached path, per sample | {d.get('cached_path_per_sample_ms')} ms |",
                  f"| online Qwen2.5-VL generation, median | {d.get('online_path', {}).get('median_s')} s |",
                  f"| online range | {d.get('online_path', {}).get('min_s')}--{d.get('online_path', {}).get('max_s')} s |",
                  f"| speed-up (online / cached) | {d.get('speedup_online_over_cached')}$\\times$ |",
                  f"| one-off cache build (all splits) | {d.get('cache_build', {}).get('one_off_gpu_hours')} GPU-hour |",
                  f"| clips in cache | {d.get('cache_build', {}).get('n_clips')} |", ""]
    return lines


def _routing(base: Path) -> list[str]:
    lines = ["**Expert-routing specialization (TeMoE, seed 42, MOSI test)**", "",
             "| quantity | value |", "|---|---|"]
    for candidate in (base / "mosi" / "analysis" / "expert_routing.json",
                      base / "mosi" / "analysis" / "expert_routing_pb.json"):
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            entry = payload[0] if isinstance(payload, list) else payload
            for key in ("n_experts", "n_samples", "top1_vs_polarity_nmi",
                        "mean_routing_entropy_nats", "mean_routing_entropy_normalized"):
                lines.append(f"| {key} | {entry.get(key)} |")
            util = entry.get("expert_utilization_top1_share")
            if util:
                lines.append("| top-1 utilization per expert | " + ", ".join(f"{u:.4f}" for u in util) + " |")
            break
    return lines + [""]


def _num(value, digits: int = 3) -> str:
    """Format a number for a markdown table, tolerating missing/NaN values."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    return f"{number:.{digits}f}"


def _rationale_quality(base: Path) -> list[str]:
    """Explanation-cache validation: is the cached rationale real, meaningful signal?

    Rendered from outputs/mmsa/<ds>/analysis/rationale_quality.json (see
    mmsa.analysis.rationale_quality).  Three numbers matter to a reviewer:
    polarity agreement shows the rationale carries affect rather than only
    description, the own-vs-other transcript similarity shows it is not just a
    paraphrase of the text modality, and the linear probe bounds how much of the
    label is readable from the rationale embedding alone.
    """
    lines: list[str] = []
    for dataset in ("mosi", "sims", "mosei"):
        path = base / dataset / "analysis" / "rationale_quality.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        probe = payload.get("linear_probe_on_rationale_only", {}) or {}
        for split, row in (payload.get("splits") or {}).items():
            text_stats = row.get("text", {})
            polarity = row.get("polarity_agreement", {})
            visual = row.get("visual_grounding", {})
            leak = row.get("transcript_leakage")
            lines += [f"**Explanation-cache validation ({dataset.upper()}, {split})**", "",
                      "| quantity | value |", "|---|---|",
                      f"| cache dir | {row.get('cache_dir')} |",
                      f"| n | {row.get('n')} |",
                      "| rationale tokens (mean / median) | {} / {} |".format(
                          _num(text_stats.get("tokens_mean"), 1), _num(text_stats.get("tokens_median"), 1)),
                      "| tokens p10--p90 | {}--{} |".format(
                          _num(text_stats.get("tokens_p10"), 1), _num(text_stats.get("tokens_p90"), 1)),
                      "| empty / unique rationale ratio | {} / {} |".format(
                          text_stats.get("empty_count"), _num(text_stats.get("unique_ratio"), 3)),
                      "| distinct-1 / distinct-2 | {} / {} |".format(
                          _num(text_stats.get("distinct_1"), 3), _num(text_stats.get("distinct_2"), 3)),
                      "| visual-grounding rate | {} ({}) |".format(
                          _num(visual.get("rate"), 3), visual.get("count")),
                      "| polarity vs label (Pearson / Spearman) | {} / {} |".format(
                          _num(polarity.get("pearson_vs_label"), 3), _num(polarity.get("spearman_vs_label"), 3)),
                      "| polarity sign accuracy (|label|>= {}, n={}) | {} |".format(
                          _num(polarity.get("neutral_band"), 1), polarity.get("sign_n"),
                          _num(polarity.get("sign_accuracy"), 3))]
            if leak:
                lines.append("| cos(rationale, own transcript) vs cos(rationale, other) | {} vs {} (gap {}) |".format(
                    _num(leak.get("cos_own_transcript_mean"), 3),
                    _num(leak.get("cos_other_transcript_mean"), 3),
                    _num(leak.get("gap"), 3)))
            entry = probe.get(split)
            if entry:
                lines.append("| linear probe on rationale only: MAE / Corr | {} / {} (n_train={}) |".format(
                    _num(entry.get("mae"), 4), _num(entry.get("corr"), 3), entry.get("n_train")))
            lines.append("")
    return lines


def _official_baselines(base: Path) -> list[str]:
    """The authors' own MMSA implementation, run on the paper's features and splits."""
    path = base / "official" / "official_baselines.json"
    if not path.exists():
        return [f"_official baselines: not found ({path})_", ""]
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    best: dict = {}
    for r in rows:
        key = (r.get("dataset"), r.get("model"))
        if key not in best or r.get("tag") == "aligned":
            best[key] = r
    lines = ["| corpus | model | MAE | Corr | Acc-7 |", "|---|---|---|---|---|"]
    for key in sorted(best, key=lambda k: (str(k[0]), best[k].get("mae") or 9e9)):
        r = best[key]
        lines.append("| {} | {} | {:.4f} $\\pm$ {:.4f} | {:.4f} | {:.2f} |".format(
            str(key[0]).upper(), key[1], r.get("mae") or float("nan"),
            r.get("mae_std") or float("nan"), r.get("corr") or float("nan"),
            r.get("acc7") or float("nan")))
    lines.append("")
    return lines


def _control_caches(base: Path) -> list[str]:
    """Modality ablation: which input of the generator actually carries the effect."""
    try:
        from mmsa.analysis.control_caches import ARCHS, load, paired_stats, per_seed_mae
    except Exception as exc:
        return [f"_modality ablation: analysis helpers unavailable ({exc})_", ""]
    root = base / "mosi"
    mapping = (("none", "none"), ("clip-only", "videonly"),
               ("transcript-only", "textonly"), ("matched", "matched"))
    table: dict = {}
    for arch in ARCHS:
        for label, cond in mapping:
            got, why = load(root, arch, cond)
            table[(arch, label)] = got or {"missing": why}
    lines = ["| architecture | " + " | ".join(l for l, _ in mapping) + " |",
             "|---" * (len(mapping) + 1) + "|"]
    for arch in ARCHS:
        cells = []
        for label, _ in mapping:
            v = table[(arch, label)]
            cells.append("--" if "missing" in v
                         else "{:.4f}$\\pm${:.4f}".format(v["mae"], v["mae_std"]))
        lines.append("| {} | {} |".format(arch, " | ".join(cells)))
    lines += ["", "**Paired $t$-tests over the five seeds** (negative diff = the first arm has lower MAE)", "",
              "| architecture | contrast | mean diff | 95% CI | t | p |", "|---|---|---|---|---|---|"]
    contrasts = (("clip-only", "none"), ("transcript-only", "none"),
                 ("clip-only", "transcript-only"), ("matched", "transcript-only"))
    for arch in ARCHS:
        series = {}
        for label, _ in mapping:
            v = table[(arch, label)]
            series[label] = {} if "missing" in v else per_seed_mae(v["path"])
        for x, y in contrasts:
            st = paired_stats(series[x], series[y])
            if st["n"] < 2:
                lines.append("| {} | {} vs {} | -- | -- | -- | -- |".format(arch, x, y))
                continue
            lines.append("| {} | {} vs {} | {:+.4f} | [{:+.4f}, {:+.4f}] | {:+.2f} | {:.4f} |".format(
                arch, x, y, st["diff"], st["lo"], st["hi"], st["t"], st["p"]))
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/mmsa")
    parser.add_argument("--output", default="outputs/mmsa/supplementary.md")
    args = parser.parse_args()

    base = Path(args.results_dir)
    out: list[str] = [
        "# TeMoE - Supplementary Material",
        "",
        "Per-method numbers behind the ranges reported in the main paper, plus the full",
        "ablation, efficiency, deployment and routing detail. All main results are",
        "mean$\\pm$std over 5 seeds; * marks $p<0.05$ against TeMoE in paired $t$-tests.",
        "",
    ]
    sec = 1
    datasets = [("mosi", "CMU-MOSI"), ("sims", "CH-SIMS")]
    mosei_root = base / "mosei"
    if mosei_root.is_dir() and any(mosei_root.glob("multiseed*/*/multiseed_*.json")):
        datasets.append(("mosei", "CMU-MOSEI"))
    for ds, label in datasets:
        out += [f"## S{sec}. Per-method results ({label})", ""] + _multiseed_table(base, ds)
        sec += 1
        out += [f"## S{sec}. Significance vs TeMoE ({label})", ""] + _significance(
            base / ds / "summary" / "significance.json", f"Paired t-tests vs TeMoE ({ds.upper()})")
        sec += 1
    out += [f"## S{sec}. Cross-architecture explanation crossing", "",
            "Rebuilt from each run's own `use_interpretation` flag rather than from directory",
            "names, because the first MOSI sweep wrote explanation-augmented and",
            "explanation-free runs into the same `multiseed/` folder. Rows therefore appear",
            "only where both conditions were actually run.", ""]
    for ds, label in datasets:
        out += [f"**{label}**", ""]
        report = _audit_conditions(ds, base / ds) if _audit_conditions else None
        if report is None or not any(report["conditions"].values()):
            out += [f"_{ds}: crossing runs not found_", ""]
            continue
        out += _crossing_markdown(report)
        out += _paper_check(report)
        out += _control_markdown(report)
    sec += 1
    out += [f"## S{sec}. Ablations", ""]
    sec += 1
    out += _ablation_table(base / "mosi" / "ablation" / "ablation_summary.json",
                           "Component ablation (CMU-MOSI, 11 classic variants where run)")
    out += _ablation_table(base / "sims" / "ablation" / "ablation_summary.json",
                           "Component ablation (CH-SIMS)")
    out += _ablation_table(base / "mosi" / "planb_ablation" / "ablation_summary.json",
                           "Channel-decomposed variants (CMU-MOSI)")
    out += _ablation_table(base / "sims" / "planb_ablation" / "ablation_summary.json",
                           "Channel-decomposed variants (CH-SIMS)")
    if _audit_conditions:
        gaps = []
        for ds, label in datasets:
            report = _audit_conditions(ds, base / ds)
            for item in report.get("incomplete", []):
                gaps.append(f"- {label}: {item}")
        if gaps:
            out += [f"## S{sec}. Incomplete runs", "",
                    "Architecture/condition pairs with fewer seeds than the maximum of the sweep.",
                    "These are excluded from any claim that requires a full seed set.", ""] + gaps + [""]
            sec += 1
    paired = base / "crossing_paired_stats.md"
    if paired.exists():
        out += [f"## S{sec}. Paired statistics for the crossed configurations", "",
                "Computed from the run reports that the main table is built from (paired by seed over the",
                "same five seeds), with Holm-Bonferroni correction across all pairs. Diverging",
                "configurations are excluded, as in the paper.", ""]
        body = paired.read_text(encoding="utf-8")
        out += body.split("\n\n", 1)[1].strip().split("\n")
        out += [""]
        sec += 1
    out += [f"## S{sec}. Efficiency and deployment", ""] + _efficiency(base)
    sec += 1
    out += [f"## S{sec}. Routing analysis", ""] + _routing(base)
    sec += 1
    out += [f"## S{sec}. Explanation-cache validation", "",
            "Direct measurements of the cached explanations as *data*: length and diversity",
            "(not one template repeated), visual grounding, agreement between the rationale's",
            "affect and the ground-truth label, similarity to the transcript it might merely",
            "paraphrase, and a closed-form ridge probe from rationale embeddings to labels.",
            ""] + _rationale_quality(base)
    # This section never incremented the counter, which was invisible until another
    # section followed it -- the generated file then carried two "## S12" headings.
    sec += 1
    out += ["", "**Note on CH-SIMS coverage.** ALMT and CorMulT diverged numerically on CH-SIMS", "(non-finite predictions under the shared training budget), so that dataset reports four", "reproduced baselines instead of six, as stated in the main paper table caption.", ""]

    out += [f"## S{sec}. Official baseline reproduction", "",
            "The authors' own MMSA implementation (2.2.1, `pip install MMSA`) trained on exactly the",
            "aligned features, splits and five seeds used everywhere else in this paper. MMSA writes",
            "its regression metrics multiplied by 100 in `results/*.csv`; the values below are divided",
            "back, a rule that was verified column by column against MMSA's own per-seed log lines",
            "(`mmsa/analysis/official_logs_check.py`). MMSA reports the population standard deviation.",
            ""] + _official_baselines(base)
    sec += 1
    out += [f"## S{sec}. Modality ablation: which input carries the effect", "",
            "Holding the generator (Qwen2.5-VL-7B-Instruct), the prompt template, greedy decoding and",
            "the sentence encoder fixed, and withholding one input at a time: *clip-only* is the cache",
            "generated with the transcript withheld, *transcript-only* the one generated with the clip",
            "withheld, *matched* the paper's cache, *none* no explanation at all. The generator and",
            "prompt are identical in all three cache builds, so the comparison varies exactly one input.",
            ""] + _control_caches(base)
    sec += 1
    text = "\n".join(out) + "\n"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path} ({len(text)} chars, {text.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
