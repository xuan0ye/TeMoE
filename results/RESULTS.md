# Formal Result Index

This directory contains aggregate evidence only. Licensed inputs, predictions,
per-sample rationales, caches, logs, and checkpoints are excluded.

## Main controlled crossing (mean MAE, five seeds)

| Architecture | MOSI no/with | CH-SIMS no/with | MOSEI no/with |
|---|---:|---:|---:|
| Late Fusion | 0.927 / 0.702 | 0.438 / 0.387 | 0.565 / 0.519 |
| LMF | 0.932 / 0.711 | 0.432 / 0.382 | 0.567 / 0.519 |
| Cross-Modal | 0.909 / 0.711 | diverged | 0.676 / 0.552 |
| MISA-lite | 0.937 / 0.722 | 0.436 / 0.391 | 0.558 / 0.515 |
| KuDA-lite | 0.918 / 0.698 | 0.432 / 0.389 | 0.563 / 0.518 |
| MAG-BERT-lite | 0.923 / 0.717 | 0.445 / 0.385 | 0.560 / 0.524 |
| Self-MM-lite | 0.941 / 0.714 | 0.438 / 0.391 | 0.568 / 0.517 |
| ALMT-lite | 0.927 / 0.683 | diverged | 0.575 / 0.521 |
| CorMulT-lite | 0.918 / 0.702 | diverged | 0.610 / 0.550 |
| TeMoE | 0.924 / 0.703 | 0.450 / 0.396 | 0.556 / 0.524 |

Source: `condition_audit.json`, `condition_audit.md`, and
`crossing_paired_stats.md`. “Lite” systems are controlled implementations in
this pipeline and are not claimed to reproduce the original papers exactly.

## Content controls and equivalence

- `official/control_caches.json` records matched, video-only, transcript-only,
  and no-cache conditions for TeMoE, LMF, and Late Fusion.
- `official/raw_transcript_control.json` compares raw-transcript encoding with
  Qwen transcript rewriting and the matched multimodal cache.
- `official/text_teacher_control.json` compares the 1.5B text teacher with the
  7B video-language teacher and includes the measured latency distributions.
- `official/equivalence_tests.{json,md}` records paired TOSTs, margins,
  confidence intervals, and Holm-adjusted decisions.

## MOSEI transcript-only rationale source control

results/mosei/analysis/mosei_source_control.json retains five paired test MAEs
per condition and consumer for TeMoE, LMF, and Late Fusion. The companion
Markdown displays mean MAEs, paired differences, 90%/95% CIs, and
within-family Holm-adjusted TOSTs. Transcript-only minus
video-plus-transcript mean differences are -0.001958, +0.000028, and
-0.002668 MAE respectively. All three pass the declared +/-0.06 margin
(maximum Holm p=1.984e-5) and a stricter +/-0.01 sensitivity check
(maximum Holm p=0.03203). The no-cache means are 0.5557, 0.5674,
and 0.5649. Recompute the two margin families with:

    python -m mmsa.analysis.verify_mosei_source_control --report results/mosei/analysis/mosei_source_control.json

This source comparison covers only these three MOSEI consumers on the fixed
test split; the released report does not claim a MOSEI clip-only control.

## Zero-shot and official-code baselines

- `official/qwen25vl_zeroshot_summary.json`: direct-score Qwen2.5-VL results on
  all three datasets, including availability and fallback counts.
- `official/official_baselines.json`: MMSA 2.2.1 baseline summaries. These runs
  establish an external implementation reference; they do not by themselves
  constitute the new cache/no-cache crossing.

The completed data-only crossing uses the released MMSA 2.2.1 LMF model and
trainer without source changes. Both arms have 701,679 trainable parameters and
the same configuration, feature shape, and seeds. Mean MAE changes from
0.9605+/-0.0142 without the cache to 0.7019+/-0.0205 with it; the paired
cached-minus-no-cache delta is -0.2586 (95% CI [-0.2770, -0.2401],
two-sided paired p=2.61e-6), and the cache is better on all five seeds.

Evidence:

```text
official/official_crossing_mosi_full.json
official/official_crossing_mosi_full.md
official_crossing/manifest.json
official_crossing/{no_cache,cached}_LMF_run_record.json
official_crossing/{no_cache,cached}_LMF_mosi.csv
```

The released MISA, Self-MM, and MMIM implementations require BERT-token input;
adding this continuous cache would require modifying their model input paths,
so they are retained only as uncached external references rather than described
as an unmodified official-code crossing.

## Efficiency

`efficiency/end_to_end.json` records the legacy 30-clip online median of
1.351 s and the 3.8123 ms GPU-resident TeMoE forward. These have different
boundaries and yield a **component ratio** of 354.4x, not an end-to-end speedup.
The completed matched-boundary audit (`efficiency/end_to_end_v2_full.json`)
includes cache lookup, tensor construction, host-to-device transfer, the same
trained TeMoE forward, and CUDA synchronization in both arms. Its medians are
4.8876 ms cached and 1318.5486 ms online, a 269.8x ratio; one-time model and
dataset loading and external network transport are excluded from both.

`efficiency/mosi_tradeoff.{json,md}` contains the internal efficiency profile.

Exact primary and official-MMSA package freezes are under
`environment/{wjhenv,mmsa_official}.freeze.txt`.
