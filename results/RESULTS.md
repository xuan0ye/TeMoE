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

## Zero-shot and official-code baselines

- `official/qwen25vl_zeroshot_summary.json`: direct-score Qwen2.5-VL results on
  all three datasets, including availability and fallback counts.
- `official/official_baselines.json`: MMSA 2.2.1 baseline summaries. These runs
  establish an external implementation reference; they do not by themselves
  constitute the new cache/no-cache crossing.

The five-seed official MMSA cache crossing introduced for the release audit was
still running when this snapshot was built. Its smoke result is intentionally
excluded from formal claims. The final files belong at:

```text
official/official_crossing_mosi_full.json
official/official_crossing_mosi_full.md
```

## Efficiency

`efficiency/end_to_end.json` records the legacy 30-clip online median of
1.351 s and the 3.8123 ms GPU-resident TeMoE forward. These have different
boundaries and yield a **component ratio** of 354.4x, not an end-to-end speedup.
The matched-boundary v2 full report is pending and must be added as
`efficiency/end_to_end_v2_full.json` after the full audit completes.

`efficiency/mosi_tradeoff.{json,md}` contains the internal efficiency profile.
