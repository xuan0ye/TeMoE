# Condition audit (from raw run reports)

Condition, not folder name, is read from each run's `use_interpretation`
flag in `<arch>_report.json`, so a mixed folder cannot mis-attribute a
number. `delta` is the MAE change from adding the cached explanation.

## MOSI  (125 runs, 5 seeds)

| architecture | MAE w/o expl. | MAE w/ expl. | delta | Corr w/o | Corr w/ | seeds |
|---|---|---|---|---|---|---|
| Late Fusion | 0.927 | 0.702 | -0.225 | 0.682 | 0.824 | 5 |
| LMF | 0.932 | 0.711 | -0.221 | 0.679 | 0.814 | 5 |
| Cross-Modal | 0.909 | 0.711 | -0.198 | 0.692 | 0.818 | 5 |
| MISA-lite | 0.937 | 0.722 | -0.215 | 0.677 | 0.820 | 5 |
| KuDA-lite | 0.918 | 0.698 | -0.220 | 0.681 | 0.821 | 5 |
| MAG-BERT-lite | 0.923 | 0.717 | -0.206 | 0.682 | 0.813 | 5 |
| Self-MM-lite | 0.941 | 0.714 | -0.227 | 0.689 | 0.819 | 5 |
| ALMT-lite | 0.927 | 0.683 | -0.245 | 0.680 | 0.825 | 5 |
| CorMulT-lite | 0.918 | 0.702 | -0.216 | 0.704 | 0.819 | 5 |
| TeMoE (ours) | 0.924 | 0.703 | -0.221 | 0.670 | 0.819 | 5 |

**Verification against the submitted table**

| architecture | paper w/o | audit w/o | paper w/ | audit w/ | verdict |
|---|---|---|---|---|---|
| Late Fusion | 0.927 | 0.927 | 0.702 | 0.702 | ok |
| LMF | 0.932 | 0.932 | 0.711 | 0.711 | ok |
| Cross-Modal | 0.909 | 0.909 | 0.711 | 0.711 | ok |
| MISA-lite | 0.937 | 0.937 | 0.722 | 0.722 | ok |
| KuDA-lite | 0.918 | 0.918 | 0.698 | 0.698 | ok |
| MAG-BERT-lite | 0.923 | 0.923 | 0.717 | 0.717 | ok |
| Self-MM-lite | 0.941 | 0.941 | 0.714 | 0.714 | ok |
| ALMT-lite | 0.927 | 0.927 | 0.683 | 0.683 | ok |
| CorMulT-lite | 0.918 | 0.918 | 0.702 | 0.702 | ok |
| TeMoE (ours) | 0.924 | 0.924 | 0.703 | 0.703 | ok |

**Paper-vs-raw-data verification: PASS**

**Anomalies and scope**

- duplicate run for `cross_modal|interp` in `multiseed/cross_modal`, `multiseed_interp/cross_modal` (used `multiseed/cross_modal`, identical values)
- duplicate run for `kuda|nointerp` in `multiseed/kuda`, `multiseed_nointerp/kuda` (used `multiseed/kuda`, identical values)
- duplicate run for `late_fusion|interp` in `multiseed/late_fusion`, `multiseed_interp/late_fusion` (used `multiseed/late_fusion`, identical values)
- duplicate run for `lmf|interp` in `multiseed/lmf`, `multiseed_interp/lmf` (used `multiseed/lmf`, identical values)
- duplicate run for `misa|nointerp` in `multiseed/misa`, `multiseed_nointerp/misa` (used `multiseed/misa`, identical values)
- `use_channels` coverage: 0/213 runs record the flag (all recorded values are []); 213 predate the field, i.e. were written before the channel-path fix
- out of scope for the crossing table: `ablation` (43 runs), `planb_ablation` (25 runs)

**Control runs (not part of the crossing table)**

| control | run | condition | MAE | Corr | seeds |
|---|---|---|---|---|---|
| load-balancing sweep, no explanation (control) | `ablation_aux/temoe` | w/o expl. | 0.925 | 0.677 | 10 |
| mispaired cache (control) | `multiseed_shuffled/temoe` | w/ expl. | 0.912 | 0.684 | 5 |

## SIMS  (120 runs, 5 seeds)

| architecture | MAE w/o expl. | MAE w/ expl. | delta | Corr w/o | Corr w/ | seeds |
|---|---|---|---|---|---|---|
| Late Fusion | 0.438 | 0.387 | -0.051 | 0.591 | 0.684 | 5 |
| LMF | 0.432 | 0.382 | -0.050 | 0.596 | 0.688 | 5 |
| Cross-Modal | 0.606 | 0.606 | diverged | -- | -- | 5 |
| MISA-lite | 0.436 | 0.391 | -0.046 | 0.593 | 0.679 | 5 |
| KuDA-lite | 0.432 | 0.389 | -0.043 | 0.594 | 0.678 | 5 |
| MAG-BERT-lite | 0.445 | 0.385 | -0.060 | 0.580 | 0.676 | 5 |
| Self-MM-lite | 0.438 | 0.391 | -0.047 | 0.591 | 0.681 | 5 |
| ALMT-lite | -- | -- | -- | -- | -- | _with run missing_ |
| CorMulT-lite | -- | -- | -- | -- | -- | _with run missing_ |
| TeMoE (ours) | 0.450 | 0.396 | -0.054 | 0.557 | 0.666 | 5 |

**Verification against the submitted table**

| architecture | paper w/o | audit w/o | paper w/ | audit w/ | verdict |
|---|---|---|---|---|---|
| Late Fusion | 0.438 | 0.438 | 0.387 | 0.387 | ok |
| LMF | 0.432 | 0.432 | 0.382 | 0.382 | ok |
| MISA-lite | 0.436 | 0.436 | 0.391 | 0.391 | ok |
| KuDA-lite | 0.432 | 0.432 | 0.389 | 0.389 | ok |
| MAG-BERT-lite | 0.445 | 0.445 | 0.385 | 0.385 | ok |
| Self-MM-lite | 0.438 | 0.438 | 0.391 | 0.391 | ok |
| TeMoE (ours) | 0.454 | 0.450 | 0.396 | 0.396 | MISMATCH |

**Paper-vs-raw-data verification: FAIL**

**Anomalies and scope**

- duplicate run for `kuda|nointerp` in `multiseed/kuda`, `multiseed_nointerp/kuda` (used `multiseed/kuda`, identical values)
- duplicate run for `late_fusion|interp` in `multiseed/late_fusion`, `multiseed_interp/late_fusion` (used `multiseed/late_fusion`, identical values)
- duplicate run for `lmf|interp` in `multiseed/lmf`, `multiseed_interp/lmf` (used `multiseed/lmf`, identical values)
- duplicate run for `magbert|nointerp` in `multiseed/magbert`, `multiseed_nointerp/magbert` (used `multiseed/magbert`, identical values)
- duplicate run for `misa|nointerp` in `multiseed/misa`, `multiseed_nointerp/misa` (used `multiseed/misa`, identical values)
- duplicate run for `selfmm|nointerp` in `multiseed/selfmm`, `multiseed_nointerp/selfmm` (used `multiseed/selfmm`, identical values)
- `use_channels` coverage: 5/185 runs record the flag (all recorded values are [False]); 180 predate the field, i.e. were written before the channel-path fix
- out of scope for the crossing table: `ablation` (35 runs), `planb_ablation` (25 runs)

## MOSEI  (100 runs, 5 seeds)

| architecture | MAE w/o expl. | MAE w/ expl. | delta | Corr w/o | Corr w/ | seeds |
|---|---|---|---|---|---|---|
| Late Fusion | 0.565 | 0.519 | -0.046 | 0.732 | 0.783 | 5 |
| LMF | 0.567 | 0.519 | -0.048 | 0.730 | 0.785 | 5 |
| Cross-Modal | 0.676 | 0.552 | -0.124 | 0.596 | 0.756 | 5 |
| MISA-lite | 0.558 | 0.515 | -0.043 | 0.736 | 0.785 | 5 |
| KuDA-lite | 0.563 | 0.518 | -0.045 | 0.732 | 0.784 | 5 |
| MAG-BERT-lite | 0.560 | 0.524 | -0.036 | 0.730 | 0.780 | 5 |
| Self-MM-lite | 0.568 | 0.517 | -0.051 | 0.726 | 0.784 | 5 |
| ALMT-lite | 0.575 | 0.521 | -0.054 | 0.717 | 0.781 | 5 |
| CorMulT-lite | 0.610 | 0.550 | -0.060 | 0.672 | 0.760 | 5 |
| TeMoE (ours) | 0.556 | 0.524 | -0.032 | 0.738 | 0.782 | 5 |

**Verification against the submitted table**

| architecture | paper w/o | audit w/o | paper w/ | audit w/ | verdict |
|---|---|---|---|---|---|
| Late Fusion | 0.565 | 0.565 | 0.519 | 0.519 | ok |
| LMF | 0.567 | 0.567 | 0.519 | 0.519 | ok |
| Cross-Modal | 0.676 | 0.676 | 0.552 | 0.552 | ok |
| MISA-lite | 0.558 | 0.558 | 0.515 | 0.515 | ok |
| KuDA-lite | 0.563 | 0.563 | 0.518 | 0.518 | ok |
| MAG-BERT-lite | 0.560 | 0.560 | 0.524 | 0.524 | ok |
| Self-MM-lite | 0.568 | 0.568 | 0.517 | 0.517 | ok |
| ALMT-lite | 0.575 | 0.575 | 0.521 | 0.521 | ok |
| CorMulT-lite | 0.610 | 0.610 | 0.550 | 0.550 | ok |
| TeMoE (ours) | 0.556 | 0.556 | 0.524 | 0.524 | ok |

**Paper-vs-raw-data verification: PASS**

**Anomalies and scope**

- `use_channels` coverage: 40/150 runs record the flag (all recorded values are [False]); 110 predate the field, i.e. were written before the channel-path fix
- out of scope for the crossing table: `ablation` (30 runs)

**Control runs (not part of the crossing table)**

| control | run | condition | MAE | Corr | seeds |
|---|---|---|---|---|---|
| load-balancing sweep, no explanation (control) | `ablation_aux/temoe` | w/ expl. | 0.558 | 0.736 | 10 |
| mispaired cache (control) | `multiseed_shuffled/temoe` | w/ expl. | 0.561 | 0.734 | 5 |

**How to read this.** A row appears only when both conditions were run for that
architecture, so a partially finished sweep shows explicit `missing` cells instead of
silently mixing conditions -- which is precisely the error this audit exists to catch.

