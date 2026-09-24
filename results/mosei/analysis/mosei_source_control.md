# MOSEI rationale-source control

Five paired training seeds (1, 2, 3, 4, 42); lower test MAE is better.
Difference is transcript-only minus video-plus-transcript.
Equivalence uses the declared +/-0.06 MAE margin, a 90% t CI, and Holm correction across the three architectures.

| Architecture | None MAE | Transcript-only MAE | Video+transcript MAE | Difference | 95% CI | 90% CI | Holm TOST p | Equivalent |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| temoe | 0.5557 | 0.5217 | 0.5237 | -0.0020 | [-0.0089, +0.0050] | [-0.0073, +0.0034] | 1.984e-05 | yes |
| lmf | 0.5674 | 0.5190 | 0.5190 | +0.0000 | [-0.0064, +0.0065] | [-0.0049, +0.0050] | 1.984e-05 | yes |
| late_fusion | 0.5649 | 0.5159 | 0.5185 | -0.0027 | [-0.0091, +0.0037] | [-0.0076, +0.0023] | 1.984e-05 | yes |

The same per-seed values also pass a stricter +/-0.01 MAE sensitivity check (maximum Holm-adjusted TOST p=0.03203). Recompute both margins with:

    python -m mmsa.analysis.verify_mosei_source_control --report results/mosei/analysis/mosei_source_control.json

These comparisons are conditional on the fixed MOSEI test split and fixed cache generation. They cover these three consumers only; no MOSEI clip-only control is claimed.
