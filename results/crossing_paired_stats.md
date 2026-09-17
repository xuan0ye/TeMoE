# Paired statistics for the crossed configurations

Paired by seed over the same five seeds; CI is a 95% Student-t interval for the mean MAE
change (cached minus no-cache, so negative = the cache helps), and Holm is Holm-Bonferroni
over all 27 pairs. Diverging configurations (CH-SIMS Cross-Modal/ALMT/CorMulT) are excluded,
as in the paper.

| dataset | architecture | mean MAE change | 95% CI | t | p | Holm |
|---|---|---|---|---|---|---|
| MOSI | ALMT-lite | -0.245 | [-0.267, -0.222] | -30.1 | 0.0000 | yes |
| MOSI | Cross-Modal | -0.198 | [-0.219, -0.176] | -25.4 | 0.0000 | yes |
| MOSEI | ALMT-lite | -0.054 | [-0.061, -0.047] | -22.0 | 0.0000 | yes |
| SIMS | MAG-BERT-lite | -0.060 | [-0.069, -0.051] | -18.8 | 0.0000 | yes |
| SIMS | MISA-lite | -0.046 | [-0.052, -0.039] | -18.8 | 0.0000 | yes |
| MOSEI | MAG-BERT-lite | -0.036 | [-0.041, -0.031] | -18.8 | 0.0000 | yes |
| MOSI | LMF | -0.221 | [-0.254, -0.188] | -18.4 | 0.0001 | yes |
| MOSEI | LMF | -0.048 | [-0.056, -0.041] | -17.9 | 0.0001 | yes |
| SIMS | Late Fusion | -0.051 | [-0.059, -0.043] | -17.4 | 0.0001 | yes |
| MOSI | KuDA-lite | -0.220 | [-0.256, -0.185] | -17.2 | 0.0001 | yes |
| MOSEI | Late Fusion | -0.046 | [-0.054, -0.039] | -16.5 | 0.0001 | yes |
| MOSEI | KuDA-lite | -0.045 | [-0.054, -0.036] | -14.5 | 0.0001 | yes |
| SIMS | LMF | -0.050 | [-0.060, -0.041] | -14.4 | 0.0001 | yes |
| MOSI | TeMoE (ours) | -0.221 | [-0.264, -0.178] | -14.3 | 0.0001 | yes |
| SIMS | TeMoE (ours) | -0.054 | [-0.065, -0.043] | -13.5 | 0.0002 | yes |
| SIMS | KuDA-lite | -0.043 | [-0.052, -0.034] | -13.4 | 0.0002 | yes |
| MOSEI | CorMulT-lite | -0.060 | [-0.073, -0.047] | -12.9 | 0.0002 | yes |
| MOSEI | MISA-lite | -0.043 | [-0.052, -0.033] | -12.7 | 0.0002 | yes |
| MOSI | Late Fusion | -0.225 | [-0.278, -0.173] | -11.9 | 0.0003 | yes |
| SIMS | Self-MM-lite | -0.047 | [-0.058, -0.036] | -11.8 | 0.0003 | yes |
| MOSEI | Self-MM-lite | -0.051 | [-0.064, -0.039] | -11.5 | 0.0003 | yes |
| MOSI | MISA-lite | -0.215 | [-0.272, -0.159] | -10.5 | 0.0005 | yes |
| MOSI | MAG-BERT-lite | -0.206 | [-0.266, -0.146] | -9.5 | 0.0007 | yes |
| MOSI | Self-MM-lite | -0.227 | [-0.295, -0.158] | -9.2 | 0.0008 | yes |
| MOSEI | TeMoE (ours) | -0.032 | [-0.042, -0.021] | -8.5 | 0.0011 | yes |
| MOSI | CorMulT-lite | -0.216 | [-0.290, -0.142] | -8.1 | 0.0013 | yes |
| MOSEI | Cross-Modal | -0.124 | [-0.200, -0.048] | -4.5 | 0.0106 | yes |
