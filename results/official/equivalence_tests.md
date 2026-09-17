# Paired equivalence tests

Practical equivalence is 1% of the dataset label span: 0.06 MAE for MOSI/MOSEI and 0.02 MAE for CH-SIMS.

| family | comparison | mean difference | 90% CI | margin | TOST p | Holm p | equivalent |
|---|---|---:|---:|---:|---:|---:|:---:|
| mosi_qwen7b_textonly_vs_matched | temoe | +0.0156 | [+0.0005, +0.0308] | +/-0.060 | 0.001685 | 0.005054 | yes |
| mosi_qwen7b_textonly_vs_matched | lmf | -0.0069 | [-0.0300, +0.0163] | +/-0.060 | 0.004054 | 0.008108 | yes |
| mosi_qwen7b_textonly_vs_matched | late_fusion | -0.0034 | [-0.0440, +0.0373] | +/-0.060 | 0.02056 | 0.02056 | yes |
| mosi_cliponly_vs_none | temoe | -0.0231 | [-0.0526, +0.0064] | +/-0.060 | 0.02803 | 0.02812 | yes |
| mosi_cliponly_vs_none | lmf | +0.0100 | [-0.0216, +0.0417] | +/-0.060 | 0.01406 | 0.02812 | yes |
| mosi_cliponly_vs_none | late_fusion | -0.0054 | [-0.0326, +0.0217] | +/-0.060 | 0.006382 | 0.01915 | yes |
| mosi_mispaired_vs_none | temoe | -0.0124 | [-0.0290, +0.0043] | +/-0.060 | 0.001843 | 0.001843 | yes |
| mosi_architecture_choices_vs_full | no_mamba | -0.0138 | [-0.0291, +0.0015] | +/-0.060 | 0.001503 | 0.00313 | yes |
| mosi_architecture_choices_vs_full | ta_gated_conv_text | +0.0026 | [-0.0146, +0.0199] | +/-0.060 | 0.001043 | 0.00313 | yes |
| mosi_architecture_choices_vs_full | dense_moe | -0.0171 | [-0.0336, -0.0005] | +/-0.060 | 0.002612 | 0.00313 | yes |
| sims_architecture_choices_vs_full | no_mamba | -0.0029 | [-0.0185, +0.0127] | +/-0.020 | 0.04014 | 0.04014 | yes |
| sims_architecture_choices_vs_full | ta_gated_conv_text | +0.0046 | [-0.0020, +0.0113] | +/-0.020 | 0.003888 | 0.01166 | yes |
| sims_architecture_choices_vs_full | dense_moe | +0.0017 | [-0.0113, +0.0146] | +/-0.020 | 0.01967 | 0.03934 | yes |
| mosei_architecture_choices_vs_full | no_mamba | +0.0015 | [-0.0020, +0.0049] | +/-0.060 | 1.703e-06 | 5.11e-06 | yes |
| mosei_architecture_choices_vs_full | ta_gated_conv_text | +0.0001 | [-0.0049, +0.0051] | +/-0.060 | 7.016e-06 | 1.403e-05 | yes |
| mosei_architecture_choices_vs_full | dense_moe | +0.0012 | [-0.0038, +0.0063] | +/-0.060 | 7.708e-06 | 1.403e-05 | yes |
