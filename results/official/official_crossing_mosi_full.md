# Official MMSA cache crossing (MOSI)

MMSA 2.2.1's released model and trainer are unmodified. The cached arm appends the
384-D rationale embedding to each aligned text timestep; the control appends zeros
of the same size. Both arms therefore have identical parameters and training budgets.

| official model | params/arm | no cache MAE | cached MAE | paired delta | 95% CI | two-sided paired p | cached better seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| LMF | 701679 | 0.9605+-0.0142 | 0.7019+-0.0205 | -0.2586 | [-0.2770, -0.2401] | 2.607e-06 | 5/5 |
