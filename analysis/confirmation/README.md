# Confirmation paired statistical supplement

Run from any directory:

```sh
python3 -m pip install -r analysis/requirements.txt
python3 analysis/confirmation/analyze.py
python3 -m pytest analysis/confirmation/test_statistics.py
```

Outputs are `artifacts/tables/confirmation_statistics/paired_comparisons.csv`, its JSON equivalent and `protocol.json`. The input hash binds results to the 18,900-record published projection. No live model or benchmark calls are made.

Effects are **right minus left**, with equal weight per task. Every row gives the exact cohort, matched outcome cells, task count, interval, raw p-value and Holm-adjusted p-value. Model and topology contrasts are separate families, within domain and metric, following the historical analysis protocol. Missing recovery observations are excluded rather than imputed. The 10,000-sample bootstrap and sign-flip protocol is retained; these outputs are recomputed from the release projection using the explicitly documented NumPy random generator, rather than claimed byte-identical to an unpublished historical statistics table.

A-layer category comparisons need original A labels, which this compact input does not contain. The supplement covers clean/fault success, recorded MAS consequence and recorded recovery. It does not turn configuration-level associations or post-fault labels into causal claims.
