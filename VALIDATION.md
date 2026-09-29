# Packaging validation — 2026-09-24

Environment: macOS, Python 3.13.3; isolated virtualenv. Historical Linux/Python 3.11 runtime was not revalidated.

- All Python sources parsed; all 308 imported source hashes verified. Only analysis-config paths were changed.
- Credential-pattern matches reviewed: local placeholders and synthetic test credentials; no real API tokens or private keys found. This is a bounded review, not a security certification.
- Main snapshot + RQ4 analysis: `python -m pytest -q tests test_mas_rq4_analysis_20260914.py`: **1450 passed, 1 failed, 85 subtests passed** (49.32s).
- Failure: `tests/test_local_inference_config_probe.py::test_crossover_exact_requests_warmup_excluded_cost_included`. Serial/parallel case order differed. The test appends to a shared list from concurrent worker threads and compares list ordering, suggesting an ordering-sensitive test. No source/test behavior changed. Further diagnosis is required before claiming a full pass.
- RQ4 tests isolated with their own legacy source (README command): **20 passed, 48 subtests passed** (0.47s).
- No live models, server experiments, full benchmark replay or analysis-data recomputation performed.

Initial combined collection revealed omitted RQ4 legacy dependencies; the original legacy sources were included and RQ4 tests rerun separately. Do not mix both dated module versions in one Python process.
