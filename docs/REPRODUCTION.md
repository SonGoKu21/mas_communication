# Reproduction guide

## Offline statistics

Install `analysis/requirements.txt`, then run the five analysis commands in the root README. Outputs are written next to their scripts. The position table retains all 16,200 scheduled records, including three unknown outcomes. Unknown outcomes are not coerced to failures. `confirmation_records.json` preserves the detection and decision fields needed to recompute silent failures. Protection projections preserve paired identities, outcomes, incorrect-acceptance counts and resource accounting; request bodies and credentials are excluded.

The model/topology analysis uses 10,000 domain-stratified task-cluster bootstrap draws. The protection analysis resamples intact product clusters. Neither analysis treats individual repeated runs as independently randomized model seeds.

## Static evidence website

```sh
python3 -m unittest discover -s web/tests -v
python3 web/tools/verify.py
python3 web/tools/build_methodology.py
python3 -m http.server 8000 --directory web/site
```

The committed catalog is independently usable. `web/tools/export.py` is the archive-ingestion tool and requires the original raw inputs; it is not needed to browse the website or recompute the published projected-data analyses. The catalog contains the common matrix, Reddit extension, source fixtures and historical protection runs. Later position and protection studies are distributed as separate analysis inputs.

## Experiment code and tests

Use Python 3.11 or newer and install `code/requirements.txt` plus pytest and pytest-socket. Run shared tests from `code/`:

```sh
PYTHONPATH=src:. python3 -m pytest -q
```

Run each protection experiment in a separate Python process from its directory. For `factorial`, `evidence_checks` and `recovery_boundary`:

```sh
PYTHONPATH=legacy/src:legacy:. python3 -m pytest -q -o pythonpath='legacy/src legacy .' test_*.py
```

For `reddit_recovery`:

```sh
PYTHONPATH=../../../src:. python3 -m pytest -q -o pythonpath='../../../src .' test_*.py
```

The experiment runners preserve their evaluated fault transformations, workflow decisions and bounded recovery policies. Some test fixtures retain historical gate summaries to verify source binding; they are not fresh service-health evidence.

## Live-execution prerequisites

Provision the corresponding benchmark tasks and service snapshots, model configuration, model/API access and domain evaluator. Provide task manifests and output locations explicitly. The generic paths are examples. The source-mapping experiment may require Linux networking privileges. SWE/TAC fault trials replay frozen evidence and do not represent a fresh end-to-end repair run for every fault.

Root-level entry points within `code/` cover the common confirmation matrix, benchmark preparation and source mapping. `experiments/position_extension` provides the distinct I3-era implementation. Protection entry points are `factorial/runner.py`, `evidence_checks/p1_runner.py`, `reddit_recovery/launcher.py` (clean), `reddit_recovery/fault_launcher.py` (fault), and `recovery_boundary/p3_runner.py`.

`python3 analysis/position/analyze.py` reproduces the matched position table. `artifacts/tables/context/figure7_reported_rates.csv` contains the reported Figure 7 plot values; it is a figure input, separate from the per-execution analysis projections. Three optional Admin archive-regression tests require `MAS_HISTORICAL_ADMIN_RECORDS`; the remaining tests run from bundled inputs.

## Position-extension tests

Run each suite in its own process. From `code/experiments/position_extension`, use `PYTHONPATH=src:. python3 -m pytest -q -o pythonpath='src .' test_deepseek_inference_policy.py`. From either `admin_semantic/` or `qwen27/`, use `PYTHONPATH=../src:. python3 -m pytest -q -o pythonpath='../src .' test_*.py`. The three optional raw-archive regression tests are skipped when the original Admin archive is not supplied.
