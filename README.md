# MAS Communication Fault Experiments

Code, experiment configurations, analysis scripts, and archived evidence for a study of communication faults in LLM-based multi-agent systems.

## Explore the evidence

The [MAS Fault Observatory](web/site/index.html) presents four research questions, ten findings, illustrated cases, and an explorer of historical runs. It contains **26,613 indexed records**, including **6,800 records with detailed evidence**. The website makes no live model calls.

See the [extended methodology](web/site/methodology.html), [website documentation](web/README.md) for methods, evidence coverage, and local browsing instructions. The anonymous repository viewer may require enabling JavaScript for interactive features.

To run the website locally, serve the repository root:

```bash
python3 -m http.server 8000
# Open http://localhost:8000/web/site/
```

## Repository contents

| Location | Contents |
| --- | --- |
| `src/mas_faults/` | Fault injection, workflows, evidence validation, and mitigation implementations. |
| `run_*.py`, `scripts/` | Experiment entry points, audits, parallel execution utilities, and inference probes. |
| `tests/` | Offline tests for the main code snapshot. |
| `mas_rq4_20260914/`, `mas_rq4_*.py` | RQ4 experiment and analysis code, retained as a separate version. |
| `analysis/rq123_20260915/` | Historical RQ1–3 analysis snapshot. Its `mas_faults` module serves that snapshot only. |
| `reports/domain_pattern_audit.py`, `reports/*/analysis_config.json` | Domain-level analysis and experiment-matrix configurations. |
| `web/` | Static website, sanitized evidence, and export and verification tools. |
| `deploy/`, `*-env.sh` | Deployment templates requiring local paths, ports, and device assignments. |
| `SOURCE_MANIFEST.json` | Source provenance and original and packaged file hashes. |
| `PACKAGE_SHA256.json` | SHA-256 checksums for the anonymous artifact snapshot. |

See [PUBLIC_CONFIGURATION.md](PUBLIC_CONFIGURATION.md) for environment variables and deployment placeholders.

## Environment and offline tests

The historical deployment used Python 3.11. `requirements.lock.txt` and `conda-explicit-linux-64.txt` describe the Linux deployment environment; they are not general-purpose macOS installation specifications.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PWD"
python -m pytest -q
python -m pytest -q test_mas_rq4_analysis_20260914.py
# Validate RQ4 with its own legacy source in a separate Python process.
(cd mas_rq4_20260914 && PYTHONPATH="$PWD/legacy/src:$PWD/legacy:$PWD" python -m pytest -q -o pythonpath='' test_contract.py test_design.py test_exposure.py test_runner.py test_runtime.py)
```

Tests disable network sockets by default while allowing Unix sockets. Offline tests do not replace validation against live model and benchmark services. Configure service endpoints and data directories before running deployment or batch-experiment scripts. Model credentials are supplied through environment variables; `local-no-secret` and test tokens are placeholders.

To validate the exported evidence:

```bash
python3 -m unittest discover -s web/tests -v
python3 web/tools/verify.py
```

## Analysis inputs

Input paths in analysis configurations are relative to the repository root. Run analysis scripts from that directory. Raw experimental JSONL files are not included and must be supplied at the configured locations to recompute aggregate results.

`domain_pattern_audit.py` uses `reports/rq123_six_condition_inputs_20260916_v2/analysis_config.json`. The published evidence supports case inspection but does not, by itself, support recomputation of every reported aggregate.

## Reproduction scope

The main implementation comes from the `mas_reproduction_20260910` code snapshot. Later RQ4 and analysis snapshots are retained separately to preserve their version boundaries. Equivalence to the latest server implementation has not been established.

The historical recovery record identifies a missing `mas_faults.causal_trace_report` module and interface differences between the SWE multi-turn entry point and the recovered SWE modules. Those execution paths have not been fully reproduced. See [VALIDATION.md](VALIDATION.md) for verification results and limitations; historical run counts and test results are not current validation.

The artifact includes selected research figures and sanitized evidence. It excludes the manuscript, raw experiment logs, model weights, third-party benchmark repositories, and real credentials. No open-source license has been specified.
