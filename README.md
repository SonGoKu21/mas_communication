# Broken Telephone

Research artifact for **A Systematic Empirical Study of Communication Faults in LLM-Based Multi-Agent Systems**.

The artifact connects seven findings to experiment implementations, de-identified analysis inputs, and a static historical evidence explorer. The manuscript is not included.

- [Evidence website](https://songoku21.github.io/mas_communication/)
- [Extended methodology](https://songoku21.github.io/mas_communication/methodology.html)
- [Reproduction guide](docs/REPRODUCTION.md)
- [Paper-to-artifact map](docs/PAPER_MAP.md)
- [Configuration](PUBLIC_CONFIGURATION.md)

## Repository layout

| Directory | Contents |
|---|---|
| `code/src/mas_faults/` | Shared communication faults, workflows and benchmark adapters |
| `code/experiments/position_extension/` | I2/I3/I4 experiment implementation and model-specific adapters |
| `code/experiments/protection/factorial/` | Main, information-path and finite-exposure protection experiments |
| `code/experiments/protection/evidence_checks/` | Targeted Shopping evidence checks and readback policies |
| `code/experiments/protection/reddit_recovery/` | Sequential Reddit clean and fault recovery stages |
| `code/experiments/protection/recovery_boundary/` | Persistent and shared-scope recovery exposure |
| `analysis/` | Offline matched topology, model, silent-failure and protection analyses |
| `artifacts/tables/` | De-identified analysis fields and aggregate tables |
| `web/` | Static website, 26,613 indexed executions and 6,800 detailed evidence exports |

Runtime-specific compatibility modules preserve the implementations needed by each experiment. They are isolated because substituting a newer same-named adapter can change an experiment's behavior. The replay catalog and the additional analysis cohorts are separate resources; their row counts must not be added as a single experimental denominator.

## Browse locally

```sh
python3 -m http.server 8000 --directory web/site
```

Open `http://localhost:8000/`. Browsing archived evidence does not contact model providers or benchmark services.

## Recompute analyses

```sh
python3 -m pip install -r analysis/requirements.txt
python3 analysis/topology/analyze.py
python3 analysis/model_topology/analyze.py
python3 analysis/model_topology/verify.py
python3 analysis/silent/analyze.py
python3 analysis/protection/analyze_protection.py
```

These commands operate on published projections of completed experiments. Full live execution additionally requires models, external benchmark assets and configured services; see the reproduction guide.
