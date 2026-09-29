# Paper-to-artifact map

| Finding | Evidence and analysis | Principal denominator |
|---|---|---|
| 1 · Source, exposure and outcome | `artifacts/tables/rq1/`, common-matrix catalog, source-mapping fixtures | 3,780 fixtures; 15,882 applied MAS faults |
| 2 · Silent failures | `analysis/silent/`, `confirmation_records.json`, Figure 5 | 1,722 / 8,335 detection-complete failures |
| 3 · Recoverable information | `position_records.json`, position-extension runtime, Figure 6 | Matched applied position-specific cohorts |
| 4 · Task context | Common-matrix projected fields, Figure 7 | Per-domain Flat fault executions |
| 5 · Alternate evidence path | `analysis/topology/`, Figures 8–9 | Both-applied known-outcome pairs |
| 6 · Model versus topology rescue | `analysis/model_topology/` | 1,408 shared fault units across nine configurations |
| 7 · Dependency-matched protection | `analysis/protection/`, protection runtime directories, Figures 10–11 | Separate 1,098, 432, 162 and 288 execution studies |

The main confirmation matrix is 100 tasks × 3 models × 3 topologies × 7 condition-position cells × 3 repetitions = 18,900. Of 16,200 scheduled faults, 15,882 were applied and 318 did not reach the target. The six faulted cells include two positions for non-delivery; they are not six distinct fault families.

The position-extension study contains 16,200 scheduled records. The historical protection study contains 864 factorial, 90 information-path and 144 finite-exposure runs. Targeted evidence checks, Reddit recovery and persistent/shared recovery-boundary trials are separate studies. The website’s browsable-record counts describe artifact coverage, not statistical sample sizes for individual findings.
