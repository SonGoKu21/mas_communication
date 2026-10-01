# Release validation

Validated on 2026-09-29 in an independent publication checkout. No live model or benchmark experiment was run.

| Check | Result |
|---|---|
| Shared experiment code | 1,397 tests passed; 85 subtests passed |
| Historical protection runtime | 20 tests passed; 48 subtests passed |
| Targeted evidence checks | 45 tests passed; 53 subtests passed |
| Recovery boundary | 38 tests passed; 44 subtests passed |
| Reddit recovery | 20 tests passed |
| Position inference policy | 5 tests passed |
| Position Admin semantics | 21 tests passed; 6 subtests passed; 3 optional original-archive checks skipped |
| Position Qwen runtime | 9 tests passed |
| Paired-analysis unit checks | 14 tests passed |
| Static export unit checks | 10 tests passed |
| Website catalog | 26,613 unique indexed records; 6,800 detailed exports; case and task references valid |
| Browser checks | Home, methodology, paper cases, replay, comparison, downloads, stable numbering, error recovery and mobile layout passed |
| Privacy paths | No flagged personal home paths, numbered storage mounts or non-example machine IPs in tracked text |

Offline analyses reproduced the 1,408-unit common fault cohort, independently recounted 1,296 model/topology contrasts and 324 failure-overlap groups, reproduced 135 matched position cells including the Admin table, and reconciled silent failures and protection resource accounting. Figure 7 is distributed as manuscript-reported plot values and is not included in the claim of per-execution recomputation.

Figures 2–11 were extracted from the bound manuscript digest and visually checked. Source links in the methodology resolve to files in this release. A separate code review checked portability, dependency closure and evidence references.

Full live reproduction requires external benchmark assets and configured services. Historical gate fixtures are offline test data, not current service admission. The original local experiment sources and previous staging checkout were not edited by release curation.

## Manuscript content synchronization — 2026-10-01

Updated the title, findings, taxonomy citations, statistical protocol and Figures 1–11 against the latest manuscript digest recorded in `web/site/data/paper-alignment.json`. The source-mapping percentage uses 3,600 faulted executions, separate from 180 clean controls. The manuscript PDF is not distributed.

Fresh checks: 10 static-export tests; catalog uniqueness, case links and task mapping; anonymous archive loader; all 6,832 archive scripts byte-verified against JSON; home, methodology and paper-case browser checks including mobile layout; private-path audit with zero findings. All passed. Figure crops were visually inspected. Experiment implementations and archived execution records were unchanged; no new experiment results are claimed.
