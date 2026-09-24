# Verification — 2026-09-24

Aligned with manuscript revision 7: RQ1 Findings 1–3, RQ2 Findings 4–5, RQ3 Findings 6–7, RQ4 Findings 8–10. The PDF itself is excluded; its hash is in `site/data/paper-alignment.json`.

## Coverage

26,613 indexed records in six distinct collections; 6,800 bundled evidence records. No cross-collection pooled success rate.

| Collection | Indexed | Evidence |
|---|---:|---:|
| RQ1–RQ3 | 18,900 | 1,497 |
| Reddit extension | 2,835 | 425 |
| RQ4 main | 864 | 864 |
| RQ4 stress | 144 | 144 |
| RQ4 diagnostic | 90 | 90 |
| Source mapping | 3,780 | 3,780 |

## Current checks

- Ten exporter unit tests passed.
- Full count, unique-ID, evidence-reference, finding-reference and privacy-pattern audit passed.
- Paper browser test was first run against the old nine-card page and failed; after alignment it passed for four RQ groups, ten findings, and exact Fig. 5 / Fig. 7 run links.
- Browser suite passed for all findings, catalog loading, filters, pagination, playback, comparison, downloads, desktop/mobile widths, no JavaScript errors and no external runtime calls.
- Edge suite passed for subdirectory hosting, missing-evidence retry, stale-fetch handling and mobile comparison.
- Desktop and mobile finding screenshots visually inspected.

## Limits

No live model or benchmark run. Manuscript summary statistics are attributed to revision 7, not recomputed by the browser. Redaction is a tested pattern-based export, not a formal privacy certification. Benchmark task text remains available for inspection.

Fig. 4, Fig. 5 and Fig. 7 identify specific runs. Malformed/partial examples are explicitly cross-collection illustrations, not the exact earlier pair. Fig. 10 belongs to the earlier guarded-recheck pilot and its exact pair is not bundled; current factorial examples are labeled separately. The pilot and main mechanism study must not be conflated.

Independent review found no blocking data/paper/privacy issues. A case-navigation issue was reproduced with a failing browser assertion, then fixed so named-case selection also navigates the run list to the selected record's page. The full website test suite was rerun after this change.
