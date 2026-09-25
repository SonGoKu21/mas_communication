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

## Unified task numbering update

Added a fixed 109-entry identity map (100 main tasks, 6 RQ4 tasks, 3 source-mapping fixtures). New browser coverage first reproduced missing unified labels, then verified ordering, shared IDs across configurations and repeats, extension reuse, original-ID preservation and download metadata. Review exposed three-digit original-ID search being shadowed by new IDs; a failing `208` search regression was added and the unified numeric alias limited to 001–100.

## Research Home update — 2026-09-25

Home is now the default view, with four RQs, ten findings sourced from the same finding dataset, and nine manuscript figures (2–10). The full manuscript PDF is excluded. Figure crops and source digest are recorded in `site/figures/source.json`; extraction is reproducible with `tools/extract_figures.py`. The Fig. 10 pilot scope caveat remains explicit.

Ten exporter unit tests, the full data audit, and all five browser scripts passed. Home coverage includes figure loading and enlargement, evidence navigation, mobile overflow, and delayed-catalog navigation. Independent review identified early navigation before initialization and a tight Fig. 10 crop; both were corrected and verified. Desktop/mobile Home screenshots and figure crops were visually inspected.

## Extended methodology — 2026-09-25

Added a standalone, static methods supplement. The 29 taxonomy rows are extracted from the author-supplied methods table; exact worked-example projections are generated from existing sanitized Admin 41 clean/fault records (unified Task 033). Framework scope, operator support and evaluated condition-position cells are explicitly separated. Historical label semantics and archive completeness are disclosed.

The new methodology browser check first failed on the missing navigation link, then passed for navigation, all six sections, taxonomy count, exact fault-run routing and mobile width. Home, paper and numbering browser regressions, ten exporter tests and the full data audit also passed. Desktop/mobile screenshots were inspected. Repeated builds produced identical HTML. Independent review confirmed matched case data, pinned source paths and scope caveats with no blocking issues.
