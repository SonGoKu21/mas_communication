# MAS Fault Observatory — manuscript revision 7

A static historical replay site aligned with the four RQs and Findings 1–10 in the manuscript supplied on 2026-09-24. The manuscript itself is not published. Its digest and mapping are in `site/data/paper-alignment.json`.

## Preview

```sh
python3 -m http.server 8767 --bind 127.0.0.1 --directory web/site
```

Open http://127.0.0.1:8767/ from the repository root. Production is hosted by GitHub Pages at https://SonGoKu21.github.io/mas_communication/ . Relative paths support the project subdirectory.

## Scope

This is archived evidence, with no model calls, GPU, database, live injection or experiment backend. All 26,613 records remain indexed across six separate collections. Additional selected traces cover the latest paper cases. The manifest reports exact coverage; summary-only records are explicitly labeled.

- RQ1 / Findings 1–3: source mapping, propagation versus outcome, observability. Fig. 4 links to Reddit 67 stale replay; malformed/partial examples span different collections and are illustrative.
- RQ2 / Findings 4–5: injection dependency and task-domain sensitivity. Fig. 5 links to Reddit 67 Flat I2/I4 r1; Admin 208/292 illustrate lookup versus aggregation.
- RQ3 / Findings 6–7: information-path diversity and deployed model configuration. Fig. 7 links to Admin 41 Qwen3.5-9B Sequential/Hierarchical semantic-corruption r1.
- RQ4 / Findings 8–10: complementary protection, information-source diversity, finite repeated exposure. Fig. 10 is from an earlier guarded-recheck pilot whose exact pair is not bundled. Main factorial examples are labeled separately.

Summary statistics are transcribed from manuscript revision 7, not recalculated by the browser. Individual examples are not substitutes for aggregate statistical tests. Repetitions are not model sampling seeds; handling-response flags are not verified recovery chains. Independent reacquisition still shares the Shopping backend.

## Rebuild

The original input archives are not tracked. Set `MAS_SOURCE_ROOT` to a workspace containing the original `reports/` inputs, then run:

```sh
MAS_SOURCE_ROOT=/path/to/source/workspace python3 web/tools/export.py
python3 -m unittest discover -s web/tests
python3 web/tools/verify.py
# With Python Playwright + Chromium installed, and a preview server running:
python3 web/tests/browser.py
python3 web/tests/paper_browser.py
REPLAY_URL=http://127.0.0.1:8767/ python3 web/tests/browser_edges.py
```

`paper-findings.json` is the source for the RQ/finding narrative and exact case queries. The exporter requires every named case to resolve to exactly one record with evidence. It sanitizes credentials, URLs, private paths, hosts, email addresses and selected personal-data fields. Benchmark task content remains visible for inspection.

The GitHub Pages workflow publishes only `web/site/`; Python tools, tests and development files are not deployed. Full original JSONL archives, paper PDFs and model weights are excluded.

## Stable display task IDs

`site/data/task-map.json` is the fixed identity map: Shopping Task 001–030, Admin 031–050, Reddit 051–065, SWE-bench 066–085, and TheAgentCompany 086–100. It is checked in and must not be regenerated or renumbered when catalog ordering changes. Repeated runs, models and topologies share the same display ID. The Reddit extension reuses the main-study mapping. The separate RQ4 study uses RQ4 Task 001–006 and source-mapping payload fixtures use Fixture 001–003.

Original task IDs, run IDs, source evidence and existing deep links remain unchanged. Search `Task 051` or `051` for a unified ID; unpadded `27`, `208`, or `xarray-2905` still search original identifiers. Downloads retain the source task ID and add `metadata.display_id`. The static audit rejects unmapped tasks so future additions require an explicit map update.

## Research Home page

Home introduces the study and separates the 18,900-run main matrix, 3,780 source-mapping tests and 1,098 protection-study runs. Four RQ sections reuse the canonical ten finding summaries, with Figures 2–10 extracted from the reviewed manuscript. Extra topology/case/model figures expand on demand; each figure can be enlarged. Evidence links connect to the existing Findings view, run explorer, task map and provenance.

`site/figures/source.json` records the manuscript digest, page numbers and reviewed crop rectangles. With PyMuPDF installed, `python3 web/tools/extract_figures.py /path/to/manuscript.pdf` rebuilds the images from that exact manuscript. The PDF itself is not included. Figure 10's earlier-pilot limitation remains visible.

## Extended methodology

`site/methodology.html` expands the author-supplied methods text with a 29-entry taxonomy, injection boundaries, six evaluated condition-position cells, A/M/O/B evidence criteria, an archived Admin 41 clean/fault walkthrough, and reproduction boundaries. Build it with `python3 web/tools/build_methodology.py`; `content/taxonomy.json` retains the manuscript citation keys. Worked excerpts are extracted from existing sanitized evidence, not manually reconstructed. Code links pin the reviewed historical repository snapshot. The supplement distinguishes framework coverage, implemented operators and evaluated conditions.

Methodology citations now resolve to 34 complete entries from the supplied manuscript bibliography, stored in `content/references.json`. `site/data/methodology-references.bib` contains the cited subset for download. Reference numbers are local to this page, not manuscript numbering.

Figure 2 was replaced with the author-supplied `layered_propagation.pdf` on 25 September 2026. Rebuilding figures now requires `--figure-2-pdf /path/to/layered_propagation.pdf` alongside the manuscript argument; both source hashes are checked. Only the rendered PNG is published.
