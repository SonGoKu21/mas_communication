# Historical implementation ledger — 2026-09-22

The entries below describe the original local-only delivery. Current authorization and validation are recorded in README.md and VERIFICATION.md. On 2026-09-24 the user authorized public historical replay on GitHub Pages and alignment with manuscript revision 7.

# Execution Ledger

- Implemented in an isolated new directory within the existing dirty workspace; no existing application or experimental source was modified.
- Data tests first failed on missing exporter, then passed after implementation.
- Run identity test exposed duplicate historical run IDs across model configurations in the extension. Public identity now includes model, source, study and original ID; the regression test passes.
- Ruling: Python Playwright replaces the plan's JavaScript browser runner because Node was unavailable on PATH. Browser coverage remains the same.
- Ruling: all fixture records retain observations but no fabricated conversation. Representative MAS detailed traces are bundled alongside a complete verified catalog to avoid shipping gigabytes of raw archives.
- Ruling: no external deployment or new model call. Only local preview and upload artifacts.
- Independent review completed by a separate agent. Three P2 findings fixed: fixture matching includes severity/deadline/payload; source-evidence run IDs survive export; RQ4 decision_correct is retained separately.
- Ten unit tests pass. Browser verification includes all nine finding entries, failures/retry, stale-fetch protection and subdirectory hosting. Data count/reference/privacy-pattern audit passed.
