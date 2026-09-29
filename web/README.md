# Static evidence explorer

The website presents seven findings from the study, methodology, figures, and historical execution evidence. It uses static HTML, CSS, JavaScript and JSON; no backend or model calls are required.

From the repository root:

```sh
python3 -m http.server 8000 --directory web/site
python3 -m unittest discover -s web/tests -v
python3 web/tools/verify.py
```

Open `http://localhost:8000/`. Browser interaction tests under `web/tests/` use Playwright and `REPLAY_URL` (default `http://127.0.0.1:8767/`).

`paper-findings.json` is the canonical finding content. The generated catalog keeps historical task/run identities. `site/data/paper-alignment.json` and `site/figures/source.json` bind content and figure crops to the manuscript digest without distributing the manuscript.

`tools/export.py` ingests original archives and requires those raw files. Browsing and the separate root-level statistical analyses use the committed public data and do not need original machine paths.
