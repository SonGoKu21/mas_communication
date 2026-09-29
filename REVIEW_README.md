# Anonymous replication package

This snapshot contains experiment code, configuration, analysis scripts and the static evidence explorer. It includes 26,613 indexed records and 6,800 detailed evidence records. Raw experimental JSONL inputs and model weights are not included; see README.md and VALIDATION.md for reproduction boundaries.

## Browse the website locally

From this directory, run:

```sh
python3 -m http.server 8000
```

Open http://localhost:8000/web/site/. Serving the package root also makes source-file links accessible. The website performs no live model calls.

## Validate exported evidence

```sh
python3 -m unittest discover -s web/tests -v
python3 web/tools/verify.py
```

The snapshot does not include Git history or deployment workflows. Bibliography author names and third-party project attributions are retained. The source manifest describes imported source files; the package checksum inventory describes this snapshot.
