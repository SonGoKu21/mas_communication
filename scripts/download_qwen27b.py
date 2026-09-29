"""Download the experiment model from ModelScope and retain a local hash manifest."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from modelscope import snapshot_download
from modelscope.hub.api import HubApi


def main():
    root = Path(os.environ.get("MAS_DATA_ROOT", "/data3/hqn/mas"))
    model_id = "Qwen/Qwen3.8-27B"
    target = root / "models" / "Qwen3.8-27B"
    endpoint = "https://www.modelscope.cn"
    api = HubApi(endpoint=endpoint)
    revision = api.get_valid_revision_detail(model_id, revision="master")
    files = api.get_model_files(model_id, revision="master", recursive=True)
    manifest = {"model_id": model_id, "source": endpoint, "requested_revision": "master",
                "revision_detail": revision, "remote_files": files,
                "started_at": datetime.now(timezone.utc).isoformat()}
    artifact = root / "artifacts" / "qwen38_27b_download_manifest_20260910.json"
    if artifact.exists():
        previous = json.loads(artifact.read_text())
        if previous.get("file_hashes"):
            raise RuntimeError("completed model manifest exists; verify it instead of downloading again")
    else:
        with artifact.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"download_start": model_id, "target": str(target),
                      "expected_bytes": sum(item.get("Size", 0) for item in files)}), flush=True)
    snapshot_download(model_id, revision="master", local_dir=str(target),
                      cache_dir=str(root / "cache" / "modelscope"),
                      max_workers=4, endpoint=endpoint)
    hashes = []
    for item in files:
        path = target / item["Path"]
        if not path.is_file() or path.stat().st_size != item["Size"]:
            raise RuntimeError(f"missing or incomplete model file: {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes.append({"path": item["Path"], "size": path.stat().st_size, "sha256": digest.hexdigest()})
    config = json.loads((target / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise RuntimeError("unexpected downloaded model architecture")
    manifest.update(file_hashes=hashes, model_config=config,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    provenance_note="ModelScope master snapshot; local SHA256 freezes downloaded bytes, not proof of historical-weight identity.")
    final = artifact.with_name(artifact.stem + "_verified.json")
    with final.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"download_complete": str(target), "verified_files": len(hashes),
                      "manifest": str(final)}), flush=True)


if __name__ == "__main__":
    main()
