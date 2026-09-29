from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPOSITORY_QUOTAS = {
    "django/django": 5,
    "pallets/flask": 1,
    "astropy/astropy": 10,
    "sympy/sympy": 2,
    "pydata/xarray": 2,
    "matplotlib/matplotlib": 1,
    "mwaskom/seaborn": 1,
    "scikit-learn/scikit-learn": 3,
    "pytest-dev/pytest": 2,
    "pylint-dev/pylint": 1,
    "sphinx-doc/sphinx": 2,
    "psf/requests": 1,
}
DEFAULT_MAX_FAIL_TO_PASS = 10


def repository_domain(repo: str) -> str:
    if repo in {"django/django", "pallets/flask"}:
        return "web_framework"
    if repo in {"astropy/astropy", "sympy/sympy", "pydata/xarray"}:
        return "numerical_and_data"
    if repo in {"matplotlib/matplotlib", "mwaskom/seaborn", "scikit-learn/scikit-learn"}:
        return "visualization_and_ml"
    return "tooling_and_library"


def patch_paths(patch: str) -> tuple[str, ...]:
    return tuple(re.findall(r"^diff --git a/(.+?) b/", patch, flags=re.MULTILINE))


def fail_to_pass_count(record: dict[str, Any]) -> int:
    return len(json.loads(str(record.get("FAIL_TO_PASS", "[]"))))


def filter_by_fail_to_pass(records: list[dict[str, Any]], max_fail_to_pass: int) -> list[dict[str, Any]]:
    if max_fail_to_pass < 1:
        raise ValueError("max_fail_to_pass must be at least 1")
    return [record for record in records if fail_to_pass_count(record) <= max_fail_to_pass]


def filter_to_runnable_records(records: list[dict[str, Any]], image_exists: Any) -> list[dict[str, Any]]:
    """Keep instances whose official SWE evaluation image is already local."""
    return [record for record in records if image_exists(record)]


def classify_repair_shape(record: dict[str, Any]) -> str:
    paths = patch_paths(str(record.get("patch", "")))
    source_paths = [path for path in paths if "/tests/" not in path and not path.startswith("tests/")]
    issue = str(record.get("problem_statement", "")).lower()
    if any(token in issue for token in ("config", "setting", "option", "compatib", "deprecat", "version", "environment")):
        return "state_or_configuration"
    if len(source_paths) >= 2 or any(token in issue for token in ("api", "interface", "parameter", "attribute", "public method")):
        return "cross_file_api_contract"
    if len(json.loads(str(record.get("FAIL_TO_PASS", "[]")))) > 1 or any(token in issue for token in ("traceback", "exception", "error message", "test failure")):
        return "test_or_diagnostic_feedback"
    return "local_logic"


def select_records(records: list[dict[str, Any]], max_fail_to_pass: int = DEFAULT_MAX_FAIL_TO_PASS, *, image_exists: Any | None = None) -> list[dict[str, Any]]:
    records = filter_by_fail_to_pass(records, max_fail_to_pass)
    if image_exists is not None:
        records = filter_to_runnable_records(records, image_exists)
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_repo.setdefault(str(record["repo"]), []).append(record)
    selected: list[dict[str, Any]] = []
    for repo, quota in REPOSITORY_QUOTAS.items():
        candidates = sorted(by_repo.get(repo, []), key=lambda item: str(item["instance_id"]))
        shape_counts: Counter[str] = Counter()
        chosen: list[dict[str, Any]] = []
        while candidates and len(chosen) < quota:
            candidates.sort(key=lambda item: (shape_counts[classify_repair_shape(item)], str(item["instance_id"])))
            candidate = candidates.pop(0)
            shape_counts[classify_repair_shape(candidate)] += 1
            chosen.append(candidate)
        if len(chosen) != quota:
            raise ValueError(f"insufficient records for {repo}: wanted {quota}, found {len(chosen)}")
        selected.extend(chosen)
    return selected


def manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": record["instance_id"],
        "repo": record["repo"],
        "domain_stratum": repository_domain(str(record["repo"])),
        "repair_shape_heuristic": classify_repair_shape(record),
        "patch_file_count": len(patch_paths(str(record.get("patch", "")))),
        "fail_to_pass_count": fail_to_pass_count(record),
        "docker_image": swe_instance_image_from_id(str(record["instance_id"])),
        "issue_excerpt": str(record["problem_statement"]).replace("\n", " ")[:280],
    }


def swe_instance_image_from_id(instance_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{instance_id.lower().replace('__', '_1776_')}:latest"


def local_official_image_exists(record: dict[str, Any]) -> bool:
    completed = subprocess.run(
        ["docker", "image", "inspect", swe_instance_image_from_id(str(record["instance_id"]))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced SWE-bench Verified main-experiment manifest.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-fail-to-pass", type=int, default=DEFAULT_MAX_FAIL_TO_PASS)
    parser.add_argument("--only-local-images", action="store_true", help="Select only instances whose official image is cached locally.")
    args = parser.parse_args()
    records = pq.read_table(args.dataset).to_pylist()
    selected = select_records(
        records,
        args.max_fail_to_pass,
        image_exists=local_official_image_exists if args.only_local_images else None,
    )
    manifest = {
        "selection_design": "30 candidates balanced across repository domains and repair-shape heuristics",
        "label_note": "repair_shape_heuristic is derived from issue and patch metadata, not an official SWE-bench label",
        "max_fail_to_pass": args.max_fail_to_pass,
        "only_local_images": args.only_local_images,
        "repository_quotas": REPOSITORY_QUOTAS,
        "records": [manifest_record(record) for record in selected],
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(selected), "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
