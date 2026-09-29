"""Run SWE-bench's official evaluator without fetching setup files online.

The evaluator still builds a ``TestSpec`` when an exact local instance image
already exists.  That setup path fetches environment files from GitHub even
though the cached image makes them unnecessary.  This wrapper permits that
offline path only after verifying the exact image is available locally.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence


def cached_instance_image_name(instance_id: str, namespace: str = "swebench") -> str:
    key = f"sweb.eval.x86_64.{instance_id.lower()}:latest".replace("__", "_1776_")
    return f"{namespace}/{key}" if namespace else key


def parse_wrapper_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--instance_ids", nargs="+", required=True)
    parser.add_argument("--namespace", default="swebench")
    return parser.parse_known_args(argv)[0]


def require_cached_instance_images(instance_ids: Sequence[str], namespace: str) -> None:
    import docker

    client = docker.from_env()
    missing: list[str] = []
    for instance_id in instance_ids:
        image_name = cached_instance_image_name(instance_id, namespace)
        try:
            client.images.get(image_name)
        except docker.errors.ImageNotFound:
            missing.append(image_name)
    if missing:
        raise SystemExit(
            "offline SWE evaluator requires cached instance images; missing: "
            + ", ".join(missing)
        )


def install_offline_setup_stubs() -> None:
    """Keep unused image-build setup scripts from issuing GitHub requests."""
    from swebench.harness.test_spec import python as python_spec

    python_spec.get_environment_yml_by_commit = lambda _repo, _commit, env_name: f"name: {env_name}\n"
    python_spec.get_requirements_by_commit = lambda _repo, _commit: ""


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parsed = parse_wrapper_args(args)
    require_cached_instance_images(parsed.instance_ids, parsed.namespace)
    install_offline_setup_stubs()
    sys.argv = ["swebench.harness.run_evaluation", *args]
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")


if __name__ == "__main__":
    main()
