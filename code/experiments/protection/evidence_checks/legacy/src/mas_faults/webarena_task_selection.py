"""Create a reproducible Shopping task manifest from locally deployed products."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def build_task_manifest(
    products: Iterable[dict[str, Any]], *, task_count: int, quantities: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    if task_count < 1:
        raise ValueError("task_count must be at least 1")
    candidates = list(products)
    if not candidates:
        raise ValueError("at least one deployed Shopping product is required")
    if not quantities or any(quantity < 1 for quantity in quantities):
        raise ValueError("quantities must contain positive integers")

    tasks: list[dict[str, Any]] = []
    for quantity in quantities:
        for product_index, product in enumerate(candidates, start=1):
            task = copy.deepcopy(product)
            task["task_id"] = f"shopping-{product_index:03d}-q{quantity}"
            task["quantity"] = quantity
            task["task_stratum"] = "product_identity_x_requested_quantity"
            tasks.append(task)
            if len(tasks) == task_count:
                return {
                    "benchmark": "WebArena-Verified-Shopping",
                    "selection_design": "product_identity_x_requested_quantity",
                    "candidate_products": len(candidates),
                    "quantities": list(quantities),
                    "tasks": tasks,
                }
    raise ValueError(
        f"requested {task_count} tasks but {len(candidates)} products x {len(quantities)} quantities only provides {len(tasks)}"
    )


def filter_preflight_products(
    products: Iterable[dict[str, Any]], can_add_to_cart: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    eligible: list[dict[str, Any]] = []
    skipped: list[str] = []
    for product in products:
        if can_add_to_cart(product):
            eligible.append(product)
        else:
            skipped.append(str(product.get("task_id", "unknown")))
    return eligible, skipped


def load_task_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"task manifest {path} has no tasks")
    required = {"task_id", "product_title", "product_url", "quantity"}
    for task in tasks:
        if not isinstance(task, dict) or not required.issubset(task):
            raise ValueError(f"task manifest {path} contains an incomplete task")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a WebArena Shopping task manifest from the local site.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7770")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from mas_faults.webarena_shopping_real import ShoppingHTTPExecutor

    products = ShoppingHTTPExecutor(args.base_url).discover_tasks(100)
    preflight_errors: dict[str, str] = {}

    def can_add_to_cart(product: dict[str, Any]) -> bool:
        try:
            ShoppingHTTPExecutor(args.base_url).add_to_cart(product)
        except Exception as exc:
            preflight_errors[str(product["task_id"])] = str(exc)
            return False
        return True

    products, skipped = filter_preflight_products(products, can_add_to_cart)
    manifest = build_task_manifest(products, task_count=args.tasks, quantities=(1, 2, 3, 4))
    manifest["preflight_skipped"] = [{"task_id": task_id, "error": preflight_errors[task_id]} for task_id in skipped]
    manifest["timestamp"] = datetime.now(timezone.utc).isoformat()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(manifest["tasks"]), "output": str(path)}))


if __name__ == "__main__":
    main()
