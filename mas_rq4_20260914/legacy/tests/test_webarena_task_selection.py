from __future__ import annotations

import json

from mas_faults.webarena_task_selection import build_task_manifest, filter_preflight_products, load_task_manifest


def test_manifest_expands_real_products_into_unique_quantity_tasks():
    products = [
        {"task_id": "shopping-001", "product_title": "Tea", "product_url": "http://shop/tea", "quantity": 1},
        {"task_id": "shopping-002", "product_title": "Coffee", "product_url": "http://shop/coffee", "quantity": 1},
    ]

    manifest = build_task_manifest(products, task_count=5, quantities=(1, 2, 3))

    assert len(manifest["tasks"]) == 5
    assert len({task["task_id"] for task in manifest["tasks"]}) == 5
    assert {task["product_title"] for task in manifest["tasks"]} == {"Tea", "Coffee"}
    assert {task["quantity"] for task in manifest["tasks"]} == {1, 2, 3}
    assert manifest["selection_design"] == "product_identity_x_requested_quantity"


def test_loader_preserves_frozen_task_ids_and_quantities(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": [{"task_id": "shopping-004-q2", "product_title": "Sprinkles", "product_url": "http://shop/item", "quantity": 2}]}), encoding="utf-8")

    tasks = load_task_manifest(path)

    assert tasks == [{"task_id": "shopping-004-q2", "product_title": "Sprinkles", "product_url": "http://shop/item", "quantity": 2}]


def test_preflight_filter_excludes_products_that_cannot_be_added_to_cart():
    products = [{"task_id": "shopping-001"}, {"task_id": "shopping-002"}]

    eligible, skipped = filter_preflight_products(products, lambda product: product["task_id"] == "shopping-001")

    assert [product["task_id"] for product in eligible] == ["shopping-001"]
    assert skipped == ["shopping-002"]
