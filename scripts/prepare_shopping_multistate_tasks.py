"""Real HTTP-only transition gate; this is not LLM clean admission."""
import argparse
import json
from pathlib import Path

from mas_faults.multimechanism_matrix import config_digest
from mas_faults.shopping_action_protocol import MultiStateShoppingExecutor, ActionLedger


def task_variants(product, index):
    transitions = [((1, 2), (3, 1)), ((1, 3), (3, 2)), ((2, 3), (2, 1))]
    return [{**product, "task_id": f"{product['task_id']}-q{start}to{end}",
             "initial_quantity": start, "quantity": end,
             "task_family": "quantity_increase" if start < end else "quantity_decrease"}
            for start, end in transitions[index % len(transitions)]]


def assert_transition(task, observed):
    if (observed.get("task_id") != task["task_id"] or observed.get("product_title") != task["product_title"]
            or observed.get("observed_quantity") != task["quantity"] or observed.get("cart_verified") is not True
            or not observed.get("sku") or not observed.get("product_id")):
        raise ValueError("real Shopping transition did not satisfy target")


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:17770")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--products", default=5, type=int)
    args = parser.parse_args()
    if not 2 <= args.products <= 10:
        raise ValueError("product gate bound must be 2..10")
    args.output.mkdir(parents=True, exist_ok=False)
    candidates = MultiStateShoppingExecutor(args.base_url).discover_tasks(20)
    admitted, product_count = [], 0
    for product in candidates:
        variants = task_variants(product, product_count)
        passed = []
        for task in variants:
            executor = MultiStateShoppingExecutor(args.base_url)
            try:
                initial = {**task, "quantity": task["initial_quantity"]}
                executor.add_to_cart(initial)
                assert_transition(initial, executor.reobserve_cart(initial))
                if task["quantity"] > task["initial_quantity"]:
                    delta = task["quantity"] - task["initial_quantity"]
                    executor.add_quantity(task, delta)
                else:
                    executor.set_quantity(task, task["quantity"])
                observed = executor.reobserve_cart(task)
                assert_transition(task, observed)
                # Demonstrate actual unguarded POST duplication and guarded suppression.
                if task["quantity"] > task["initial_quantity"]:
                    executor.add_quantity(task, 1)
                    higher = {**task, "quantity": task["quantity"] + 1}
                    assert_transition(higher, executor.reobserve_cart(higher))
                    executor.set_quantity(task, task["quantity"])
                    ledger = ActionLedger(args.output / (task["task_id"] + ".sqlite"))
                    params = {"operation": "add_quantity", "quantity": 1}
                    action = lambda: executor.add_quantity(higher, 1)
                    ledger.execute_once(task["task_id"], "gate", "gate-action", params, action)
                    ledger.execute_once(task["task_id"], "gate", "gate-action", params, action)
                    assert_transition(higher, executor.reobserve_cart(higher))
                append(args.output / "gate_attempts.jsonl", {"task": task, "passed": True,
                       "observation": observed, "http_receipts": executor.http_receipts})
                passed.append(task)
            except Exception as exc:
                append(args.output / "gate_attempts.jsonl", {"task": task, "passed": False,
                       "error_type": type(exc).__name__, "http_receipts": executor.http_receipts})
                print(json.dumps({"task_id": task["task_id"], "passed": False, "error_type": type(exc).__name__}), flush=True)
        if len(passed) == len(variants):
            admitted.extend(passed)
            product_count += 1
        if product_count == args.products:
            break
    result = {"tasks": admitted, "task_count": len(admitted), "product_count": product_count,
              "status": "passed" if product_count == args.products else "insufficient",
              "gate_type": "real_http_transitions_not_llm_clean", "candidate_count": len(candidates)}
    result["manifest_sha256"] = config_digest(result)
    with (args.output / "tasks.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
