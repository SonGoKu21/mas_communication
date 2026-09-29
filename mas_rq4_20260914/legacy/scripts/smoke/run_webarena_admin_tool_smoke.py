#!/usr/bin/env python3
"""真实 Shopping Admin 只读工具与独立 evaluator 的无模型 smoke。"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from mas_faults.webarena_admin_controlled import (
    EvaluatorWorkerClient,
    write_sanitized_browser_config,
)
from mas_faults.webarena_admin_real import BrowserWorkerClient
from run_webarena_admin_controlled_clean import browser_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(
            "/data2/system5/mas/task_configs/"
            "webarena_shopping_admin_verified_20260814"
        ),
    )
    parser.add_argument(
        "--shopping-admin-url",
        default="http://10.102.35.120:7780/admin",
    )
    return parser.parse_args()


def run_tools(config: Path, tools: list[tuple[str, dict[str, str]]], env: dict[str, str]):
    browser = BrowserWorkerClient(browser_only=True, env=env)
    try:
        browser.reset(str(config))
        result = {}
        for name, arguments in tools:
            result = browser.tool(name, arguments)
        return result
    finally:
        browser.close()


def main() -> None:
    args = parse_args()
    environment = browser_environment(args.shopping_admin_url)
    with tempfile.TemporaryDirectory(prefix="admin-tool-smoke-") as temp:
        temp_path = Path(temp)
        task0 = args.config_dir / "0.json"
        task184 = args.config_dir / "184.json"
        browser0 = write_sanitized_browser_config(task0, temp_path / "task0")
        browser184 = write_sanitized_browser_config(task184, temp_path / "task184")

        bestseller = run_tools(
            browser0.path,
            [
                ("open_bestsellers_report", {}),
                (
                    "set_date_range",
                    {
                        "from_date": "01/01/2022",
                        "to_date": "12/31/2022",
                        "period": "Year",
                    },
                ),
                ("show_report", {}),
                ("read_visible_table", {}),
            ],
            environment,
        )
        inventory = run_tools(
            browser184.path,
            [
                ("open_catalog_products", {}),
                ("reveal_data_grid", {}),
                ("open_filters", {}),
                (
                    "set_text_filter",
                    {"field": "Name", "value": "Sinbad Fitness Tank"},
                ),
                ("apply_filters", {}),
                ("read_visible_table", {}),
            ],
            environment,
        )
        bestseller_rows = bestseller["visible_evidence"]
        inventory_rows = inventory["visible_evidence"]
        if not any("Quest Lumaflex" in " ".join(row) for row in bestseller_rows):
            raise AssertionError("task 0 真实报表证据缺少预期 calibration 行")
        if not any(
            len(row) > 8
            and row[3] == "Sinbad Fitness Tank"
            and row[8] == "0.0000"
            for row in inventory_rows[1:]
        ):
            raise AssertionError("task 184 真实商品证据没有稳定返回 Quantity=0")

        evaluator = EvaluatorWorkerClient(env=environment)
        try:
            evaluation = evaluator.evaluate(str(task0), "Quest Lumaflex™ Band")
        finally:
            evaluator.close()
        if evaluation["score"] != 1.0:
            raise AssertionError("task 0 官方 evaluator 接线失败")

        print(
            json.dumps(
                {
                    "task0_rows": len(bestseller_rows),
                    "task0_first_data_row": bestseller_rows[2],
                    "task0_official_score": evaluation["score"],
                    "task184_rows": len(inventory_rows),
                    "task184_first_data_row": inventory_rows[1],
                    "browser_config_hashes": [browser0.sha256, browser184.sha256],
                    "wiring_only": True,
                    "counted_as_clean_admission": False,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
