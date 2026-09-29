#!/usr/bin/env python3
"""JSONL bridge to the official WebArena browser environment and evaluator."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mas_faults.webarena_admin_tools import validate_admin_tool_call


FILTER_BUTTON_SELECTOR = ".data-grid-filters-action-wrap"
REPORT_DATE_SELECTORS = {
    "from_date": "input[name='from']",
    "to_date": "input[name='to']",
    "period": "select[name='period_type']",
}
GRID_LOADING_MASK_SELECTOR = ".admin__data-grid-loading-mask"


def available_tools_for_page(url: str, title: str) -> tuple[str, ...]:
    """Expose only tools whose DOM contract is valid on the current Magento page."""
    if (
        "/reports/report_sales/bestsellers/" in url
        or "/reports/report_sales/sales/" in url
    ):
        return (
            "set_date_range",
            "show_report",
            "reveal_data_grid",
            "read_visible_table",
            "read_table_head",
            "finish_with_evidence",
        )
    if "/sales/order/" in url:
        return (
            "open_filters",
            "set_range_filter",
            "set_select_filter",
            "apply_filters",
            "reveal_data_grid",
            "read_visible_table",
            "read_table_head",
            "finish_with_evidence",
        )
    if "/catalog/product/" in url:
        return (
            "open_filters",
            "set_text_filter",
            "set_range_filter",
            "apply_filters",
            "reveal_data_grid",
            "read_visible_table",
            "read_table_head",
            "finish_with_evidence",
        )
    if "/customer/index/" in url:
        return (
            "open_filters",
            "set_text_filter",
            "apply_filters",
            "reveal_data_grid",
            "read_visible_table",
            "read_table_head",
            "finish_with_evidence",
        )
    return (
        "open_catalog_products",
        "open_customers",
        "open_sales_orders",
        "open_sales_orders_report",
        "open_bestsellers_report",
    )


def navigation_name_pattern(label: str) -> re.Pattern[str]:
    """Match a Magento menu label with its optional private-use icon glyph."""
    return re.compile(rf"^(?:[\ue000-\uf8ff]\s*)?{re.escape(label)}$", re.IGNORECASE)


def should_capture_visible_table(tool_name: str) -> bool:
    """Large grids are read only when the Navigator explicitly requests evidence."""
    return tool_name in {"read_visible_table", "read_table_head"}


def text_filter_selectors(url: str, field: str) -> tuple[str, ...]:
    if "/customer/index/" in url:
        selectors = {
            "Name": ("input[name='name']",),
            "Email": ("input[name='email']",),
            "Phone": ("input[name='billing_telephone']",),
        }
    else:
        selectors = {
            "Quantity": ("input[name='qty[from]']", "input[name='qty[to]']"),
            "SKU": ("input[name='sku']",),
            "Name": ("input[name='name']",),
        }
    canonical_field = next(
        (name for name in selectors if name.casefold() == field.casefold()),
        None,
    )
    if canonical_field is None:
        raise ValueError(f"unsupported exact text filter field: {field!r}")
    return selectors[canonical_field]


def normalize_text_filter_value(field: str, value: str) -> str:
    if field.casefold() != "phone":
        return value
    digits = re.sub(r"\D", "", value)
    return digits[1:] if len(digits) == 11 and digits.startswith("1") else digits


def lightweight_observation(url: str, title: str) -> str:
    return f"TITLE: {title}\nURL: {url}"


def expand_table_cells(raw_rows: list[list[Any]]) -> list[list[str]]:
    """Reconstruct a rectangular table from visible cells and HTML spans."""
    if all(all(isinstance(cell, str) for cell in row) for row in raw_rows):
        return [[str(cell) for cell in row] for row in raw_rows]
    active: dict[int, tuple[str, int]] = {}
    expanded: list[list[str]] = []
    for cells in raw_rows:
        row_values: dict[int, str] = {}
        next_active: dict[int, tuple[str, int]] = {}
        for column, (text, remaining) in active.items():
            row_values[column] = text
            if remaining > 1:
                next_active[column] = (text, remaining - 1)
        column = 0
        for cell in cells:
            while column in row_values:
                column += 1
            text = str(cell.get("text", ""))
            row_span = max(1, int(cell.get("row_span", 1)))
            col_span = max(1, int(cell.get("col_span", 1)))
            for offset in range(col_span):
                value = text if offset == 0 else ""
                target = column + offset
                row_values[target] = value
                if row_span > 1:
                    next_active[target] = (value, row_span - 1)
            column += col_span
        if row_values:
            expanded.append(
                [row_values.get(index, "") for index in range(max(row_values) + 1)]
            )
        active = next_active
    return expanded


def extract_visible_table(page: Any, *, limit: int = 200) -> list[list[str]]:
    tables = page.locator("table:visible")
    count = tables.count()
    if count == 0:
        return []
    rows = tables.nth(count - 1).locator("tr:visible").evaluate_all(
        """
        (rows, limit) => rows.slice(0, limit).map((row) =>
          Array.from(row.querySelectorAll(':scope > th, :scope > td'))
            .filter((cell) => {
              const style = getComputedStyle(cell);
              const rect = cell.getBoundingClientRect();
              return style.display !== 'none' && style.visibility !== 'hidden' &&
                rect.width > 0 && rect.height > 0;
            })
            .map((cell) => ({
              text: (cell.innerText || '').trim(),
              row_span: cell.rowSpan || 1,
              col_span: cell.colSpan || 1
            }))
        ).filter((row) => row.length > 0)
        """,
        limit + 1,
    )
    return expand_table_cells(rows)


def parse_select_action(action: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"select \[(\d+)\] (?:\[([^\r\n]+)\]|([^\[\]\r\n]+))",
        action.strip(),
    )
    if not match:
        return None
    return match.group(1), (match.group(2) or match.group(3)).strip()


def combobox_nodes(observation: str) -> list[tuple[str, str]]:
    nodes = []
    for line in observation.splitlines():
        match = re.match(r"\s*\[(\d+)\] combobox '(.*)'(?:\s|$)", line)
        if match:
            nodes.append((match.group(1), match.group(2)))
    return nodes


class OfficialBrowserRuntime:
    """Own one official ScriptBrowserEnv without importing it in the MAS process."""

    def __init__(self, webarena_root: Path, *, enable_evaluator: bool = True):
        self.webarena_root = webarena_root.resolve()
        os.chdir(self.webarena_root)
        sys.path.insert(0, str(self.webarena_root))

        from browser_env import ScriptBrowserEnv, create_none_action, create_stop_action
        from browser_env.actions import ActionTypes, create_id_based_action
        from browser_env.utils import DetachedPage

        self.ActionTypes = ActionTypes
        self.create_id_based_action = create_id_based_action
        self.create_none_action = create_none_action
        self.create_stop_action = create_stop_action
        self.DetachedPage = DetachedPage
        self.evaluator_router = None
        self.enable_evaluator = enable_evaluator
        if enable_evaluator:
            from evaluation_harness.evaluators import evaluator_router

            self.evaluator_router = evaluator_router
        self.env = ScriptBrowserEnv(
            headless=True,
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720},
            save_trace_enabled=False,
            sleep_after_execution=0.25,
        )
        self.config_file: str | None = None
        self.trajectory: list[dict[str, Any]] = []
        self.state: dict[str, Any] | None = None

    def _state_response(self, *, stopped: bool = False, answer: str = "") -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("browser environment has not been reset")
        observation = str(self.state["observation"]["text"])
        option_lines = []
        for element_id, name in combobox_nodes(observation):
            locator = self._combobox_locator(name)
            if locator.count() != 1:
                continue
            options = [text.strip() for text in locator.locator("option").all_text_contents() if text.strip()]
            selected = [text.strip() for text in locator.locator("option:checked").all_text_contents() if text.strip()]
            if options:
                option_lines.append(
                    f"[{element_id}] combobox options: {json.dumps(options, ensure_ascii=False)}; "
                    f"selected: {json.dumps(selected, ensure_ascii=False)}"
                )
        if option_lines:
            observation += "\n\nCOMBOBOX OPTIONS:\n" + "\n".join(option_lines)
        return {
            "observation": observation,
            "url": self.env.page.url,
            "title": self.env.page.title(),
            "available_tools": list(
                available_tools_for_page(self.env.page.url, self.env.page.title())
            ),
            "stopped": stopped,
            "answer": answer,
            "fail_error": str(self.state["info"].get("fail_error", "")),
        }

    def _combobox_locator(self, name: str):
        labelled = self.env.page.get_by_label(name, exact=True)
        if labelled.count() == 1:
            return labelled
        return self.env.page.get_by_role("combobox", name=name, exact=True)

    def _one_visible(self, locator: Any, description: str) -> Any:
        visible = [locator.nth(index) for index in range(locator.count()) if locator.nth(index).is_visible()]
        if len(visible) != 1:
            raise ValueError(f"expected one visible {description}, found {len(visible)}")
        return visible[0]

    def _sync_state(self, raw_prediction: str) -> dict[str, Any]:
        if not self.enable_evaluator:
            self.state = {
                "observation": {
                    "text": lightweight_observation(
                        self.env.page.url,
                        self.env.page.title(),
                    )
                },
                "info": {"fail_error": ""},
            }
            return self._state_response()
        parsed = self.create_none_action()
        parsed["raw_prediction"] = raw_prediction
        self.trajectory.append(parsed)
        observation = self.env._get_obs()
        info = {
            "page": self.DetachedPage(self.env.page.url, self.env.page.content()),
            "fail_error": "",
            "observation_metadata": self.env._get_obs_metadata(),
        }
        self.state = {"observation": observation, "info": info}
        self.trajectory.append(self.state)
        return self._state_response()

    def _click_role(self, role: str, name: str, subtrace: list[dict[str, Any]]) -> None:
        locator = self._one_visible(
            self.env.page.get_by_role(role, name=name, exact=True),
            f"{role} named {name!r}",
        )
        before_url = self.env.page.url
        locator.click()
        self.env.page.wait_for_timeout(250)
        subtrace.append(
            {
                "operation": "click_role",
                "role": role,
                "name": name,
                "before_url": before_url,
                "after_url": self.env.page.url,
            }
        )

    def _click_navigation(self, name: str, subtrace: list[dict[str, Any]]) -> None:
        locator = self._one_visible(
            self.env.page.get_by_role("link", name=navigation_name_pattern(name)),
            f"navigation link named {name!r}",
        )
        before_url = self.env.page.url
        locator.click()
        self.env.page.wait_for_timeout(250)
        subtrace.append(
            {
                "operation": "click_navigation",
                "name": name,
                "before_url": before_url,
                "after_url": self.env.page.url,
            }
        )

    def _fill_selector(
        self, selector: str, value: str, subtrace: list[dict[str, Any]]
    ) -> None:
        locator = self._one_visible(self.env.page.locator(selector), selector)
        locator.fill(value)
        subtrace.append(
            {"operation": "fill", "selector": selector, "value": value}
        )

    def _click_selector(
        self, selector: str, subtrace: list[dict[str, Any]]
    ) -> None:
        locator = self._one_visible(self.env.page.locator(selector), selector)
        before_url = self.env.page.url
        locator.click()
        self.env.page.wait_for_timeout(250)
        subtrace.append(
            {
                "operation": "click_selector",
                "selector": selector,
                "before_url": before_url,
                "after_url": self.env.page.url,
            }
        )

    def _clear_active_grid_filters(self, subtrace: list[dict[str, Any]]) -> None:
        locator = self.env.page.get_by_role(
            "button", name="Clear all", exact=True
        )
        visible = [
            locator.nth(index)
            for index in range(locator.count())
            if locator.nth(index).is_visible()
        ]
        if len(visible) > 1:
            raise ValueError(
                f"expected at most one visible Clear all button, found {len(visible)}"
            )
        if not visible:
            subtrace.append(
                {"operation": "normalize_grid_filters", "result": "already_clear"}
            )
            return
        visible[0].click()
        self.env.page.wait_for_timeout(250)
        subtrace.append(
            {"operation": "normalize_grid_filters", "result": "cleared"}
        )

    def _wait_for_grid_ready(self, subtrace: list[dict[str, Any]]) -> None:
        timeout_ms = 10000
        poll_ms = 250
        stable_ms = 750
        stable_checks_required = (stable_ms // poll_ms) + 1
        max_checks = (timeout_ms // poll_ms) + 1
        stable_checks = 0

        for check_index in range(max_checks):
            loading_masks = self.env.page.locator(GRID_LOADING_MASK_SELECTOR)
            mask_visible = any(
                loading_masks.nth(index).is_visible()
                for index in range(loading_masks.count())
            )
            stable_checks = 0 if mask_visible else stable_checks + 1
            if stable_checks >= stable_checks_required:
                subtrace.append(
                    {
                        "operation": "wait_for_hidden",
                        "selector": GRID_LOADING_MASK_SELECTOR,
                        "timeout_ms": timeout_ms,
                        "stable_ms": stable_ms,
                    }
                )
                return
            if check_index < max_checks - 1:
                self.env.page.wait_for_timeout(poll_ms)

        raise TimeoutError(
            f"grid loading mask did not remain hidden for {stable_ms}ms "
            f"within {timeout_ms}ms"
        )

    def _normalize_opened_grid(self, subtrace: list[dict[str, Any]]) -> None:
        self._wait_for_grid_ready(subtrace)
        self._clear_active_grid_filters(subtrace)

    def _visible_table(self, *, limit: int = 200) -> list[list[str]]:
        return extract_visible_table(self.env.page, limit=limit)

    def reset(self, config_file: str) -> dict[str, Any]:
        path = Path(config_file).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        observation, info = self.env.reset(options={"config_file": str(path)})
        self.config_file = str(path)
        self.state = {"observation": observation, "info": info}
        self.trajectory = [self.state]
        return self._state_response()

    def step(self, action: str) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        select_action = parse_select_action(action)
        if select_action is not None:
            element_id, option = select_action
            observation = str(self.state["observation"]["text"])
            names = dict(combobox_nodes(observation))
            if element_id not in names:
                raise ValueError(f"combobox element {element_id} is not present in the observation")
            locator = self._combobox_locator(names[element_id])
            if locator.count() != 1:
                raise ValueError(
                    f"expected one combobox named {names[element_id]!r}, found {locator.count()}"
                )
            locator.select_option(label=option)
            self.env.page.wait_for_timeout(250)
            parsed = self.create_none_action()
            parsed["raw_prediction"] = action
            self.trajectory.append(parsed)
            observation_value = self.env._get_obs()
            info = {
                "page": self.DetachedPage(self.env.page.url, self.env.page.content()),
                "fail_error": "",
                "observation_metadata": self.env._get_obs_metadata(),
            }
            self.state = {"observation": observation_value, "info": info}
            self.trajectory.append(self.state)
            return self._state_response()
        parsed = self.create_id_based_action(action)
        parsed["raw_prediction"] = action
        self.trajectory.append(parsed)
        if parsed["action_type"] == self.ActionTypes.STOP:
            return self._state_response(stopped=True, answer=str(parsed["answer"]))

        observation, _, _, _, info = self.env.step(parsed)
        self.state = {"observation": observation, "info": info}
        self.trajectory.append(self.state)
        return self._state_response()

    def tool(self, name: object, arguments: object) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("reset must be called before tool")
        request = validate_admin_tool_call(name, arguments)
        subtrace: list[dict[str, Any]] = []

        if request.name == "open_catalog_products":
            self._click_navigation("Catalog", subtrace)
            self._click_role("link", "Products", subtrace)
            self._normalize_opened_grid(subtrace)
        elif request.name == "open_customers":
            self._click_navigation("Customers", subtrace)
            self._click_role("link", "All Customers", subtrace)
            self._normalize_opened_grid(subtrace)
        elif request.name == "open_sales_orders":
            self._click_navigation("Sales", subtrace)
            self._click_role("link", "Orders", subtrace)
            self._normalize_opened_grid(subtrace)
        elif request.name == "open_sales_orders_report":
            self._click_navigation("Reports", subtrace)
            self._click_role("link", "Orders", subtrace)
        elif request.name == "open_bestsellers_report":
            self._click_navigation("Reports", subtrace)
            self._click_role("link", "Bestsellers", subtrace)
        elif request.name == "reveal_data_grid":
            self.env.page.mouse.wheel(0, 900)
            self.env.page.wait_for_timeout(250)
            subtrace.append({"operation": "scroll", "delta_y": 900})
        elif request.name == "open_filters":
            self._wait_for_grid_ready(subtrace)
            self.env.page.locator(f"{FILTER_BUTTON_SELECTOR}:visible").first.wait_for(
                state="visible",
                timeout=10000,
            )
            subtrace.append(
                {
                    "operation": "wait_for_visible",
                    "selector": FILTER_BUTTON_SELECTOR,
                    "timeout_ms": 10000,
                }
            )
            self._click_selector(FILTER_BUTTON_SELECTOR, subtrace)
            self._clear_active_grid_filters(subtrace)
        elif request.name == "set_text_filter":
            field = request.arguments["field"]
            value = normalize_text_filter_value(field, request.arguments["value"])
            for selector in text_filter_selectors(self.env.page.url, field):
                self._fill_selector(selector, value, subtrace)
        elif request.name == "set_range_filter":
            field = request.arguments["field"]
            selectors = {
                "Quantity": ("input[name='qty[from]']", "input[name='qty[to]']"),
                "Price": ("input[name='price[from]']", "input[name='price[to]']"),
                "ID": ("input[name='entity_id[from]']", "input[name='entity_id[to]']"),
            }
            if field not in selectors:
                raise ValueError(f"unsupported exact range filter field: {field!r}")
            for selector, value in zip(
                selectors[field],
                (request.arguments["from_value"], request.arguments["to_value"]),
            ):
                self._fill_selector(selector, value, subtrace)
        elif request.name == "set_select_filter":
            field = request.arguments["field"]
            value = request.arguments["value"]
            selectors = {"Status": "select[name='status']"}
            if field not in selectors:
                raise ValueError(f"unsupported exact select filter field: {field!r}")
            locator = self._one_visible(
                self.env.page.locator(selectors[field]),
                selectors[field],
            )
            locator.select_option(label=value)
            subtrace.append(
                {
                    "operation": "select",
                    "selector": selectors[field],
                    "field": field,
                    "value": value,
                }
            )
        elif request.name == "set_date_range":
            for key in ("from_date", "to_date"):
                self._fill_selector(
                    REPORT_DATE_SELECTORS[key],
                    request.arguments[key],
                    subtrace,
                )
            period = request.arguments["period"]
            locator = self._one_visible(
                self.env.page.locator(REPORT_DATE_SELECTORS["period"]),
                REPORT_DATE_SELECTORS["period"],
            )
            locator.select_option(label=period)
            subtrace.append(
                {
                    "operation": "select",
                    "selector": REPORT_DATE_SELECTORS["period"],
                    "value": period,
                }
            )
        elif request.name == "apply_filters":
            self._click_role("button", "Apply Filters", subtrace)
        elif request.name == "show_report":
            self._click_role("button", "Show Report", subtrace)
        elif request.name in {"read_visible_table", "read_table_head"}:
            self._wait_for_grid_ready(subtrace)
            self.env.page.wait_for_function(
                """
                () => Array.from(document.querySelectorAll('table')).some((table) => {
                  const tableRect = table.getBoundingClientRect();
                  if (tableRect.width <= 0 || tableRect.height <= 0) return false;
                  const rows = Array.from(table.querySelectorAll('tr')).filter((row) => {
                    const rect = row.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                  });
                  if (rows.length < 2) return false;
                  return true;
                })
                """,
                timeout=10000,
            )
            subtrace.append(
                {
                    "operation": "wait_for_table_rows",
                    "minimum_visible_rows": 2,
                    "timeout_ms": 10000,
                }
            )
        elif request.name == "finish_with_evidence":
            pass
        else:  # pragma: no cover - validator and exhaustive branches guard this.
            raise ValueError(f"unsupported tool: {request.name!r}")

        state = self._sync_state(
            json.dumps(
                {"tool": request.name, "arguments": request.arguments},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return {
            **state,
            "tool_status": "ok",
            "tool_name": request.name,
            "tool_arguments": request.arguments,
            "visible_evidence": (
                self._visible_table(
                    limit=(
                        int(request.arguments["max_rows"])
                        if request.name == "read_table_head"
                        else 200
                    )
                )
                if should_capture_visible_table(request.name)
                else []
            ),
            "tool_subtrace": subtrace,
        }

    def evaluate(self, answer: str | None = None) -> dict[str, Any]:
        if self.config_file is None or self.state is None:
            raise RuntimeError("reset must be called before evaluate")
        if self.evaluator_router is None:
            raise RuntimeError("evaluator is disabled in browser-only mode")
        trajectory = list(self.trajectory)
        if answer is not None:
            if trajectory and "action_type" in trajectory[-1]:
                trajectory.pop()
            trajectory.append(self.create_stop_action(answer))
        elif not trajectory or "action_type" not in trajectory[-1]:
            raise RuntimeError("evaluate requires a stop action or an answer override")
        evaluator = self.evaluator_router(self.config_file)
        score = evaluator(
            trajectory=trajectory,
            config_file=self.config_file,
            page=self.env.page,
            client=self.env.get_page_client(self.env.page),
        )
        return {"score": float(score), "answer": answer}

    def close(self) -> dict[str, Any]:
        self.env.close()
        return {"closed": True}


def dispatch(runtime: Any, payload: dict[str, Any]) -> dict[str, Any]:
    command = payload.get("command")
    if command == "reset":
        return runtime.reset(str(payload["config_file"]))
    if command == "step":
        return runtime.step(str(payload["action"]))
    if command == "tool":
        return runtime.tool(payload.get("tool"), payload.get("arguments"))
    if command == "evaluate":
        answer = payload.get("answer")
        return runtime.evaluate(None if answer is None else str(answer))
    if command == "close":
        return runtime.close()
    raise ValueError(f"unsupported worker command: {command!r}")


def serve_commands(reader: TextIO, writer: TextIO, runtime: Any) -> None:
    for line in reader:
        if not line.strip():
            continue
        payload: dict[str, Any] = {}
        try:
            payload = json.loads(line)
            response = {"ok": True, **dispatch(runtime, payload)}
        except Exception as exc:
            response = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        writer.write(json.dumps(response, ensure_ascii=False) + "\n")
        writer.flush()
        if payload.get("command") == "close":
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--webarena-root",
        type=Path,
        default=Path("/data2/system5/mas/third_party/webarena"),
    )
    parser.add_argument("--browser-only", action="store_true")
    args = parser.parse_args()
    serve_commands(
        sys.stdin,
        sys.stdout,
        OfficialBrowserRuntime(
            args.webarena_root,
            enable_evaluator=not args.browser_only,
        ),
    )


if __name__ == "__main__":
    main()
