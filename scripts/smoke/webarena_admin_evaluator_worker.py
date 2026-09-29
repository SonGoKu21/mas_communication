#!/usr/bin/env python3
"""Isolated JSONL process for deterministic official WebArena evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, TextIO


MONTH_ALIASES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "sepetember": 9,
    "oct": 10,
    "october": 10,
    "octorbor": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _month_counts(text: str) -> dict[int, int]:
    token = r"(?:[A-Za-z]+|0?[1-9]|1[0-2])"
    matches = re.findall(
        rf"\b({token})(?:/\d{{4}})?\s*[:=\-]\s*(\d+)\b",
        text,
        flags=re.IGNORECASE,
    )
    result: dict[int, int] = {}
    for raw_month, raw_count in matches:
        lowered = raw_month.lower()
        month = MONTH_ALIASES.get(lowered)
        if month is None and raw_month.isdigit():
            month = int(raw_month)
        if month is not None:
            result[month] = int(raw_count)
    return result


def deterministic_fuzzy_score(config: dict[str, Any], answer: str) -> float:
    """Evaluate the benchmark's structured month-count fuzzy task without an LLM judge."""
    intent = str(config.get("intent", "")).lower()
    references = config.get("eval", {}).get("reference_answers", {}).get(
        "fuzzy_match"
    )
    if "monthly count" not in intent or "mm:count" not in intent:
        raise ValueError("controlled evaluator has no deterministic fuzzy adapter")
    if not isinstance(references, list) or not references:
        raise ValueError("deterministic fuzzy reference must be a non-empty list")
    expected = _month_counts(" ".join(str(value) for value in references))
    observed = _month_counts(answer)
    if not expected:
        raise ValueError("deterministic fuzzy reference contains no month counts")
    return float(all(observed.get(month) == count for month, count in expected.items()))


class OfficialEvaluatorRuntime:
    def __init__(self, webarena_root: Path):
        self.webarena_root = webarena_root.resolve()
        os.chdir(self.webarena_root)
        sys.path.insert(0, str(self.webarena_root))

        from browser_env import create_stop_action
        from evaluation_harness.evaluators import evaluator_router
        from playwright.sync_api import sync_playwright

        self.create_stop_action = create_stop_action
        self.evaluator_router = evaluator_router
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.client = self.context.new_cdp_session(self.page)

    def evaluate(self, config_file: str, answer: str) -> dict[str, Any]:
        path = Path(config_file).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        eval_config = payload.get("eval", {})
        if eval_config.get("eval_types") != ["string_match"]:
            raise ValueError("controlled evaluator requires string_match")
        approaches = set(eval_config.get("reference_answers", {}))
        if approaches == {"fuzzy_match"}:
            return {
                "score": deterministic_fuzzy_score(payload, answer),
                "answer": answer,
                "evaluator_mode": "deterministic_structured_fuzzy_adapter",
            }
        if not approaches or not approaches <= {"exact_match", "must_include"}:
            raise ValueError("controlled evaluator requires deterministic reference answers")
        evaluator = self.evaluator_router(str(path))
        score = evaluator(
            trajectory=[self.create_stop_action(answer)],
            config_file=str(path),
            page=self.page,
            client=self.client,
        )
        return {
            "score": float(score),
            "answer": answer,
            "evaluator_mode": "official_webarena_string_match",
        }

    def close(self) -> dict[str, Any]:
        self.context.close()
        self.browser.close()
        self.playwright.stop()
        return {"closed": True}


def dispatch(runtime: Any, payload: dict[str, Any]) -> dict[str, Any]:
    command = payload.get("command")
    if command == "evaluate":
        return runtime.evaluate(str(payload["config_file"]), str(payload["answer"]))
    if command == "close":
        return runtime.close()
    raise ValueError(f"unsupported evaluator command: {command!r}")


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
    args = parser.parse_args()
    serve_commands(sys.stdin, sys.stdout, OfficialEvaluatorRuntime(args.webarena_root))


if __name__ == "__main__":
    main()
