"""Official WebArena Reddit clean-admission adapter."""

from __future__ import annotations

import csv
import json
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol
from urllib.parse import quote

from autogen_core import AgentId, RoutedAgent, SingleThreadedAgentRuntime, message_handler

from mas_faults.benchmark_trace_contract import normalize_run_record
from mas_faults.webarena_admin_real import extract_browser_action


REDDIT_TASK_STRATA = {
    **{task_id: "comment_state_query" for task_id in range(27, 32)},
    **{task_id: "top_post_semantic_query" for task_id in range(66, 70)},
    723: "no_op_user_lookup",
    726: "no_op_user_lookup",
}


class BrowserRuntime(Protocol):
    def reset(self, config_file: str) -> dict[str, Any]: ...
    def step(self, action: str) -> dict[str, Any]: ...


class EvaluatorRuntime(Protocol):
    def evaluate(self, config_file: str, answer: str) -> dict[str, Any]: ...


@dataclass
class RedditAgentMessage:
    kind: str
    payload: dict[str, Any]


def load_reddit_tasks(config_dir: Path, task_ids: Iterable[int]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for task_id in task_ids:
        path = config_dir / f"{task_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Reddit config {path} is not an object")
        if payload.get("sites") != ["reddit"]:
            raise ValueError(f"Reddit admission requires a Reddit-only task: {task_id}")
        if payload.get("eval", {}).get("eval_types") != ["string_match"]:
            raise ValueError(f"Reddit admission requires deterministic string_match: {task_id}")
        references = payload.get("eval", {}).get("reference_answers", {})
        fuzzy = references.get("fuzzy_match")
        supported = set(references) <= {"exact_match", "must_include"} or (
            set(references) == {"fuzzy_match"} and str(fuzzy).strip().upper() == "N/A"
        )
        if not references or not supported:
            raise ValueError(f"Reddit task has no deterministic evaluator adapter: {task_id}")
        task = dict(payload)
        task["task_stratum"] = REDDIT_TASK_STRATA.get(task_id, "reddit_string_query")
        tasks.append(task)
    return tasks


def routed_start_url(task: dict[str, Any], reddit_url: str) -> str:
    """Route to a forum named by the task without using evaluator-only data."""
    parameters = task.get("instantiation_dict")
    parameters = parameters if isinstance(parameters, dict) else {}
    forum = parameters.get("forum") or parameters.get("subreddit")
    if not isinstance(forum, str) or not forum.strip():
        return reddit_url.rstrip("/")
    route = f"{reddit_url.rstrip('/')}/f/{quote(forum.strip(), safe='')}"
    if "latest post" in str(task.get("intent", "")).casefold():
        route += "/new"
    return route


def task_answer_guidance(task: dict[str, Any]) -> str:
    intent = str(task.get("intent", "")).casefold()
    if "count" in intent:
        return "For this count task, no matching records means answer 0, not N/A."
    return "Use N/A only when the requested entity itself does not exist."


def is_duplicate_action_without_state_change(
    action: str,
    last_action: str,
    current_signature: tuple[str, str],
    last_result_signature: tuple[str, str] | None,
) -> bool:
    return bool(
        action == last_action
        and current_signature == last_result_signature
        and action.startswith(("click ", "type ", "hover "))
    )


def evaluate_reddit_answer(
    task: dict[str, Any],
    evaluator: EvaluatorRuntime | None,
    config_file: str,
    answer: str,
) -> dict[str, Any]:
    references = task["eval"]["reference_answers"]
    if set(references) == {"fuzzy_match"}:
        expected = str(references["fuzzy_match"]).strip().upper()
        if expected != "N/A":
            raise ValueError("unsupported Reddit fuzzy reference")
        observed = re.sub(r"[^a-z]", "", answer.casefold())
        return {
            "score": float(observed in {"na", "notapplicable"}),
            "evaluator_mode": "deterministic_na_match",
        }
    if evaluator is None:
        raise ValueError("official evaluator is required")
    return evaluator.evaluate(config_file, answer)


class RedditPlannerAgent(RoutedAgent):
    def __init__(self, client: Any) -> None:
        super().__init__("WebArena Reddit Planner")
        self.client = client

    @message_handler
    async def handle(self, message: RedditAgentMessage, ctx: Any) -> RedditAgentMessage:
        task = message.payload["task"]
        prompt = (
            "You are the Planner in a real read-only WebArena Reddit workflow. "
            "Write a short browser plan grounded only in the task. Do not invent an answer. "
            "The Browser Navigator will inspect the real Reddit clone.\n"
            f"TASK: {task['intent']}"
        )
        return RedditAgentMessage("plan", {"plan": self.client.complete(prompt).strip()})


class RedditNavigatorAgent(RoutedAgent):
    def __init__(self, client: Any, max_observation_chars: int) -> None:
        super().__init__("WebArena Reddit Browser Navigator")
        self.client = client
        self.max_observation_chars = max_observation_chars

    @message_handler
    async def handle(self, message: RedditAgentMessage, ctx: Any) -> RedditAgentMessage:
        payload = message.payload
        observation = str(payload["observation"])[-self.max_observation_chars :]
        prompt = (
            "You are the Browser Navigator in a real read-only WebArena Reddit task. "
            "Follow the plan and choose exactly one action that is valid in the current accessibility tree. "
            "Put exactly one action in triple backticks. Allowed actions: click [id], type [id] [content] [0|1], "
            "hover [id], press [key], scroll [down|up], goto [http://...], go_back, go_forward, new_tab, "
            "close_tab, page_focus [index], stop [answer]. Use stop only when the requested answer is supported "
            "by visible evidence. This admission set is read-only: never vote, subscribe, post, edit, delete, or "
            "change account state. If the requested user or submission does not exist, return stop [N/A].\n"
            f"ANSWER RULE: {task_answer_guidance(payload['task'])}\n"
            f"TASK: {payload['task']['intent']}\n"
            f"PLAN: {payload['plan']}\n"
            f"URL: {payload['url']}\n"
            f"PREVIOUS ACTIONS: {json.dumps(payload['history'][-8:], ensure_ascii=False)}\n"
            f"OBSERVATION:\n{observation}"
        )
        return RedditAgentMessage(
            "navigator_completion", {"raw": self.client.complete(prompt).strip()}
        )


class RedditCoordinatorAgent(RoutedAgent):
    def __init__(self, client: Any, max_observation_chars: int) -> None:
        super().__init__("WebArena Reddit Coordinator")
        self.client = client
        self.max_observation_chars = max_observation_chars

    @message_handler
    async def handle(self, message: RedditAgentMessage, ctx: Any) -> RedditAgentMessage:
        payload = message.payload
        observation = str(payload["observation"])[-self.max_observation_chars :]
        prompt = (
            "You are the Coordinator in a real WebArena Reddit workflow. Return strict JSON only with exactly "
            "the keys decision, answer, and reason. decision must be accept or reject. Accept only an answer "
            "supported by the final visible page evidence; preserve exact numbers, names, and URLs.\n"
            f"ANSWER RULE: {task_answer_guidance(payload['task'])}\n"
            f"TASK: {payload['task']['intent']}\n"
            f"CANDIDATE: {payload['candidate']}\n"
            f"FINAL URL: {payload['url']}\n"
            f"FINAL OBSERVATION:\n{observation}"
        )
        return RedditAgentMessage(
            "coordinator_completion",
            {"raw": self.client.complete(prompt, json_mode=True).strip()},
        )


async def _ask(
    runtime: SingleThreadedAgentRuntime,
    recipient: AgentId,
    message: RedditAgentMessage,
) -> RedditAgentMessage:
    response = await runtime.send_message(message, recipient)
    if not isinstance(response, RedditAgentMessage):
        raise TypeError(f"unexpected AutoGen response: {response!r}")
    return response


def _coordinator_result(raw: str, candidate: str) -> dict[str, str]:
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"decision": "reject", "answer": "N/A", "reason": "invalid_coordinator_json"}
    decision = "accept" if value.get("decision") == "accept" else "reject"
    return {
        "decision": decision,
        "answer": str(value.get("answer") or (candidate if decision == "accept" else "N/A")),
        "reason": str(value.get("reason", "")),
    }


def _event(
    run_id: str,
    trace_id: str,
    step_index: int,
    source: str,
    target: str,
    original: Any,
    delivered: Any,
    effect: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step_index": step_index,
        "step_id": f"step-{step_index:03d}",
        "message_id": f"{run_id}:step-{step_index:03d}",
        "source_agent": source,
        "target_agent": target,
        "original_message": original,
        "delivered_message": delivered,
        "fault_type": "clean",
        "fault_applied": False,
        "observed_runtime_effect": effect,
        "observed_A_symptom": "none",
        "observed_M_consequence": ["none"],
    }


async def run_clean_reddit_task(
    client: Any,
    browser: BrowserRuntime,
    evaluator: EvaluatorRuntime,
    task: dict[str, Any],
    *,
    browser_config_file: str,
    evaluator_config_file: str,
    run_index: int,
    max_steps: int = 20,
    max_observation_chars: int = 12000,
) -> dict[str, Any]:
    run_id = f"reddit-official-{task['task_id']}-clean-r{run_index}-{uuid.uuid4().hex[:8]}"
    trace_id = f"trace-{uuid.uuid4()}"
    started = time.perf_counter()
    before = (client.call_count, client.prompt_tokens, client.completion_tokens)
    runtime = SingleThreadedAgentRuntime()
    planner_id = AgentId("reddit_planner", "default")
    navigator_id = AgentId("reddit_navigator", "default")
    coordinator_id = AgentId("reddit_coordinator", "default")
    await runtime.register_agent_instance(RedditPlannerAgent(client), planner_id)
    await runtime.register_agent_instance(
        RedditNavigatorAgent(client, max_observation_chars), navigator_id
    )
    await runtime.register_agent_instance(
        RedditCoordinatorAgent(client, max_observation_chars), coordinator_id
    )
    runtime.start()
    events: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    history: list[str] = []
    candidate = "N/A"
    termination_reason = "max_steps"
    step_index = 0
    parse_failures = 0
    duplicate_rejections = 0
    last_action = ""
    last_result_signature: tuple[str, str] | None = None
    try:
        state = browser.reset(browser_config_file)
        step_index += 1
        events.append(_event(run_id, trace_id, step_index, "WebArena Reddit", "Browser Navigator", {"config_file": browser_config_file}, state, "environment_reset"))
        plan_message = await _ask(runtime, planner_id, RedditAgentMessage("task", {"task": task}))
        plan = str(plan_message.payload["plan"])
        step_index += 1
        events.append(_event(run_id, trace_id, step_index, "Planner", "Browser Navigator", {"task": task["intent"]}, {"plan": plan}, "plan_handoff"))

        for _ in range(max_steps):
            completion = await _ask(
                runtime,
                navigator_id,
                RedditAgentMessage(
                    "navigate",
                    {
                        "task": task,
                        "plan": plan,
                        "observation": state.get("observation", ""),
                        "url": state.get("url", ""),
                        "history": history,
                    },
                ),
            )
            raw = str(completion.payload["raw"])
            try:
                action = extract_browser_action(raw)
            except ValueError:
                parse_failures += 1
                history.append("PARSE_FAILURE")
                if parse_failures >= 3:
                    termination_reason = "action_parse_failure"
                    break
                continue
            parse_failures = 0
            current_signature = (
                str(state.get("url", "")),
                str(state.get("observation", "")),
            )
            if is_duplicate_action_without_state_change(
                action, last_action, current_signature, last_result_signature
            ):
                duplicate_rejections += 1
                history.append(
                    f"REJECTED_DUPLICATE_ACTION {action}: the page did not change; choose another action or stop with the supported answer."
                )
                step_index += 1
                events.append(_event(run_id, trace_id, step_index, "Browser Navigator", "Browser Action Validator", raw, None, "duplicate_action_rejected"))
                if duplicate_rejections >= 3:
                    termination_reason = "duplicate_action_loop"
                    break
                continue
            duplicate_rejections = 0
            history.append(action)
            step_index += 1
            events.append(_event(run_id, trace_id, step_index, "Browser Navigator", "WebArena Reddit", raw, action, "browser_action_requested"))
            state = browser.step(action)
            last_action = action
            last_result_signature = (
                str(state.get("url", "")),
                str(state.get("observation", "")),
            )
            step_index += 1
            events.append(_event(run_id, trace_id, step_index, "WebArena Reddit", "Browser Navigator", {"action": action}, state, "browser_observation_delivered"))
            if action.startswith("stop "):
                candidate = str(state.get("answer") or "N/A")
                termination_reason = "navigator_stop"
                break

        coordinator_message = await _ask(
            runtime,
            coordinator_id,
            RedditAgentMessage(
                "verify",
                {
                    "task": task,
                    "candidate": candidate,
                    "observation": state.get("observation", ""),
                    "url": state.get("url", ""),
                },
            ),
        )
        coordinator = _coordinator_result(
            str(coordinator_message.payload["raw"]), candidate
        )
        step_index += 1
        events.append(_event(run_id, trace_id, step_index, "Coordinator", "Official WebArena Evaluator", {"candidate": candidate}, coordinator, "final_answer_handoff"))
        evaluation = evaluate_reddit_answer(
            task, evaluator, evaluator_config_file, coordinator["answer"]
        )
        score = float(evaluation["score"])
        success = score == 1.0
        prompt_tokens = client.prompt_tokens - before[1]
        completion_tokens = client.completion_tokens - before[2]
        return normalize_run_record(
            {
                "run_id": run_id,
                "trace_id": trace_id,
                "scenario": "webarena_reddit_official_clean",
                "dataset": "WebArena",
                "benchmark": "WebArena Reddit",
                "framework": "AutoGen",
                "topology": "sequential",
                "task_id": str(task["task_id"]),
                "task_stratum": task["task_stratum"],
                "intent": task["intent"],
                "condition": "clean",
                "model": client.model_info.model,
                "provider": client.model_info.provider,
                "seed_or_run_index": run_index,
                "fault_id": "none",
                "fault_type": "clean",
                "fault_severity": "none",
                "fault_parameters": {},
                "fault_applied": False,
                "source_agent": "Coordinator",
                "target_agent": "Official WebArena Evaluator",
                "original_message": {"candidate": candidate},
                "delivered_message": coordinator,
                "first_divergence": "none",
                "observed_runtime_effect": termination_reason,
                "observed_A_symptom": ["none"],
                "observed_M_consequence": ["none"],
                "system_consequences": ["none"],
                "semantic_consequences": ["none"],
                "recovery_detected": False,
                "recovery_type": "none",
                "recovery_evidence": [],
                "expected_answer": task["eval"]["reference_answers"],
                "final_answer": coordinator,
                "task_score": score,
                "final_task_success": success,
                "propagation_class": "clean" if success else "clean_task_failure",
                "termination_reason": termination_reason,
                "evaluator_mode": evaluation.get("evaluator_mode", "unknown"),
                "browser_final_url": state.get("url", ""),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "api_call_count": client.call_count - before[0],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "error": None,
                "events": events,
            }
        )
    finally:
        await runtime.stop()


def write_clean_outputs(
    rows: list[dict[str, Any]], output: Path, experiment_config: dict[str, Any]
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one Reddit clean run is required")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "clean_runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({k: v for k, v in row.items() if k != "events"}, ensure_ascii=False) + "\n")
    with (output / "clean_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            for event in row.get("events", []):
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    flat = [
        {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items() if k != "events"}
        for row in rows
    ]
    with (output / "clean_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in flat for k in row}))
        writer.writeheader()
        writer.writerows(flat)
    successes = sum(bool(row["final_task_success"]) for row in rows)
    summary = {
        "runs": len(rows),
        "final_success": successes,
        "clean_task_failure": len(rows) - successes,
        "success_rate": round(successes / len(rows), 6),
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in rows),
        "termination_reasons": dict(Counter(str(row.get("termination_reason", "unknown")) for row in rows)),
    }
    (output / "clean_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "clean_summary.md").write_text(
        "# WebArena Reddit 官方任务 Clean Admission\n\n"
        f"- Runs: {summary['runs']}\n- Final success: {successes}\n"
        f"- Success rate: {summary['success_rate']:.2%}\n- Total tokens: {summary['total_tokens']}\n",
        encoding="utf-8",
    )
    (output / "experiment_config.json").write_text(json.dumps(experiment_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
