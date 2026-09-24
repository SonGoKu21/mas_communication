"""Real WebArena Shopping Admin adapter for clean MAS baselines."""

from __future__ import annotations

import json
import re
import csv
import os
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from autogen_core import AgentId, RoutedAgent, SingleThreadedAgentRuntime, message_handler

from mas_faults.benchmark_trace_contract import normalize_run_record


_ACTION_PATTERNS = (
    r"click \[\d+\]",
    r"type \[\d+\] \[[^\r\n]*\] \[[01]\]",
    r"hover \[\d+\]",
    r"press \[[^\r\n]+\]",
    r"scroll \[(?:down|up)\]",
    r"select \[\d+\] (?:\[[^\r\n]+\]|[^\[\]\r\n]+)",
    r"goto \[(?:https?://)[^\r\n]+\]",
    r"(?:new_tab|close_tab|go_back|go_forward)",
    r"(?:tab_focus|page_focus) \[\d+\]",
    r"stop \[[^\r\n]*\]",
)
_ACTION_RE = re.compile(rf"^(?:{'|'.join(_ACTION_PATTERNS)})$")


def normalize_browser_action(action: str) -> str:
    """Canonicalize common model variants without changing action semantics."""
    select = re.fullmatch(
        r"select \[(\d+)\] (?:\[([^\r\n]+)\]|([^\[\]\r\n]+))",
        action,
    )
    if select:
        option = (select.group(2) or select.group(3)).strip()
        return f"select [{select.group(1)}] [{option}]"

    canonical_type = re.fullmatch(
        r"type \[(\d+)\] \[([^\r\n]*)\] \[([01])\]",
        action,
    )
    if canonical_type:
        return action
    relaxed_type = re.fullmatch(
        r"type \[(\d+)\] (.+?) (?:\[([01])\]|([01]))",
        action,
    )
    if relaxed_type:
        content = relaxed_type.group(2).strip()
        enter = relaxed_type.group(3) or relaxed_type.group(4)
        return f"type [{relaxed_type.group(1)}] [{content}] [{enter}]"
    return action


def load_admin_tasks(path: Path) -> list[dict[str, Any]]:
    """Load the frozen, read-only Shopping Admin task manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"task manifest {path} has no tasks")

    result: list[dict[str, Any]] = []
    required = {"task_id", "task_stratum", "sites", "intent", "eval"}
    for task in tasks:
        if not isinstance(task, dict) or not required.issubset(task):
            raise ValueError(f"task manifest {path} contains an incomplete task")
        eval_types = task.get("eval", {}).get("eval_types")
        if task.get("sites") != ["shopping_admin"] or eval_types != ["string_match"]:
            raise ValueError("Shopping Admin clean baseline requires read-only string_match tasks")
        result.append(dict(task))
    return result


def select_stratified_tasks(
    tasks: list[dict[str, Any]], *, per_stratum: int
) -> list[dict[str, Any]]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be at least 1")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[str(task["task_stratum"])].append(task)
    return [
        task
        for stratum in sorted(grouped)
        for task in grouped[stratum][:per_stratum]
    ]


def extract_browser_action(raw: str) -> str:
    """Extract exactly one fenced WebArena action from a model completion."""
    fenced = [
        value.strip()
        for value in re.findall(r"```(?:[A-Za-z0-9_+-]+\n)?([\s\S]*?)```", raw)
    ]
    if len(fenced) != 1:
        raise ValueError("model output did not contain exactly one supported browser action")
    action = normalize_browser_action(fenced[0])
    if not _ACTION_RE.fullmatch(action):
        raise ValueError("model output did not contain exactly one supported browser action")
    return action


def semantic_browser_action_key(action: str, observation: str) -> str:
    target = re.match(r"^(click|hover|type) \[(\d+)\](.*)$", action)
    if not target:
        return action
    verb, element_id, suffix = target.groups()
    node = re.search(
        rf"^\s*\[{re.escape(element_id)}\]\s+(\S+)\s+'(.*)'(?:\s|$)",
        observation,
        flags=re.MULTILINE,
    )
    if not node:
        return action
    role, name = node.groups()
    return f"{verb} {role} '{name}'{suffix}"


class BrowserRuntime(Protocol):
    def reset(self, config_file: str) -> dict[str, Any]: ...
    def step(self, action: str) -> dict[str, Any]: ...
    def tool(self, name: str, arguments: dict[str, str]) -> dict[str, Any]: ...
    def evaluate(self, answer: str | None = None) -> dict[str, Any]: ...


class BrowserWorkerClient:
    """Persistent JSONL client for the isolated official WebArena process."""

    def __init__(
        self,
        *,
        process: Any | None = None,
        python_executable: str = "/data2/system5/mas/venvs/webarena/bin/python",
        worker_script: str = "/home/systemai_5/code/mas/scripts/smoke/webarena_admin_browser_worker.py",
        webarena_root: str = "/data2/system5/mas/third_party/webarena",
        env: dict[str, str] | None = None,
        browser_only: bool = False,
    ) -> None:
        if process is None:
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            argv = [python_executable, worker_script, "--webarena-root", webarena_root]
            if browser_only:
                argv.append("--browser-only")
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=process_env,
            )
        self.process = process

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"WebArena browser worker exited: {stderr[-1000:]}")
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("WebArena browser worker has no JSONL streams")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"WebArena browser worker returned no response: {stderr[-1000:]}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(
                f"WebArena browser worker {response.get('error_type', 'Error')}: "
                f"{response.get('error', 'unknown error')}"
            )
        return {key: value for key, value in response.items() if key != "ok"}

    def reset(self, config_file: str) -> dict[str, Any]:
        return self._request({"command": "reset", "config_file": config_file})

    def step(self, action: str) -> dict[str, Any]:
        return self._request({"command": "step", "action": action})

    def tool(self, name: str, arguments: dict[str, str]) -> dict[str, Any]:
        return self._request(
            {"command": "tool", "tool": name, "arguments": arguments}
        )

    def evaluate(self, answer: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"command": "evaluate"}
        if answer is not None:
            payload["answer"] = answer
        return self._request(payload)

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._request({"command": "close"})
            finally:
                if self.process.poll() is None:
                    self.process.terminate()


@dataclass
class AdminAgentMessage:
    kind: str
    payload: dict[str, Any]


class PlannerAgent(RoutedAgent):
    def __init__(self, client: Any) -> None:
        super().__init__("Shopping Admin Planner")
        self.client = client

    @message_handler
    async def handle(self, message: AdminAgentMessage, ctx: Any) -> AdminAgentMessage:
        task = message.payload["task"]
        prompt = (
            "You are the Planner in a real WebArena Shopping Admin workflow. "
            "Write a short, generic browser plan for the Browser Navigator. Do not invent an answer "
            "and do not assume hidden page state. The navigator will operate the real Magento Admin UI.\n"
            f"Task: {task['intent']}"
        )
        return AdminAgentMessage("plan", {"plan": self.client.complete(prompt).strip()})


def build_navigator_prompt(
    payload: dict[str, Any], *, max_observation_chars: int
) -> str:
    observation = str(payload["observation"])
    if len(observation) > max_observation_chars:
        observation = observation[:max_observation_chars]
    return (
        "You are the Browser Navigator in a real Magento Admin website. Reason briefly about the first "
        "unfinished PLAN step, then choose exactly one action that is valid in the current accessibility "
        "tree. Put exactly one action in triple backticks.\n"
        "Allowed actions: click [id], type [id] [content] [0|1], hover [id], press [key], "
        "scroll [down|up], select [id] [option text], goto [http://...], go_back, go_forward, new_tab, close_tab, "
        "page_focus [index], stop [answer]. Use stop only when the exact task answer is known.\n"
        "Interaction rules: Follow the PLAN in order and do not skip required inputs. If the PLAN "
        "provides values for visible required textboxes, enter them with type before clicking a submit-like button "
        "such as Show Report, Apply, Search, or Submit. A required-field error means fill that field next. "
        "Use select [id] [option text] when the observation lists options for a combobox, and never click "
        "that combobox. Never repeat the same action when the URL and visible page state did not change. "
        "Only when no option list is available and a combobox is already expanded, use press [ArrowDown] "
        "followed on the next turn by press [Enter]. Fill or replace a textbox with type [id] [content] "
        "[0|1], not repeated clicks. Magento date textboxes accept MM/DD/YYYY.\n"
        "Safety and visibility: All tasks in this run are read-only. Never activate Add, Edit, Delete, or Save. "
        "If a read-only list page shows an Add button but not the requested records, scroll down to reveal the "
        "data grid. Use Filters for requested attributes instead of repeatedly toggling a sort column.\n"
        f"TASK: {payload['task']['intent']}\n"
        f"PLAN: {payload['plan']}\n"
        f"URL: {payload['url']}\n"
        f"PREVIOUS ACTIONS: {json.dumps(payload['history'][-6:], ensure_ascii=False)}\n"
        f"OBSERVATION:\n{observation}\n"
        "ACTION DECISION CHECKLIST: Choose the first unfinished PLAN step. Prefer select for listed "
        "combobox options. Fill task-relevant required textboxes before any submit-like button. If the "
        "previous action made no progress, choose a different action."
    )


class NavigatorAgent(RoutedAgent):
    def __init__(self, client: Any, max_observation_chars: int) -> None:
        super().__init__("Shopping Admin Browser Navigator")
        self.client = client
        self.max_observation_chars = max_observation_chars

    @message_handler
    async def handle(self, message: AdminAgentMessage, ctx: Any) -> AdminAgentMessage:
        prompt = build_navigator_prompt(
            message.payload, max_observation_chars=self.max_observation_chars
        )
        return AdminAgentMessage("navigator_completion", {"raw": self.client.complete(prompt).strip()})


class CoordinatorAgent(RoutedAgent):
    def __init__(self, client: Any, max_observation_chars: int) -> None:
        super().__init__("Shopping Admin Coordinator")
        self.client = client
        self.max_observation_chars = max_observation_chars

    @message_handler
    async def handle(self, message: AdminAgentMessage, ctx: Any) -> AdminAgentMessage:
        payload = message.payload
        observation = str(payload["observation"])[-self.max_observation_chars :]
        prompt = (
            "You are the Coordinator in a real WebArena Shopping Admin workflow. Return strict JSON only "
            "with keys decision (accept or reject), answer, and reason. Check the Browser Navigator's "
            "candidate against the final visible page evidence. Preserve exact product names, numbers, "
            "punctuation, and requested output format. If evidence cannot support an answer, reject and use "
            "answer N/A.\n"
            f"TASK: {payload['task']['intent']}\n"
            f"CANDIDATE: {payload['candidate']}\n"
            f"FINAL URL: {payload['url']}\n"
            f"FINAL OBSERVATION:\n{observation}"
        )
        raw = self.client.complete(prompt, json_mode=True).strip()
        return AdminAgentMessage("coordinator_completion", {"raw": raw})


async def _ask(
    runtime: SingleThreadedAgentRuntime,
    recipient: AgentId,
    message: AdminAgentMessage,
) -> AdminAgentMessage:
    response = await runtime.send_message(message, recipient)
    if not isinstance(response, AdminAgentMessage):
        raise TypeError(f"unexpected AutoGen response: {response!r}")
    return response


def _coordinator_result(raw: str, candidate: str) -> dict[str, str]:
    match = re.search(r"\{[\s\S]*\}", raw)
    try:
        value = json.loads(match.group(0) if match else raw)
    except (json.JSONDecodeError, AttributeError):
        return {"decision": "accept", "answer": candidate, "reason": "coordinator_non_json_fallback"}
    decision = "accept" if str(value.get("decision", "reject")).lower() == "accept" else "reject"
    answer = str(value.get("answer") or (candidate if decision == "accept" else "N/A"))
    return {"decision": decision, "answer": answer, "reason": str(value.get("reason", ""))}


def _event(
    *,
    run_id: str,
    trace_id: str,
    step_index: int,
    source: str,
    target: str,
    original: Any,
    delivered: Any,
    effect: str = "clean_delivery",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step_index": step_index,
        "step_id": f"step-{step_index:03d}",
        "source_agent": source,
        "target_agent": target,
        "message_id": f"{run_id}:step-{step_index:03d}",
        "original_message": original,
        "delivered_message": delivered,
        "fault_type": "clean",
        "fault_applied": False,
        "observed_runtime_effect": effect,
        "observed_A_symptom": "none",
        "observed_M_consequence": ["none"],
    }


async def run_clean_task(
    client: Any,
    browser: BrowserRuntime,
    task: dict[str, Any],
    *,
    config_file: str,
    run_index: int,
    max_steps: int = 20,
    max_observation_chars: int = 12000,
) -> dict[str, Any]:
    """Run one clean Planner -> Navigator -> Coordinator WebArena task."""
    run_id = f"admin-{task['task_id']}-clean-r{run_index}-{uuid.uuid4().hex[:8]}"
    trace_id = f"trace-{uuid.uuid4()}"
    started = time.perf_counter()
    before = (client.call_count, client.prompt_tokens, client.completion_tokens)
    events: list[dict[str, Any]] = []
    runtime = SingleThreadedAgentRuntime()
    planner_id = AgentId("admin_planner", "default")
    navigator_id = AgentId("admin_navigator", "default")
    coordinator_id = AgentId("admin_coordinator", "default")
    await runtime.register_agent_instance(PlannerAgent(client), planner_id)
    await runtime.register_agent_instance(NavigatorAgent(client, max_observation_chars), navigator_id)
    await runtime.register_agent_instance(CoordinatorAgent(client, max_observation_chars), coordinator_id)
    runtime.start()

    termination_reason = "max_steps"
    candidate = "N/A"
    history: list[str] = []
    parse_failures = 0
    duplicate_rejections = 0
    last_semantic_action = ""
    last_result_signature: tuple[str, str] | None = None
    browser_state: dict[str, Any] = {}
    step_index = 0
    try:
        browser_state = browser.reset(config_file)
        step_index += 1
        events.append(
            _event(
                run_id=run_id,
                trace_id=trace_id,
                step_index=step_index,
                source="WebArena Shopping Admin",
                target="Browser Navigator",
                original={"config_file": config_file},
                delivered=browser_state,
                effect="environment_reset",
            )
        )
        plan_message = await _ask(runtime, planner_id, AdminAgentMessage("task", {"task": task}))
        plan = str(plan_message.payload["plan"])
        step_index += 1
        events.append(
            _event(
                run_id=run_id,
                trace_id=trace_id,
                step_index=step_index,
                source="Planner",
                target="Browser Navigator",
                original={"task": task["intent"]},
                delivered={"plan": plan},
                effect="plan_handoff",
            )
        )

        for _ in range(max_steps):
            completion = await _ask(
                runtime,
                navigator_id,
                AdminAgentMessage(
                    "navigate",
                    {
                        "task": task,
                        "plan": plan,
                        "observation": browser_state.get("observation", ""),
                        "url": browser_state.get("url", ""),
                        "history": history,
                    },
                ),
            )
            raw = str(completion.payload["raw"])
            try:
                action = extract_browser_action(raw)
            except ValueError:
                parse_failures += 1
                step_index += 1
                events.append(
                    _event(
                        run_id=run_id,
                        trace_id=trace_id,
                        step_index=step_index,
                        source="Browser Navigator",
                        target="WebArena Shopping Admin",
                        original=raw,
                        delivered=None,
                        effect="model_action_parse_failure",
                    )
                )
                history.append("PARSE_FAILURE")
                if parse_failures >= 3:
                    termination_reason = "action_parse_failure"
                    break
                continue

            parse_failures = 0
            current_signature = (
                str(browser_state.get("url", "")),
                str(browser_state.get("observation", "")),
            )
            semantic_action = semantic_browser_action_key(
                action, str(browser_state.get("observation", ""))
            )
            duplicate_without_change = bool(
                semantic_action == last_semantic_action
                and current_signature == last_result_signature
                and action.startswith(("click ", "type ", "hover "))
            )
            if duplicate_without_change:
                duplicate_rejections += 1
                feedback = (
                    f"REJECTED_DUPLICATE_ACTION {semantic_action}: this semantic target was already "
                    "used and the page did not change. Do not activate it again; scroll or choose a "
                    "different valid action."
                )
                history.append(feedback)
                step_index += 1
                events.append(
                    _event(
                        run_id=run_id,
                        trace_id=trace_id,
                        step_index=step_index,
                        source="Browser Navigator",
                        target="Browser Action Validator",
                        original=raw,
                        delivered=None,
                        effect="duplicate_action_rejected",
                    )
                )
                if duplicate_rejections >= 3:
                    termination_reason = "duplicate_action_loop"
                    break
                continue

            duplicate_rejections = 0
            history.append(
                action if semantic_action == action else f"{action} -> {semantic_action}"
            )
            step_index += 1
            events.append(
                _event(
                    run_id=run_id,
                    trace_id=trace_id,
                    step_index=step_index,
                    source="Browser Navigator",
                    target="WebArena Shopping Admin",
                    original=raw,
                    delivered=action,
                    effect="browser_action_requested",
                )
            )
            browser_state = browser.step(action)
            last_semantic_action = semantic_action
            last_result_signature = (
                str(browser_state.get("url", "")),
                str(browser_state.get("observation", "")),
            )
            step_index += 1
            events.append(
                _event(
                    run_id=run_id,
                    trace_id=trace_id,
                    step_index=step_index,
                    source="WebArena Shopping Admin",
                    target="Browser Navigator",
                    original={"action": action},
                    delivered=browser_state,
                    effect="browser_observation_delivered",
                )
            )
            if action.startswith("stop "):
                candidate = str(browser_state.get("answer") or "N/A")
                termination_reason = "navigator_stop"
                break

        coordinator_message = await _ask(
            runtime,
            coordinator_id,
            AdminAgentMessage(
                "verify",
                {
                    "task": task,
                    "candidate": candidate,
                    "observation": browser_state.get("observation", ""),
                    "url": browser_state.get("url", ""),
                },
            ),
        )
        coordinator = _coordinator_result(str(coordinator_message.payload["raw"]), candidate)
        step_index += 1
        events.append(
            _event(
                run_id=run_id,
                trace_id=trace_id,
                step_index=step_index,
                source="Coordinator",
                target="Official WebArena Evaluator",
                original={"candidate": candidate, "final_url": browser_state.get("url", "")},
                delivered=coordinator,
                effect="final_answer_handoff",
            )
        )
        evaluation = browser.evaluate(coordinator["answer"])
        score = float(evaluation["score"])
        success = score == 1.0
        expected = task.get("eval", {}).get("reference_answers", {})
        record = {
            "run_id": run_id,
            "trace_id": trace_id,
            "scenario": "webarena_shopping_admin_clean",
            "dataset": "WebArena",
            "benchmark": "WebArena Shopping Admin",
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
            "expected_answer": expected,
            "final_answer": coordinator,
            "task_score": score,
            "final_task_success": success,
            "propagation_class": "clean" if success else "clean_task_failure",
            "termination_reason": termination_reason,
            "browser_final_url": browser_state.get("url", ""),
            "browser_final_title": browser_state.get("title", ""),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "api_call_count": client.call_count - before[0],
            "prompt_tokens": client.prompt_tokens - before[1],
            "completion_tokens": client.completion_tokens - before[2],
            "total_tokens": (client.prompt_tokens - before[1]) + (client.completion_tokens - before[2]),
            "error": None,
            "events": events,
        }
        return normalize_run_record(record)
    finally:
        await runtime.stop()


def build_clean_error_record(
    client: Any,
    task: dict[str, Any],
    *,
    run_index: int,
    error: Exception,
) -> dict[str, Any]:
    """Represent infrastructure/model exceptions without inventing fault propagation."""
    run_id = f"admin-{task['task_id']}-clean-error-r{run_index}-{uuid.uuid4().hex[:8]}"
    return normalize_run_record(
        {
            "run_id": run_id,
            "trace_id": f"trace-{uuid.uuid4()}",
            "scenario": "webarena_shopping_admin_clean",
            "dataset": "WebArena",
            "benchmark": "WebArena Shopping Admin",
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
            "fault_applied": False,
            "observed_A_symptom": ["none"],
            "observed_M_consequence": ["none"],
            "system_consequences": ["none"],
            "semantic_consequences": ["none"],
            "recovery_detected": False,
            "recovery_type": "none",
            "recovery_evidence": [],
            "expected_answer": task.get("eval", {}).get("reference_answers", {}),
            "final_answer": {},
            "task_score": 0.0,
            "final_task_success": False,
            "propagation_class": "clean_task_failure",
            "termination_reason": "runtime_error",
            "latency_ms": 0.0,
            "api_call_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": f"{type(error).__name__}: {error}",
            "events": [],
        }
    )


def write_clean_outputs(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """Write canonical clean baseline artifacts without overwriting a prior run."""
    if not rows:
        raise ValueError("at least one clean run is required")
    output.mkdir(parents=True, exist_ok=False)

    with (output / "clean_runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({key: value for key, value in row.items() if key != "events"}, ensure_ascii=False) + "\n")
    with (output / "clean_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            for event in row.get("events", []):
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
                if key != "events"
            }
        )
    with (output / "clean_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in flat_rows for key in row}))
        writer.writeheader()
        writer.writerows(flat_rows)

    by_stratum: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_stratum"])].append(row)
    for stratum, values in sorted(grouped.items()):
        successes = sum(bool(value["final_task_success"]) for value in values)
        by_stratum[stratum] = {
            "runs": len(values),
            "final_success": successes,
            "success_rate": round(successes / len(values), 6),
        }
    success_count = sum(bool(row["final_task_success"]) for row in rows)
    summary = {
        "runs": len(rows),
        "final_success": success_count,
        "clean_task_failure": len(rows) - success_count,
        "success_rate": round(success_count / len(rows), 6),
        "api_call_count": sum(int(row.get("api_call_count", 0)) for row in rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in rows),
        "termination_reasons": dict(sorted(Counter(str(row.get("termination_reason", "unknown")) for row in rows).items())),
        "by_stratum": by_stratum,
    }
    (output / "clean_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# WebArena Shopping Admin Clean Baseline",
        "",
        f"- Runs: {summary['runs']}",
        f"- Final success: {summary['final_success']}",
        f"- Clean task failure: {summary['clean_task_failure']}",
        f"- Success rate: {summary['success_rate']:.2%}",
        f"- Total tokens: {summary['total_tokens']}",
        "",
        "## 分层结果",
        "",
        "| 任务类型 | Runs | Success | Success rate |",
        "|---|---:|---:|---:|",
    ]
    for stratum, values in by_stratum.items():
        markdown.append(
            f"| {stratum} | {values['runs']} | {values['final_success']} | {values['success_rate']:.2%} |"
        )
    markdown.extend(["", "本文件只报告 clean baseline，不包含通信故障传播结论。", ""])
    (output / "clean_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    (output / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
