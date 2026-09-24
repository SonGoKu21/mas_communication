"""AutoGen-oriented execution primitives for real TheAgentCompany tasks.

The primitives in this module deliberately separate tool observations from
Agent-to-Agent communication.  Communication faults are applied by the runner
only to the latter.
"""

from __future__ import annotations

import base64
import html
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_core import FunctionCall
from autogen_core.models import CreateResult
from autogen_ext.models.openai import OpenAIChatCompletionClient

from mas_faults.llm_client import get_api_key, get_base_url, get_model


Executor = Callable[[list[str], int], subprocess.CompletedProcess[str]]
ServiceProbe = Callable[[str], bool]


@dataclass(frozen=True)
class ToolObservation:
    tool_name: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float

    def as_message(self) -> str:
        def clipped(value: str, limit: int = 4000) -> str:
            value = html.unescape(value)
            if len(value) <= limit:
                return value
            half = limit // 2
            return f"{value[:half]}\n...[middle omitted by tool transport]...\n{value[-half:]}"

        return (
            f"tool={self.tool_name}\ncommand={self.command}\nexit_code={self.exit_code}\n"
            f"stdout:\n{clipped(self.stdout)}\nstderr:\n{clipped(self.stderr)}"
        )


@dataclass(frozen=True)
class ServiceReadiness:
    ready: bool
    unavailable_services: tuple[str, ...]
    error_kind: str | None


@dataclass(frozen=True)
class OperatorRun:
    final_message: str
    tool_observations: tuple[ToolObservation, ...]
    events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CoordinatorRun:
    handoff: str
    events: tuple[dict[str, Any], ...]


def autogen_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def autogen_model_name(model: str) -> str:
    """Use DeepSeek's canonical non-thinking model name for multi-turn tool calls."""
    return "deepseek-v4-flash" if model == "deepseek-chat" else model


def parse_dsml_function_calls(content: str) -> list[FunctionCall]:
    """Convert DeepSeek's textual DSML fallback into AutoGen function calls."""
    calls: list[FunctionCall] = []
    invocation_pattern = re.compile(
        r'<｜｜DSML｜｜invoke name="(?P<name>[^"]+)">(?P<body>.*?)</｜｜DSML｜｜invoke>',
        flags=re.DOTALL,
    )
    parameter_pattern = re.compile(
        r'<｜｜DSML｜｜parameter name="(?P<name>[^"]+)"(?: string="true")?>(?P<value>.*?)</｜｜DSML｜｜parameter>',
        flags=re.DOTALL,
    )
    for index, invocation in enumerate(invocation_pattern.finditer(content)):
        arguments = {
            parameter.group("name"): html.unescape(parameter.group("value").strip())
            for parameter in parameter_pattern.finditer(invocation.group("body"))
        }
        calls.append(
            FunctionCall(
                id=f"dsml-call-{index}",
                name=invocation.group("name"),
                arguments=json.dumps(arguments, ensure_ascii=False),
            )
        )
    return calls


def has_explicit_termination(content: str) -> bool:
    """Accept only an Agent's standalone termination marker, never tool output text."""
    return bool(re.search(r"(?mi)^\s*TERMINATE[.!]?\s*$", content))


def operator_iteration_guidance(iteration: int) -> str:
    if iteration <= 4:
        return "First establish the actual task state with targeted observations."
    return (
        "Reconnaissance is now complete enough to act. When the task requests a state change and the relevant "
        "path or target has been observed, your next tool call must perform that state change. Do not repeat a "
        "read-only inspection of the same target unless the last action failed. After a mutation, verify it with "
        "a concrete command such as a diff, status, test, or targeted read."
    )


class DSMLCompatibleOpenAIChatCompletionClient(OpenAIChatCompletionClient):
    """Preserve AutoGen tool execution when DeepSeek emits its native DSML form."""

    async def create(self, *args: Any, **kwargs: Any) -> CreateResult:
        result = await super().create(*args, **kwargs)
        if not isinstance(result.content, str):
            return result
        calls = parse_dsml_function_calls(result.content)
        if not calls:
            return result
        return CreateResult(
            finish_reason="function_calls",
            content=calls,
            usage=result.usage,
            cached=result.cached,
            logprobs=result.logprobs,
            thought=result.thought,
        )


def serialize_agent_event(*, turn_index: int, source: str, message_type: str, content: str) -> dict[str, Any]:
    return {
        "turn_index": turn_index,
        "source_agent": source,
        "event_type": message_type,
        "content": content,
    }


def create_autogen_model_client() -> OpenAIChatCompletionClient:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required for the TAC AutoGen runtime")
    return DSMLCompatibleOpenAIChatCompletionClient(
        model=autogen_model_name(get_model()),
        api_key=api_key,
        base_url=autogen_endpoint(get_base_url()),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
            "family": "unknown",
        },
        temperature=0,
    )


async def run_coordinator_agent(
    *,
    model_client: OpenAIChatCompletionClient,
    task_instruction: str,
) -> CoordinatorRun:
    """Generate a constrained Coordinator-to-Operator handoff with AutoGen."""
    coordinator = AssistantAgent(
        "Coordinator",
        model_client=model_client,
        system_message=(
            "You are the Coordinator in a TheAgentCompany MAS. You have no task-container tools. Produce a concise "
            "handoff for the Operator: identify the objective, caution it to inspect the real environment and services "
            "before acting, and specify the evidence needed for completion. Do not claim any operation was executed."
        ),
    )
    result = await coordinator.run(task=f"Prepare an execution handoff for this task:\n{task_instruction.strip()}")
    events = tuple(
        serialize_agent_event(
            turn_index=index,
            source=str(getattr(message, "source", "Coordinator")),
            message_type=type(message).__name__,
            content=str(getattr(message, "content", "")),
        )
        for index, message in enumerate(result.messages, start=1)
    )
    handoff = str(getattr(result.messages[-1], "content", "")) if result.messages else ""
    return CoordinatorRun(handoff, events)


def subprocess_executor(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        return subprocess.CompletedProcess(command, 124, stdout, f"{stderr}\ncommand timed out after {timeout} seconds".strip())


class TACContainerTools:
    """Bounded shell access inside one initialized TAC task container."""

    def __init__(self, container_name: str, *, executor: Executor = subprocess_executor) -> None:
        self.container_name = container_name
        self._executor = executor

    def run_shell(self, command: str, *, timeout: int = 120) -> ToolObservation:
        started = time.perf_counter()
        completed = self._executor(["docker", "exec", self.container_name, "bash", "-lc", command], timeout)
        return ToolObservation(
            tool_name="run_shell",
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def read_file(self, path: str, *, timeout: int = 60) -> ToolObservation:
        return self.run_shell(f"cat -- {path}", timeout=timeout)

    def list_directory(self, path: str = ".", *, timeout: int = 60) -> ToolObservation:
        return self.run_shell(f"find -- {path} -maxdepth 2 -mindepth 1 -printf '%p\\n' | head -200", timeout=timeout)

    def replace_text(self, path: str, old_text: str, new_text: str, *, timeout: int = 120) -> ToolObservation:
        """Replace observed text in a real task file, failing when the expected text is absent."""
        payload = base64.b64encode(
            json.dumps({"path": path, "old_text": old_text, "new_text": new_text}).encode("utf-8")
        ).decode("ascii")
        program = (
            "import base64,json,pathlib,sys; d=json.loads(base64.b64decode(sys.argv[1])); "
            "p=pathlib.Path(d['path']); text=p.read_text(); "
            "assert d['old_text'] in text, 'expected text not found'; "
            "p.write_text(text.replace(d['old_text'], d['new_text']))"
        )
        return self.run_shell(f"python3 -c {shlex.quote(program)} {shlex.quote(payload)}", timeout=timeout)

    def remove_path(self, path: str, *, timeout: int = 120) -> ToolObservation:
        """Remove one task-local file after the Agent has inspected it."""
        return self.run_shell(f"rm -f -- {shlex.quote(path)}", timeout=timeout)


def wait_for_required_services(
    services: Sequence[str],
    *,
    probe: ServiceProbe,
    attempts: int = 24,
    sleep_seconds: float = 5,
    consecutive_successes: int = 3,
) -> ServiceReadiness:
    unavailable = tuple(services)
    stable_successes = 0
    for _ in range(max(1, attempts)):
        unavailable = tuple(service for service in services if not probe(service))
        if not unavailable:
            stable_successes += 1
            if stable_successes >= max(1, consecutive_successes):
                return ServiceReadiness(True, (), None)
        else:
            stable_successes = 0
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return ServiceReadiness(False, unavailable, "environment_error")


async def run_operator_agent(
    *,
    model_client: OpenAIChatCompletionClient,
    task_instruction: str,
    coordinator_handoff: str,
    tools: TACContainerTools,
    max_tool_iterations: int = 32,
) -> OperatorRun:
    """Run one real AutoGen Operator with bounded task-container tools."""
    observations: list[ToolObservation] = []

    async def run_shell(command: str) -> str:
        """Run one shell command in the current TAC task container and return its full observation."""
        observation = tools.run_shell(command)
        observations.append(observation)
        return observation.as_message()

    async def read_file(path: str) -> str:
        """Read a text file in the current TAC task container."""
        observation = tools.read_file(path)
        observations.append(observation)
        return observation.as_message()

    async def list_directory(path: str = ".") -> str:
        """List files below a path in the current TAC task container."""
        observation = tools.list_directory(path)
        observations.append(observation)
        return observation.as_message()

    async def replace_text(path: str, old_text: str, new_text: str) -> str:
        """Replace a verified text fragment in a task file and return the actual command result."""
        observation = tools.replace_text(path, old_text, new_text)
        observations.append(observation)
        return observation.as_message()

    async def remove_path(path: str) -> str:
        """Remove one inspected task-local file and return the actual command result."""
        observation = tools.remove_path(path)
        observations.append(observation)
        return observation.as_message()

    events: list[dict[str, Any]] = []
    final_message = ""
    # Keep one AgentChat context across tool turns. Recreating the Agent each turn
    # loses the model's own plan and makes it repeatedly rediscover the same file.
    agent = AssistantAgent(
        "Operator",
        model_client=model_client,
        tools=[run_shell, read_file, list_directory, replace_text, remove_path],
        max_tool_iterations=1,
        reflect_on_tool_use=False,
        system_message=(
            "You are the Operator in a real TheAgentCompany task container. Inspect before assuming paths, "
            "credentials, dependencies, or service state. Repair failed attempts using observations. Never claim "
            "success without concrete tool evidence. Use replace_text or remove_path for a requested file "
            "change after inspecting its target; do not report completion from read-only observations."
        ),
    )
    for iteration in range(1, max_tool_iterations + 1):
        prior_observations = "\n\n".join(item.as_message()[-1600:] for item in observations[-3:]) or "none yet"
        task_prefix = (
            f"Task:\n{task_instruction.strip()}\n\nCoordinator handoff:\n{coordinator_handoff.strip()}\n\n"
            if iteration == 1
            else "Continue the same task using your existing tool and reasoning context.\n\n"
        )
        task = (
            f"{task_prefix}Iteration: {iteration}. Latest real tool observations:\n{prior_observations}\n\n"
            f"Execution policy: {operator_iteration_guidance(iteration)}\n\n"
            "Continue the task. Call a tool when further evidence or an action is needed. Only reply with a concise "
            "completion report ending in TERMINATE when the evidence proves the task is complete."
        )
        observation_count = len(observations)
        result = await agent.run(task=task)
        for message in result.messages:
            events.append(
                serialize_agent_event(
                    turn_index=len(events) + 1,
                    source=str(getattr(message, "source", "Operator")),
                    message_type=type(message).__name__,
                    content=str(getattr(message, "content", "")),
                )
            )
        final_message = str(getattr(result.messages[-1], "content", "")) if result.messages else final_message
        if len(observations) == observation_count or has_explicit_termination(final_message):
            break
    return OperatorRun(final_message, tuple(observations), tuple(events))
