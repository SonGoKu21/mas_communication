from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from mas_faults.bottom_up_bridge import (
    BridgeExposure,
    BridgeRunSpec,
    enrich_run_record,
    now,
)
from mas_faults.llm.communication_interceptor import (
    DeliveryResult,
    MessageEnvelope,
)


@dataclass(frozen=True)
class NetemTaskCondition:
    name: str
    source_fault_ids: tuple[str, ...]
    source_layers: tuple[str, ...]
    transport_mediation: tuple[str, ...]
    mechanism: str
    tc_args: tuple[str, ...]
    request_timeout_seconds: float
    padding_bytes: int = 524_288
    injection_step: int = 4
    severity: str = "controlled"
    a_latency_threshold_ms: float = 250.0


@dataclass(frozen=True)
class NetemRunSpec:
    task_id: str
    topology: str
    condition: str
    repeat_index: int
    injection_step: int = 4

    @property
    def job_key(self) -> str:
        return "::".join(
            (
                self.task_id,
                self.topology,
                self.condition,
                str(self.injection_step),
                str(self.repeat_index),
            )
        )


@dataclass(frozen=True)
class RelayObservation:
    delivered_payload: Any | None
    latency_ms: float
    http_status: int | None
    payload_preserved: bool
    error: str | None
    response_bytes: int = 0
    relay_message_id: str | None = None


NETEM_TASK_CONDITIONS: dict[str, NetemTaskCondition] = {
    "clean": NetemTaskCondition(
        name="clean",
        source_fault_ids=(),
        source_layers=(),
        transport_mediation=("T1",),
        mechanism="the logical evidence message traverses the Docker TCP relay without qdisc impairment",
        tc_args=(),
        request_timeout_seconds=5.0,
        severity="none",
    ),
    "packet_loss_retransmission": NetemTaskCondition(
        name="packet_loss_retransmission",
        source_fault_ids=("D1", "N1", "T1"),
        source_layers=("D", "N", "T"),
        transport_mediation=("T1",),
        mechanism="tc-netem drops response packets while TCP retransmission attempts to preserve the logical evidence message",
        tc_args=("netem", "loss", "10%", "seed", "42"),
        request_timeout_seconds=5.0,
    ),
    "deadline_delay": NetemTaskCondition(
        name="deadline_delay",
        source_fault_ids=("N2", "T5"),
        source_layers=("N", "T"),
        transport_mediation=("T5",),
        mechanism="tc-netem response delay exceeds the application read deadline",
        tc_args=("netem", "delay", "1200ms", "50ms", "25%"),
        request_timeout_seconds=0.35,
    ),
    "link_outage": NetemTaskCondition(
        name="link_outage",
        source_fault_ids=("D8", "N4", "T4", "T5"),
        source_layers=("D", "N", "T"),
        transport_mediation=("T4", "T5"),
        mechanism="tc-netem drops all relay egress packets, operationalizing a short link outage",
        tc_args=("netem", "loss", "100%"),
        request_timeout_seconds=0.35,
    ),
}


def build_netem_specs(
    *,
    tasks: Sequence[dict[str, Any]],
    topologies: Sequence[str],
    repetitions: int,
    conditions: Sequence[str] | None = None,
) -> list[NetemRunSpec]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    selected = tuple(conditions or NETEM_TASK_CONDITIONS)
    missing = [name for name in selected if name not in NETEM_TASK_CONDITIONS]
    if missing:
        raise ValueError(f"unsupported netem condition(s): {', '.join(missing)}")
    specs = [
        NetemRunSpec(
            task_id=str(task["task_id"]),
            topology=topology,
            condition=condition_name,
            repeat_index=repeat_index,
        )
        for topology in topologies
        for task in tasks
        for condition_name in selected
        for repeat_index in range(1, repetitions + 1)
    ]
    keys = [spec.job_key for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("netem matrix contains duplicate job keys")
    return specs


RelaySender = Callable[..., RelayObservation]


class RelayDeliveryAdapter:
    def __init__(
        self,
        condition: NetemTaskCondition,
        relay_url: str,
        *,
        sender: RelaySender | None = None,
    ) -> None:
        self.condition = condition
        self.relay_url = relay_url.rstrip("/")
        self.sender = sender or send_relay_message
        self.last_observation: RelayObservation | None = None

    @property
    def fault_type(self) -> str:
        return "none" if self.condition.name == "clean" else self.condition.name

    def transmit(self, envelope: MessageEnvelope) -> DeliveryResult:
        started = time.perf_counter()
        try:
            observation = self.sender(
                self.relay_url,
                envelope,
                timeout_seconds=self.condition.request_timeout_seconds,
                padding_bytes=self.condition.padding_bytes,
            )
        except Exception as exc:
            observation = RelayObservation(
                delivered_payload=None,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                http_status=None,
                payload_preserved=False,
                error=type(exc).__name__,
            )
        self.last_observation = observation

        delivered: list[MessageEnvelope]
        notes: list[str]
        if observation.error or observation.delivered_payload is None:
            delivered = []
            symptom = "A2/A4/A5" if self.condition.name == "link_outage" else "A2/A5"
            notes = [
                f"real Docker relay delivery failed: {observation.error or 'empty response'}",
                "no second application-level fault operator was applied",
            ]
        elif not observation.payload_preserved:
            delivered = [_replace_payload(envelope, observation.delivered_payload)]
            symptom = "A6"
            notes = ["real relay returned a payload that differed from the original"]
        else:
            delivered = [_replace_payload(envelope, observation.delivered_payload)]
            if (
                self.condition.name != "clean"
                and observation.latency_ms >= self.condition.a_latency_threshold_ms
            ):
                symptom = "A1"
                notes = ["transport recovered the original payload with application-visible latency"]
            elif self.condition.name == "clean":
                symptom = "none"
                notes = ["real Docker relay delivered the exact logical payload"]
            else:
                symptom = "none"
                notes = ["transport recovered the original payload"]

        return DeliveryResult(
            original=envelope,
            delivered=delivered,
            fault_injected=self.condition.name != "clean",
            a_layer_symptom=symptom,
            latency_ms=observation.latency_ms,
            retry_count=0,
            notes=notes,
        )


def _replace_payload(envelope: MessageEnvelope, payload: Any) -> MessageEnvelope:
    return MessageEnvelope(
        source_agent=envelope.source_agent,
        target_agent=envelope.target_agent,
        payload=payload,
        message_type=envelope.message_type,
        logical_message_id=envelope.logical_message_id,
    )


def send_relay_message(
    relay_url: str,
    envelope: MessageEnvelope,
    *,
    timeout_seconds: float,
    padding_bytes: int,
) -> RelayObservation:
    body = json.dumps(
        {
            "message_id": envelope.logical_message_id,
            "payload": envelope.payload,
            "padding_bytes": padding_bytes,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{relay_url.rstrip('/')}/relay",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
            status = int(response.status)
    except (TimeoutError, OSError, urllib.error.URLError) as exc:
        raise TimeoutError(f"relay request failed: {exc}") from exc
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    decoded = json.loads(raw.decode("utf-8"))
    delivered_payload = decoded.get("payload")
    relay_message_id = decoded.get("message_id")
    preserved = (
        delivered_payload == envelope.payload
        and relay_message_id == envelope.logical_message_id
    )
    return RelayObservation(
        delivered_payload=delivered_payload,
        latency_ms=latency_ms,
        http_status=status,
        payload_preserved=preserved,
        error=None,
        response_bytes=len(raw),
        relay_message_id=relay_message_id,
    )


class DockerNetemController:
    def __init__(
        self,
        *,
        container_name: str,
        repo_root: Path | str = Path("/home/systemai_5/code/mas"),
        host_port: int = 18090,
        image: str = "mas-autogen-netem:latest",
    ) -> None:
        self.container_name = container_name
        self.repo_root = Path(repo_root)
        self.host_port = int(host_port)
        self.image = image

    @property
    def relay_url(self) -> str:
        return f"http://127.0.0.1:{self.host_port}"

    def command_for(self, condition: NetemTaskCondition) -> list[str]:
        if not condition.tc_args:
            return []
        return [
            "docker",
            "exec",
            self.container_name,
            "tc",
            "qdisc",
            "replace",
            "dev",
            "eth0",
            "root",
            *condition.tc_args,
        ]

    def start(self) -> None:
        existing = self._run(
            ["docker", "container", "inspect", self.container_name],
            check=False,
        )
        if existing.returncode == 0:
            label = self._run(
                [
                    "docker",
                    "container",
                    "inspect",
                    "--format",
                    "{{ index .Config.Labels \"mas.netem.bridge\" }}",
                    self.container_name,
                ],
                check=True,
            ).stdout.strip()
            if label != "1":
                raise RuntimeError(
                    f"refusing to replace unrelated container {self.container_name}"
                )
            self.stop()
        self._run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.container_name,
                "--label",
                "mas.netem.bridge=1",
                "--cap-add",
                "NET_ADMIN",
                "-p",
                f"127.0.0.1:{self.host_port}:8080",
                "-e",
                "PYTHONPATH=/app/src",
                "-v",
                f"{self.repo_root}:/app:ro",
                self.image,
                "python",
                "-m",
                "mas_faults.netem_relay",
                "--port",
                "8080",
            ],
            check=True,
        )
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{self.relay_url}/health", timeout=1.0) as response:
                    if response.status == 200:
                        return
            except Exception:
                time.sleep(0.2)
        logs = self._run(
            ["docker", "logs", self.container_name], check=False
        ).stdout[-2000:]
        raise RuntimeError(f"netem relay did not become healthy: {logs}")

    def stop(self) -> None:
        self._run(
            ["docker", "rm", "-f", self.container_name],
            check=False,
        )

    def clear(self) -> None:
        self._run(
            [
                "docker",
                "exec",
                self.container_name,
                "tc",
                "qdisc",
                "del",
                "dev",
                "eth0",
                "root",
            ],
            check=False,
        )

    def apply(self, condition: NetemTaskCondition) -> None:
        self.clear()
        command = self.command_for(condition)
        if command:
            self._run(command, check=True)

    def stats(self) -> str:
        return self._run(
            [
                "docker",
                "exec",
                self.container_name,
                "tc",
                "-s",
                "qdisc",
                "show",
                "dev",
                "eth0",
            ],
            check=True,
        ).stdout.strip()

    @staticmethod
    def _run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=check,
        )


def parse_tc_dropped_packets(stats: str) -> int:
    matches = re.findall(r"dropped\s+(\d+)", stats)
    return max((int(value) for value in matches), default=0)


def enrich_netem_record(
    base_row: dict[str, Any],
    spec: NetemRunSpec,
    *,
    observation: RelayObservation,
    qdisc_before: str,
    qdisc_after: str,
    dropped_delta: int,
) -> dict[str, Any]:
    condition = NETEM_TASK_CONDITIONS[spec.condition]
    a_labels = _labels(base_row.get("observed_A_symptom"))
    a_exposed = any(label != "none" for label in a_labels)
    if condition.name == "clean":
        fault_triggered = False
    elif condition.name == "packet_loss_retransmission":
        fault_triggered = dropped_delta > 0
    elif condition.name == "deadline_delay":
        fault_triggered = bool(observation.error) or observation.latency_ms >= (
            condition.request_timeout_seconds * 1000.0
        )
    else:
        fault_triggered = bool(observation.error) or dropped_delta > 0
    masked_before_a = bool(
        fault_triggered
        and not a_exposed
        and observation.payload_preserved
    )
    evidence = {
        "validation_method": "docker_tc_netem",
        "real_network_path": True,
        "relay_url": "host-published Docker TCP relay",
        "tc_args": list(condition.tc_args),
        "qdisc_before": qdisc_before,
        "qdisc_after": qdisc_after,
        "dropped_packets_delta": dropped_delta,
        "relay_observation": asdict(observation),
    }
    exposure = BridgeExposure(
        condition=condition.name,
        source_fault_ids=condition.source_fault_ids,
        source_layers=condition.source_layers,
        transport_mediation=condition.transport_mediation,
        mechanism=condition.mechanism,
        fault_triggered=fault_triggered,
        masked_before_a=masked_before_a,
        observed_a_faults=tuple(a_labels),
        a_operator=condition.name,
        injection_step=condition.injection_step,
        severity=condition.severity,
        started_at=now(),
        completed_at=now(),
        evidence=evidence,
    )
    bridge_spec = BridgeRunSpec(
        task_id=spec.task_id,
        topology=spec.topology,
        condition=spec.condition,
        repeat_index=spec.repeat_index,
        injection_step=spec.injection_step,
    )
    row = enrich_run_record(
        base_row,
        exposure,
        bridge_spec,
        scenario_name="docker_tc_netem_webarena_bridge",
    )
    row.update(
        {
            "netem_real_network_path": True,
            "netem_tc_args": list(condition.tc_args),
            "netem_qdisc_before": qdisc_before,
            "netem_qdisc_after": qdisc_after,
            "netem_dropped_packets_delta": dropped_delta,
            "relay_observation": asdict(observation),
        }
    )
    for event in row["events"][:2]:
        event["validation_method"] = "docker_tc_netem"
        event["netem_real_network_path"] = True
    return row


def _labels(value: Any) -> list[str]:
    if value in (None, "", []):
        return ["none"]
    values = [value] if isinstance(value, str) else list(value)
    labels = []
    for item in values:
        labels.extend(
            part.strip()
            for part in str(item).replace(",", "/").split("/")
            if part.strip()
        )
    non_none = sorted({label for label in labels if label != "none"})
    return non_none or ["none"]
