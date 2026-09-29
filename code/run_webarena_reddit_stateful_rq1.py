"""Real multi-state WebArena Reddit edit communication-fault experiment."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mas_faults.llm_client import get_llm_client
from mas_faults.webarena_reddit_stateful_real import RedditHTTPExecutor, build_edit_request, choose_edit_target
from mas_faults.webarena_shopping_real import parse_verdict


TASKS = (
    ("reddit-731", "MachineLearning", 1, "nvidia-rtx-4090", "EDIT: This news aged well"),
    ("reddit-732", "television", 134868, "the-night-agent-renewed-for-season-2-at-netflix", "Done watching, pretty cool!"),
    ("reddit-733", "television", 135201, "star-trek-starfleet-academy-series-from-alex-kurtzman-and", "Every watch makes me feel like a kid again"),
    ("reddit-734", "television", 135156, "ted-lasso-season-3-premiere-scores-870k-u-s-households-up-59", "Done watching. I love the renew!"),
    ("reddit-735", "television", 135152, "lord-of-the-rings-the-rings-of-power-season-2-cast-adds", "The cast is amazing!"),
    (
        "reddit-736",
        "books",
        59421,
        "friendly-reminder-bookshop-org-exists",
        "Update: local bookstores still matter.",
        "smita16",
    ),
)
CONDITIONS = ("clean", "a5_omission", "a8_state_truncation", "a12_stale_replay", "a6_state_body_poisoning")


def task_dict(value):
    task_id, forum, submission_id, slug, required_append = value[:5]
    task = {"task_id": task_id, "forum": forum, "submission_id": submission_id, "slug": slug, "required_append": required_append}
    if len(value) > 5:
        task["username"] = value[5]
    return task


def select_tasks(tasks, task_ids):
    if not task_ids:
        return list(tasks)
    requested = set(task_ids)
    selected = [task for task in tasks if task["task_id"] in requested]
    missing = sorted(requested - {task["task_id"] for task in selected})
    if missing:
        raise ValueError(f"unknown Reddit task ids: {', '.join(missing)}")
    return selected


def executor_for_task(task):
    executor = RedditHTTPExecutor()
    executor.login(str(task.get("username", "MarvelsGrantMan136")))
    return executor


def inject_state_fault(original, condition, stale):
    if condition == "clean": return copy.deepcopy(original), "none", False
    if condition == "a5_omission": return None, "A5_message_omission", True
    if condition == "a8_state_truncation":
        value = copy.deepcopy(original); value["body"] = value["body"][:max(1, len(value["body"]) // 2)]; value["truncated"] = True
        return value, "A8_message_truncation", True
    if condition == "a12_stale_replay": return copy.deepcopy(stale), "A12_timing_or_session_mismatch", True
    if condition == "a6_state_body_poisoning":
        value = copy.deepcopy(original); value["body"] = stale["body"]; value["evidence"] = stale["evidence"]; value["source_message_id"] = stale["message_id"]
        return value, "A6_message_semantic_corruption", True
    raise ValueError(condition)


def editor_verdict(client, task, state):
    payload = "missing" if state is None else json.dumps(state, ensure_ascii=False)
    return parse_verdict(client.complete('You are the Editor in a real WebArena Reddit edit task. Return JSON only: '
        '{"decision":"accept"|"reject","task_id":string,"reason":string}. The delivered state is the PRE-EDIT snapshot. Accept when it matches current task id, forum, submission id, title and body; then the workflow will append the requested line. Reject missing, stale, truncated, or mismatched state. '
        f"Task: {json.dumps(task)}\nReader state: {payload}"))


def run_one(client, task, condition, stale):
    executor = executor_for_task(task); started = time.perf_counter(); run_id = f"{task['task_id']}-{condition}-{uuid.uuid4().hex[:8]}"; trace_id=f"trace-{uuid.uuid4()}"
    original = executor.read_submission(task["forum"], task["submission_id"], task["slug"])
    original.update({"task_id": task["task_id"], "message_id": f"message-{uuid.uuid4()}", "evidence": json.dumps(original, ensure_ascii=False)})
    delivered, symptom, applied = inject_state_fault(original, condition, stale)
    verdict = editor_verdict(client, task, delivered); wrote=False; verified=None; recovery=[]; foreign_original=None
    try:
        if verdict["decision"] == "accept" and delivered:
            forum, sid, slug = choose_edit_target(task, delivered)
            if (forum, sid, slug) != (task["forum"], task["submission_id"], task["slug"]): foreign_original=executor.read_submission(forum, sid, slug)
            executor.update_submission(forum, sid, slug, build_edit_request(task, delivered)["body"]); wrote=True
        verified=executor.read_submission(task["forum"], task["submission_id"], task["slug"])
        success=bool(wrote and task["required_append"] in verified["body"] and original["body"] in verified["body"])
    finally:
        executor.update_submission(task["forum"], task["submission_id"], task["slug"], original["body"]); recovery.append("current_state_restored")
        if foreign_original:
            executor.update_submission(foreign_original["forum"], foreign_original["submission_id"], foreign_original["slug"], foreign_original["body"]); recovery.append("foreign_state_restored")
    if delivered is None: consequences=["M2_task_timeout_or_failure", "M3_incomplete_information_aggregation"]
    elif condition == "a12_stale_replay" and verdict["decision"] == "accept": consequences=["M5_stale_context_acceptance", "M6_state_inconsistency", "M4_incorrect_collective_decision"]
    elif condition == "a8_state_truncation" and verdict["decision"] == "accept": consequences=["M14_partial_tool_or_message_result_acceptance"] + ([] if success else ["M4_incorrect_collective_decision"])
    elif condition == "a6_state_body_poisoning" and verdict["decision"] == "accept": consequences=["M6_state_inconsistency"] + ([] if success else ["M4_incorrect_collective_decision"])
    elif condition == "a8_state_truncation" and not success: consequences=["M2_task_timeout_or_failure", "M3_incomplete_information_aggregation"]
    elif not success: consequences=["M2_task_timeout_or_failure"]
    else: consequences=[]
    propagation = "clean" if not applied else ("propagated_to_M_final_failure" if consequences and not success else "silent_propagation_to_M" if consequences else "exposed_at_A_only")
    event={"run_id":run_id,"trace_id":trace_id,"timestamp":datetime.now(timezone.utc).isoformat(),"source_agent":"Reader","target_agent":"Editor","original_message":original,"delivered_message":delivered,"fault_type":condition,"fault_applied":applied,"observed_A_symptom":symptom,"observed_M_consequence":consequences or ["none"]}
    return {**event,"benchmark":"WebArena-Verified-Reddit","execution_mode":"live_site_http_stateful","scenario":"reddit_stateful_edit","task_id":task["task_id"],"condition":condition,"fault_id":"none" if condition=="clean" else f"fault-{run_id}","model":client.model_info.model,"provider":client.model_info.provider,"editor_verdict":verdict,"verified_state":verified,"final_task_success":success,"task_score":float(success),"propagation_class":propagation,"recovery_detected":False,"recovery_type":"none","recovery_evidence":[],"environment_restore_evidence":recovery,"latency_ms":round((time.perf_counter()-started)*1000,3),"event":event}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--conditions",default=",".join(CONDITIONS)); p.add_argument("--tasks",type=int,default=5); p.add_argument("--task-ids",default=""); p.add_argument("--output-dir",required=True); a=p.parse_args(); output=Path(a.output_dir); output.mkdir(parents=True,exist_ok=False); client=get_llm_client(); rows=[]; stale=None
    available = [task_dict(x) for x in TASKS]
    requested = [value.strip() for value in a.task_ids.split(",") if value.strip()]
    tasks = select_tasks(available, requested) if requested else available[:a.tasks]
    for task in tasks:
        seed=executor_for_task(task); stale=stale or seed.read_submission(task["forum"],task["submission_id"],task["slug"]); stale.update({"task_id":"previous-task","message_id":"seed","evidence":json.dumps(stale)})
        for condition in a.conditions.split(","):
            row=run_one(client,task,condition,stale); rows.append(row); print(f"DONE {len(rows)} {task['task_id']} {condition} success={row['final_task_success']}",flush=True)
        stale=rows[-1]["original_message"]
    with (output/"llm_communication_traces.jsonl").open("w") as f:
        for r in rows:f.write(json.dumps(r["event"],ensure_ascii=False)+"\n")
    with (output/"llm_communication_runs.jsonl").open("w") as f:
        for r in rows:f.write(json.dumps({k:v for k,v in r.items() if k!="event"},ensure_ascii=False)+"\n")
    summary={"runs":len(rows),"success":sum(r["final_task_success"] for r in rows),"A_exposure":sum(r["observed_A_symptom"]!="none" for r in rows),"M_propagation":sum(r["observed_M_consequence"]!=["none"] for r in rows),"final_failure":sum(not r["final_task_success"] for r in rows)}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); (output/"summary.md").write_text("# Reddit Stateful Fault Matrix\n\n"+json.dumps(summary,indent=2)+"\n")

if __name__=="__main__": main()
