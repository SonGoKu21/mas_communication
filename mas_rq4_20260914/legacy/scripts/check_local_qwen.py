"""Real local inference smoke check; not a WebArena benchmark result."""
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from mas_faults.llm_client import get_llm_client, openai_api_base


def main():
    client = get_llm_client()
    info = client.model_info
    if (info.provider, info.model, openai_api_base(info.base_url)) != (
        "modelscope_local", "Qwen/Qwen3.8-27B", "http://127.0.0.1:18001/v1"
    ):
        raise RuntimeError("unexpected inference target; no request sent")
    with urllib.request.urlopen(openai_api_base(info.base_url) + "/models", timeout=10) as response:
        models = json.load(response)
    if info.model not in {item["id"] for item in models["data"]}:
        raise RuntimeError("requested model not advertised")
    output = Path("/data3/hqn/mas/artifacts") / f"qwen38_27b_smoke_{time.time_ns()}.json"
    checks = [
        ("Return exactly this JSON object and nothing else: {\"status\":\"ready\",\"value\":2}",
         {"status": "ready", "value": 2}),
        ("Return only JSON. A request asks for quantity 2. The observed quantity is 1. "
         "Set decision to reject when the quantities differ, otherwise accept. "
         "Required object: {\"decision\":\"accept or reject\"}.", {"decision": "reject"}),
    ]
    results = []
    for prompt, expected in checks:
        start = time.perf_counter()
        text = client.complete(prompt, json_mode=True)
        try:
            actual = json.loads(text)
        except ValueError:
            actual = None
        results.append({"prompt": prompt, "response": text, "expected": expected,
                        "passed": actual == expected, "latency_seconds": time.perf_counter() - start})
    report = {"kind": "real_local_inference_smoke_not_benchmark", "model": info.model,
              "provider": info.provider, "base_url": info.base_url,
              "timestamp": datetime.now(timezone.utc).isoformat(), "checks": results,
              "prompt_tokens": client.prompt_tokens, "completion_tokens": client.completion_tokens,
              "requests": client.request_log, "all_passed": all(item["passed"] for item in results)}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"artifact": str(output), "all_passed": report["all_passed"],
                      "checks": results, "total_tokens": client.prompt_tokens + client.completion_tokens},
                     ensure_ascii=False), flush=True)
    if not report["all_passed"]:
        raise RuntimeError("local inference smoke failed; do not admit benchmark runs")


if __name__ == "__main__":
    main()
