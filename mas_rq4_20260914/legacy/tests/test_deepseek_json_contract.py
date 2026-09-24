import pytest

from mas_faults.llm_client import build_completion_payload


@pytest.mark.parametrize("prompt", [
    'Return {"ok":true}.',
    'Return {"task_id":string,"operation":"add_to_cart","quantity":integer}. '
    'Execute the initial stage specified in the task.',
])
def test_flash_json_mode_supplies_required_format_instruction_without_changing_task(prompt):
    payload = build_completion_payload(provider="deepseek", model="deepseek-flash",
                                       prompt=prompt, json_mode=True)
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"] == [
        {"role": "system", "content": "Return a valid JSON object."},
        {"role": "user", "content": prompt},
    ]
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 2048


@pytest.mark.parametrize("prompt", ["Return JSON.", "Return json.", "Return Json."])
def test_existing_json_instruction_is_not_duplicated(prompt):
    payload = build_completion_payload(provider="deepseek", model="deepseek-flash",
                                       prompt=prompt, json_mode=True)
    assert payload["messages"] == [{"role": "user", "content": prompt}]


@pytest.mark.parametrize("provider,model,json_mode", [
    ("deepseek", "deepseek-flash", False),
    ("modelscope_local", "Qwen/Qwen3.8-27B", True),
    ("deepseek", "deepseek-v4-pro", True),
])
def test_other_payloads_are_unchanged(provider, model, json_mode):
    payload = build_completion_payload(provider=provider, model=model, prompt="Task.",
                                       json_mode=json_mode)
    assert payload["messages"] == [{"role": "user", "content": "Task."}]
