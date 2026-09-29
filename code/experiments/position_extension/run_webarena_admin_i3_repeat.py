#!/usr/bin/env python3
"""Admin-only repeat partitioning around the unchanged historical runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def select_repeats(jobs, repeats):
    selected = set(repeats)
    if not selected or not selected.issubset({1, 2, 3}):
        raise ValueError('repeat indices must be a nonempty subset of 1,2,3')
    return [job for job in jobs if job.repeat_index in selected]


def is_preinjection_format_failure(row):
    error = str(row.get('error') or '')
    return bool(row.get('condition') != 'clean' and not row.get('fault_applied')
                and any(marker in error for marker in (
                    'model output is not strict JSON',
                    'model JSON does not match the required schema')))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--repeat-indices', nargs='+', type=int, required=True)
    options, remaining = parser.parse_known_args()
    select_repeats([], options.repeat_indices)
    sys.argv = [sys.argv[0], *remaining]
    import run_webarena_admin_main_confirmation as base
    original_prepare = base.prepare_execution_jobs
    original_pre_failure = base._is_clean_matched_pre_injection_model_failure

    def prepare(*args, **kwargs):
        return select_repeats(original_prepare(*args, **kwargs), options.repeat_indices)

    def pre_failure(accepted_rows, candidate, *, reference_rows=()):
        return (is_preinjection_format_failure(candidate)
                or original_pre_failure(accepted_rows, candidate,
                                        reference_rows=reference_rows))

    base.prepare_execution_jobs = prepare
    base._is_clean_matched_pre_injection_model_failure = pre_failure
    output = Path(remaining[remaining.index('--output-dir') + 1])
    try:
        base.main()
    finally:
        if output.exists():
            metadata = {
                'repeat_indices': options.repeat_indices,
                'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'base_runner_sha256': hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                'selection': 'filter original scheduled jobs; preserve job_key, matrix_run_index and schedule_position',
                'model_format_failure_policy': 'retain first pre-injection strict-JSON/schema error as pre_injection_model_failure; injection_valid=false, fault_effect_attributable=false; no retry to select success',
                'unchanged': ['tasks', 'prompts', 'tool execution', 'fault operator', 'task evaluator', 'base strict gate'],
                'inference_environment': {key: os.environ.get(key) for key in [
                    'LLM_PROVIDER', 'LLM_MODEL', 'LLM_DISABLE_THINKING',
                    'LLM_MAX_TOKENS', 'LLM_REQUEST_TIMEOUT_SECONDS',
                    'LLM_TOTAL_REQUEST_TIMEOUT_SECONDS']},
                'arguments': remaining,
            }
            (output / 'repeat_wrapper_config.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
