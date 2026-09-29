"""Domain/task/repeat-specific settings recovered from the source experiment log."""
OLD = {
    'reddit': {'27', '66', '69', '723', '726'},
    'swe': {'astropy__astropy-14309', 'django__django-10097', 'django__django-10914', 'pytest-dev__pytest-5631', 'scikit-learn__scikit-learn-10297'},
    'tac': {'ds-format-excel-sheets', 'qa-escalate-emergency', 'sde-change-license-easy', 'sde-close-an-issue', 'sde-install-go'},
}

def settings(domain, task, repeat):
    if domain not in OLD or repeat not in (1, 2, 3):
        raise ValueError('unsupported domain or repeat')
    result = {'LLM_PROVIDER': 'deepseek', 'LLM_MODEL': 'deepseek-v4-flash',
              'LLM_DISABLE_THINKING': '1', 'LLM_MAX_TOKENS': None,
              'LLM_REQUEST_TIMEOUT_SECONDS': '60', 'LLM_TOTAL_REQUEST_TIMEOUT_SECONDS': '180'}
    if str(task) in OLD[domain]:
        if domain == 'reddit':
            # Original systemd launch left the option unset. Preserve that
            # request shape; historical shell-resume inherited state is unknown.
            result['LLM_DISABLE_THINKING'] = None
            source_line, attempts = 73615, 2
        else:
            result['LLM_REQUEST_TIMEOUT_SECONDS'] = '360'
            result['LLM_TOTAL_REQUEST_TIMEOUT_SECONDS'] = '360'
            source_line, attempts = (70723 if domain == 'swe' else 70749), 3
    elif domain == 'reddit':
        result['LLM_MAX_TOKENS'] = '1024'
        result['LLM_TOTAL_REQUEST_TIMEOUT_SECONDS'] = '120'
        source_line, attempts = 84777, 2
    elif repeat == 1:
        result['LLM_MAX_TOKENS'] = '768'
        result['LLM_TOTAL_REQUEST_TIMEOUT_SECONDS'] = '120'
        source_line, attempts = (84730 if domain == 'swe' else 84674), 2
    else:
        source_line, attempts = 87902, 2
    return {'environment': result, 'max_attempts': attempts, 'source_rollout_line': source_line,
            'source_task_id': '019f64e4-3bb4-7b93-a43e-5c1376468685'}
