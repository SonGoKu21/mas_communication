"""Never contend with the live experiment's host-wide lock in offline tests."""
import os
import pytest


@pytest.fixture(autouse=True)
def private_workflow_lock(tmp_path, monkeypatch):
    real_open = os.open

    def redirected(path, *args, **kwargs):
        if os.fspath(path) == '/tmp/mas-shopping-multimechanism.lock':
            path = tmp_path / 'test-workflow.lock'
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, 'open', redirected)
