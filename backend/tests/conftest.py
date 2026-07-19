"""Shared test isolation.

execute_run checkpoints every run under thread_id=run_id. Tests reuse fixed
run ids across sessions, so without isolation LangGraph would RESUME finished
threads from data/checkpoints.sqlite3 and accumulate state. Point the
checkpoint DB at a per-test temp path instead (production run ids are unique).
"""

import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _isolated_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINTS_DB_PATH", str(tmp_path / "checkpoints.sqlite3"))
    settings.cache_clear()
    yield
    settings.cache_clear()
