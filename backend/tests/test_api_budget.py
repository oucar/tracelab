"""Daily budget cap: POST /api/runs is refused once today's spend crosses the cap."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.deps import store as store_dep
from app.runtime.events import AgentEvent, EventType


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.sqlite3"))
    settings.cache_clear()
    store_dep.cache_clear()
    from app.main import app

    yield TestClient(app)
    settings.cache_clear()
    store_dep.cache_clear()


def test_run_refused_when_daily_budget_exhausted(client):
    dataset_id = store_dep().add_dataset("d.csv", "/tmp/d.csv", {"rows": 1})
    run_id = store_dep().create_run(dataset_id, "warmup")
    store_dep().add_span(
        AgentEvent(
            run_id=run_id, agent="analyst", type=EventType.LLM_CALL,
            cost_usd=settings().daily_budget_usd + 1.0,
        )
    )
    res = client.post("/api/runs", json={"dataset_id": dataset_id, "question": "again?"})
    assert res.status_code == 429
    assert "budget" in res.json()["detail"].lower()


def test_config_endpoint_reports_spend_and_cheap_mode(client):
    res = client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert body["cheap_mode"] is True
    assert body["daily_budget_usd"] == settings().daily_budget_usd
    assert set(body["models"]) == {"planner", "analyst", "critic", "composer"}
    assert body["spent_today"] >= 0
