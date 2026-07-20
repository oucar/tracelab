"""Eval endpoints: list, detail, calibration report."""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.deps import store as store_dep


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.sqlite3"))
    settings.cache_clear()
    store_dep.cache_clear()
    from app.main import app

    yield TestClient(app)
    settings.cache_clear()
    store_dep.cache_clear()


def _seed(st):
    st.add_eval_run(id="ev1", created_at=1000.0, label="baseline", git_sha="abc",
                    config_hash="h", config_json='{"models": {"analyst": "gpt-4o-mini"}}',
                    questions_total=2, tier1_scorable=2, tier1_passed=1,
                    judge_avg=3.5, cost_usd=0.1, duration_ms=5000)
    st.add_eval_result(eval_run_id="ev1", question_id="taxi-001", run_id="r1",
                       dataset="taxi", tags_json='["aggregation"]', tier1_scorable=True,
                       tier1_passed=True, tier1_detail="matched",
                       judge_json=json.dumps({"clarity": 4, "uncertainty_honesty": 4,
                                              "chart_appropriateness": 4,
                                              "methodological_soundness": 4,
                                              "rationale": "ok"}),
                       judge_rationale="ok", cost_usd=0.05, duration_ms=2500)


def test_evals_endpoints(client):
    _seed(store_dep())

    runs = client.get("/api/evals").json()
    assert runs[0]["id"] == "ev1"
    assert runs[0]["tier1_pass_rate"] == 0.5
    assert runs[0]["config"]["models"]["analyst"] == "gpt-4o-mini"

    detail = client.get("/api/evals/ev1").json()
    assert detail["run"]["id"] == "ev1"
    assert detail["results"][0]["judge"]["clarity"] == 4
    assert detail["results"][0]["tags"] == ["aggregation"]

    assert client.get("/api/evals/nope").status_code == 404

    cal = client.get("/api/evals/calibration").json()
    assert cal["available"] is False  # no labels file in test env
