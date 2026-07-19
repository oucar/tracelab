"""Trace store: datasets, runs, spans round-trip."""

import pytest

from app.runtime.events import AgentEvent, EventType
from app.tracing.store import Store


def test_dataset_roundtrip(tmp_path):
    store = Store(tmp_path / "t.db")
    dataset_id = store.add_dataset("taxi.csv", "/tmp/x.csv", {"rows": 10})
    d = store.get_dataset(dataset_id)
    assert d["name"] == "taxi.csv" and d["profile"]["rows"] == 10


def test_run_lifecycle_and_spans(tmp_path):
    store = Store(tmp_path / "t.db")
    dataset_id = store.add_dataset("d.csv", "/tmp/d.csv", {})
    run_id = store.create_run(dataset_id, "what is the mean?")

    store.add_span(
        AgentEvent(run_id=run_id, agent="analyst", type=EventType.LLM_CALL, payload={"i": 1})
    )
    store.finish_run(run_id, "the mean is 4")

    run = store.get_run(run_id)
    assert run["status"] == "finished" and run["answer"] == "the mean is 4"
    spans = store.spans_for_run(run_id)
    assert len(spans) == 1 and spans[0]["payload"] == {"i": 1}


def test_finish_run_persists_structured_result(tmp_path):
    store = Store(tmp_path / "t.sqlite3")
    ds = store.add_dataset("d", "/tmp/d.csv", {"rows": 1})
    run_id = store.create_run(ds, "q")
    store.finish_run(
        run_id, "answer", "finished", result='{"narrative": "answer", "failed": false}'
    )
    run = store.get_run(run_id)
    assert run["result"] == '{"narrative": "answer", "failed": false}'


def test_existing_database_is_migrated_with_result_column(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, question TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running', answer TEXT DEFAULT '',
            created_at REAL NOT NULL
        );
        INSERT INTO runs VALUES ('r1', 'd1', 'q', 'finished', 'a', 0);
        """
    )
    conn.commit()
    conn.close()

    store = Store(db)  # must migrate, not crash
    assert store.get_run("r1")["result"] == ""


def test_cost_since_sums_span_costs(tmp_path):
    store = Store(tmp_path / "t.db")
    ds = store.add_dataset("d", "/tmp/d.csv", {})
    run_id = store.create_run(ds, "q")
    for cost, ts in [(0.5, 100.0), (0.25, 200.0), (1.0, 50.0)]:
        store.add_span(
            AgentEvent(
                run_id=run_id, agent="analyst", type=EventType.LLM_CALL,
                cost_usd=cost, started_at=ts,
            )
        )
    assert store.cost_since(60.0) == 0.75  # the ts=50 span is before the window
    assert store.cost_since(0.0) == 1.75


def test_replay_of_column_roundtrip_and_migration(tmp_path):
    store = Store(tmp_path / "t.db")
    ds = store.add_dataset("d", "/tmp/d.csv", {})
    original = store.create_run(ds, "q")
    replay = store.create_run(ds, "q", replay_of=original)
    assert store.get_run(original)["replay_of"] == ""
    assert store.get_run(replay)["replay_of"] == original

    # a pre-M3 database gains the column on open
    import sqlite3

    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, question TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running', answer TEXT DEFAULT '',
            result TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL
        );
        INSERT INTO runs (id, dataset_id, question, created_at) VALUES ('r1', 'd1', 'q', 0);
        """
    )
    conn.commit()
    conn.close()
    assert Store(db).get_run("r1")["replay_of"] == ""


def test_list_runs_with_stats_rolls_up_cost_latency_quality(tmp_path):
    store = Store(tmp_path / "t.db")
    ds = store.add_dataset("d", "/tmp/d.csv", {})
    run_id = store.create_run(ds, "q")
    store.add_span(
        AgentEvent(
            run_id=run_id, agent="planner", type=EventType.LLM_CALL,
            tokens_in=100, tokens_out=50, cost_usd=0.001, started_at=10.0, duration_ms=500,
        )
    )
    store.add_span(
        AgentEvent(
            run_id=run_id, agent="composer", type=EventType.LLM_CALL,
            tokens_in=200, tokens_out=100, cost_usd=0.002, started_at=12.0, duration_ms=1000,
        )
    )
    result = {
        "narrative": "n",
        "failed": False,
        "charts": [],
        "claims": [
            {"claim": {"text": "a", "kind": "numeric"}, "status": "verified", "detail": ""},
            {"claim": {"text": "b", "kind": "numeric"}, "status": "unverified", "detail": "d"},
        ],
    }
    import json as _json

    store.finish_run(run_id, "answer", "finished", _json.dumps(result))

    empty_run = store.create_run(ds, "no spans yet")

    rows = store.list_runs_with_stats()
    assert [r["id"] for r in rows] == [empty_run, run_id]  # newest first
    row = next(r for r in rows if r["id"] == run_id)
    assert row["cost_usd"] == pytest.approx(0.003)
    assert row["tokens_in"] == 300 and row["tokens_out"] == 150
    # first span starts at 10.0, last ends at 12.0 + 1.0s → 3000ms wall clock
    assert row["duration_ms"] == 3000
    assert row["claims_total"] == 2 and row["claims_verified"] == 1
    assert "result" not in row and "answer" not in row

    empty = next(r for r in rows if r["id"] == empty_run)
    assert empty["cost_usd"] == 0 and empty["duration_ms"] == 0 and empty["claims_total"] == 0
