"""Trace store: datasets, runs, spans round-trip."""

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
