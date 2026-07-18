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
