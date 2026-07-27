"""Provenance: the recorded config must explain a pass-rate move.

Before prompts were hashed in, editing `analyst.md` and swapping the analyst
model produced the same `config_hash` — regression tracking could say quality
moved but never what moved it.
"""

import json

import pytest

from app.config import settings
from app.evals.harness import (
    CONFIG_SCHEMA_VERSION,
    config_snapshot,
    effective_judge_model,
    prompt_digests,
)


def test_every_agent_prompt_is_digested():
    digests = prompt_digests()
    assert {"analyst", "critic", "composer", "planner", "router", "judge"} <= set(digests)
    assert all(len(d) == 12 for d in digests.values())


def test_editing_a_prompt_changes_the_config_hash(tmp_path, monkeypatch):
    before = config_snapshot(models=None)[1]

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "analyst.md").write_text("original")
    monkeypatch.setattr("app.evals.harness.PROMPTS_DIR", prompts)
    edited_before = config_snapshot(models=None)[1]
    (prompts / "analyst.md").write_text("reworded")
    edited_after = config_snapshot(models=None)[1]

    assert edited_before != edited_after, "a prompt edit must move the hash"
    assert before not in (edited_before, edited_after)


def test_snapshot_records_the_digests_and_a_schema_version():
    blob, _ = config_snapshot(models=None)
    snap = json.loads(blob)
    assert snap["version"] == CONFIG_SCHEMA_VERSION
    assert snap["prompts"] == prompt_digests()


def test_judge_model_recorded_is_the_one_that_will_actually_run(monkeypatch):
    """Under cheap_mode the judge collapses to the analyst model.

    Recording `cfg.judge_model` verbatim there would claim an independent gpt-4o
    judge on a run the config under test actually graded itself.
    """
    monkeypatch.setenv("CHEAP_MODE", "1")
    monkeypatch.setenv("ANALYST_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o")
    settings.cache_clear()
    try:
        assert settings().judge_model == "gpt-4o"
        assert effective_judge_model(None) == "gpt-4o-mini"  # what really runs
        assert json.loads(config_snapshot(models=None)[0])["judge_model"] == "gpt-4o-mini"
        # An explicitly pinned judge (the study path) is honoured as given.
        assert effective_judge_model("gpt-4o") == "gpt-4o"
    finally:
        settings.cache_clear()


def test_a_sweep_with_no_judge_records_no_judge_model():
    """Naming a judge model on a tier-1-only sweep implies a score nobody produced."""
    snap = json.loads(config_snapshot(models=None, judge_ran=False)[0])
    assert snap["judge_model"] is None
    assert config_snapshot(models=None, judge_ran=False)[1] != config_snapshot(models=None)[1]


@pytest.mark.parametrize("field", ["prompts", "judge_model", "models", "version"])
def test_hash_covers_every_recorded_field(field):
    blob, digest = config_snapshot(models=None)
    snap = json.loads(blob)
    snap.pop(field)
    assert json.dumps(snap, sort_keys=True) != blob
    assert digest == config_snapshot(models=None)[1], "snapshot must be deterministic"
