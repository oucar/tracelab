"""M5 Task 1: study configs load correctly, and GraphDeps.default accepts per-role overrides."""
from app.evals.study import CONFIGS_DIR, load_study_config


def test_all_four_configs_load():
    configs = [load_study_config(p) for p in sorted(CONFIGS_DIR.glob("*.yaml"))]
    names = {c.name for c in configs}
    assert names == {"mini", "strong-planner", "strong-critic", "strong"}
    for c in configs:
        assert set(c.models) == {"router", "planner", "analyst", "critic", "composer"}
        assert c.judge_model == "gpt-4o"  # judge never varies with the config under test
    strong = next(c for c in configs if c.name == "strong")
    mini = next(c for c in configs if c.name == "mini")
    assert strong.models["analyst"] == "gpt-4o"
    assert mini.models["analyst"] == "gpt-4o-mini"


def test_run_study_records_one_eval_run_per_config(tmp_path):
    from app.evals.study import StudyConfig, run_study
    from app.tracing.store import Store
    from tests.test_harness import _deps, _golden

    st = Store(tmp_path / "t.db")
    configs = [
        StudyConfig(name="a", models={r: "m" for r in
                                      ("router", "planner", "analyst", "critic", "composer")}),
        StudyConfig(name="b", models={r: "m" for r in
                                      ("router", "planner", "analyst", "critic", "composer")}),
    ]
    golden = _golden(tmp_path, 3)
    pairs = run_study(st, configs, lambda cfg: (lambda: _deps(3)), lambda cfg: None,
                      repo_root=tmp_path, enforce_budget=False, golden_sets=golden)
    assert [name for name, _ in pairs] == ["a", "b"]
    labels = {r["label"] for r in st.list_eval_runs()}
    assert labels == {"study:a", "study:b"}


def test_default_deps_accept_model_override(monkeypatch):
    """GraphDeps.default(models=...) must not raise and must not read cheap_mode."""
    from app.agents.llm import GraphDeps
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    from app.config import settings
    settings.cache_clear()
    deps = GraphDeps.default(models={"router": "gpt-4o-mini", "planner": "gpt-4o",
                                    "analyst": "gpt-4o-mini", "critic": "gpt-4o",
                                    "composer": "gpt-4o-mini"})
    assert deps.planner is not None and deps.router is not None
    settings.cache_clear()
