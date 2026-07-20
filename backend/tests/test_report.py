from app.evals.report import study_markdown, study_rows, tradeoff_png
from app.tracing.store import Store


def _seed(st: Store) -> None:
    rows = [("study:mini", 100.0, 0.30, 4, 0.9, 3.6), ("study:mini", 400.0, 0.35, 4, 0.9, 3.7),
            ("study:strong", 200.0, 4.20, 4, 1.0, 4.5)]
    for i, (label, ts, cost, total, rate, judge) in enumerate(rows):
        st.add_eval_run(id=f"ev{i}", created_at=ts, label=label, git_sha="s",
                        config_hash="h", config_json="{}", questions_total=total,
                        tier1_scorable=total, tier1_passed=int(total * rate),
                        judge_avg=judge, cost_usd=cost, duration_ms=total * 8000)


def test_study_rows_latest_per_config(tmp_path):
    st = Store(tmp_path / "t.db")
    _seed(st)
    rows = study_rows(st)
    assert {r["config"] for r in rows} == {"mini", "strong"}
    mini = next(r for r in rows if r["config"] == "mini")
    assert mini["eval_run_id"] == "ev1"  # the newer of the two mini runs
    assert mini["cost_per_question"] == round(0.35 / 4, 4)
    assert mini["latency_s_per_question"] == 8.0


def test_markdown_and_png(tmp_path):
    st = Store(tmp_path / "t.db")
    _seed(st)
    rows = study_rows(st)
    md = study_markdown(rows)
    assert "| Config |" in md and "mini" in md and "strong" in md
    out = tmp_path / "tradeoff.png"
    tradeoff_png(rows, out)
    assert out.stat().st_size > 1000


def test_tradeoff_png_empty_rows_is_noop(tmp_path):
    out = tmp_path / "x.png"
    tradeoff_png([], out)
    assert not out.exists()
