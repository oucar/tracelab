import json

import yaml

from app.evals.calibration import calibration_report, label_template
from app.tracing.store import Store


def _seed(st: Store) -> None:
    st.add_eval_run(id="ev1", created_at=1.0, label="", git_sha="", config_hash="",
                    config_json="{}", questions_total=3, tier1_scorable=2,
                    tier1_passed=2, judge_avg=4.0, cost_usd=0, duration_ms=0)
    judge = {"clarity": 4, "uncertainty_honesty": 5, "chart_appropriateness": 3,
             "methodological_soundness": 4, "rationale": "ok"}
    for qid, j in [("q1", judge), ("q2", {**judge, "clarity": 2}), ("q3", None)]:
        st.add_eval_result(eval_run_id="ev1", question_id=qid, run_id=f"r-{qid}",
                           dataset="taxi", tags_json="[]", tier1_scorable=True,
                           tier1_passed=True, tier1_detail="",
                           judge_json=json.dumps(j) if j else None,
                           judge_rationale="ok" if j else "", cost_usd=0, duration_ms=0)


def test_label_template_lists_judged_questions_with_null_scores(tmp_path):
    st = Store(tmp_path / "t.db")
    _seed(st)
    parsed = yaml.safe_load(label_template(st, "ev1"))
    assert parsed["eval_run_id"] == "ev1"
    ids = [entry["question_id"] for entry in parsed["labels"]]
    assert ids == ["q1", "q2"]  # q3 was never judged
    assert parsed["labels"][0]["clarity"] is None


def test_calibration_report_agreement_and_kappa(tmp_path):
    st = Store(tmp_path / "t.db")
    _seed(st)
    labels = {"eval_run_id": "ev1", "labels": [
        {"question_id": "q1", "clarity": 4, "uncertainty_honesty": 4,
         "chart_appropriateness": 3, "methodological_soundness": 4},
        {"question_id": "q2", "clarity": 2, "uncertainty_honesty": 5,
         "chart_appropriateness": 4, "methodological_soundness": 4},
    ]}
    report = calibration_report(st, labels)
    assert report["available"] and report["n"] == 2
    clarity = next(d for d in report["dimensions"] if d["dimension"] == "clarity")
    assert clarity["exact_pct"] == 100.0  # judge said 4 and 2; human said 4 and 2
    chart = next(d for d in report["dimensions"]
                 if d["dimension"] == "chart_appropriateness")
    assert chart["exact_pct"] == 50.0 and chart["within1_pct"] == 100.0
    assert clarity["matrix"][3][3] == 1  # human 4 / judge 4 bucket
    assert 0.0 <= report["overall"]["within1_pct"] <= 100.0
