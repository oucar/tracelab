"""Judge-vs-human agreement: the table that makes the judge trustworthy."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from sklearn.metrics import cohen_kappa_score

from app.evals.judge import DIMENSIONS
from app.tracing.store import Store

LABELS_PATH = Path(__file__).parent / "labels" / "human_labels.yaml"


def label_template(st: Store, eval_run_id: str) -> str:
    entries = []
    for row in st.eval_results(eval_run_id):
        if not row["judge"]:
            continue
        entries.append({
            "question_id": row["question_id"],
            "judge_rationale": row["judge_rationale"],  # context for the labeler; ignored on load
            **{d: None for d in DIMENSIONS},
        })
    return yaml.safe_dump({"eval_run_id": eval_run_id, "labels": entries},
                          sort_keys=False, allow_unicode=True)


def load_labels(path: Path = LABELS_PATH) -> dict | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def calibration_report(st: Store, labels: dict) -> dict:
    judged = {r["question_id"]: json.loads(r["judge"])
              for r in st.eval_results(labels["eval_run_id"]) if r["judge"]}
    pairs: dict[str, list[tuple[int, int]]] = {d: [] for d in DIMENSIONS}
    n = 0
    for entry in labels.get("labels", []):
        qid = entry.get("question_id")
        if qid not in judged or any(entry.get(d) is None for d in DIMENSIONS):
            continue
        n += 1
        for d in DIMENSIONS:
            pairs[d].append((int(entry[d]), int(judged[qid][d])))

    if n == 0:
        return {"available": False, "n": 0, "eval_run_id": labels.get("eval_run_id", ""),
                "dimensions": [], "overall": None}

    def stats(vals: list[tuple[int, int]]) -> dict:
        human = [h for h, _ in vals]
        judge = [j for _, j in vals]
        exact = sum(h == j for h, j in vals) / len(vals) * 100
        within1 = sum(abs(h - j) <= 1 for h, j in vals) / len(vals) * 100
        kappa = 0.0 if len(set(human)) < 2 or len(set(judge)) < 2 else float(
            cohen_kappa_score(human, judge, labels=[1, 2, 3, 4, 5]))
        matrix = [[0] * 5 for _ in range(5)]
        for h, j in vals:
            matrix[h - 1][j - 1] += 1
        return {"n": len(vals), "exact_pct": round(exact, 1),
                "within1_pct": round(within1, 1), "kappa": round(kappa, 3),
                "matrix": matrix}

    dimensions = [{"dimension": d, **stats(pairs[d])} for d in DIMENSIONS]
    pooled = [p for d in DIMENSIONS for p in pairs[d]]
    overall = stats(pooled)
    overall.pop("matrix")
    return {"available": True, "n": n, "eval_run_id": labels["eval_run_id"],
            "dimensions": dimensions, "overall": overall}
