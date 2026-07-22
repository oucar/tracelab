"""Read-only eval endpoints for the Evals screen."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from app.deps import store
from app.evals.calibration import calibration_report, load_labels

router = APIRouter(prefix="/api/evals", tags=["evals"])

_EMPTY_CALIBRATION = {"available": False, "n": 0, "eval_run_id": "",
                      "dimensions": [], "overall": None}


def _summary(row: dict) -> dict:
    out = dict(row)
    out["config"] = json.loads(out.pop("config_json") or "{}")
    out["tier1_pass_rate"] = (
        row["tier1_passed"] / row["tier1_scorable"] if row["tier1_scorable"] else None)
    return out


@router.get("")
def list_evals() -> list[dict]:
    return [_summary(r) for r in store().list_eval_runs()]


@router.get("/calibration")
def calibration() -> dict:
    labels = load_labels()
    if labels is None:
        return _EMPTY_CALIBRATION
    return calibration_report(store(), labels)


@router.get("/{eval_run_id}")
def eval_detail(eval_run_id: str) -> dict:
    runs = [r for r in store().list_eval_runs() if r["id"] == eval_run_id]
    if not runs:
        raise HTTPException(status_code=404, detail="eval run not found")
    results = []
    for r in store().eval_results(eval_run_id):
        row = dict(r)
        row["judge"] = json.loads(row["judge"]) if row["judge"] else None
        row["tags"] = json.loads(row["tags"] or "[]")
        results.append(row)
    return {"run": _summary(runs[0]), "results": results}
