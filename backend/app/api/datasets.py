"""Dataset upload + profiling."""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings
from app.deps import store

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def profile_dataframe(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": [
            {
                "name": str(col),
                "dtype": str(df[col].dtype),
                "nulls": int(df[col].isna().sum()),
                "unique": int(df[col].nunique()),
                "sample": [str(v) for v in df[col].dropna().head(3).tolist()],
            }
            for col in df.columns
        ],
        "preview": df.head(8).astype(str).to_dict(orient="records"),
    }


@router.post("")
async def upload_dataset(file: UploadFile) -> dict:
    cfg = settings()
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported")
    contents = await file.read()
    if len(contents) > cfg.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {cfg.max_upload_mb}MB limit")

    path = cfg.uploads_dir / f"{uuid.uuid4().hex[:12]}.csv"
    path.write_bytes(contents)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        Path(path).unlink(missing_ok=True)
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc

    profile = profile_dataframe(df)
    dataset_id = store().add_dataset(file.filename or "dataset.csv", str(path), profile)
    return {"id": dataset_id, "name": file.filename, "profile": profile}


@router.get("")
def list_datasets() -> list[dict]:
    return [
        {"id": d["id"], "name": d["name"], "profile": d["profile"]}
        for d in store().list_datasets()
    ]


@router.get("/{dataset_id}/suggestions")
def dataset_suggestions(dataset_id: str) -> dict:
    """Starter questions for the dataset — a cheap mini-model call, best-effort."""
    from app.agents.suggest import suggest_questions

    dataset = store().get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(404, "dataset not found")
    profile = dataset["profile"]
    if isinstance(profile, str):
        import json

        profile = json.loads(profile)
    try:
        return {"suggestions": suggest_questions(profile)}
    except Exception:  # noqa: BLE001 — suggestions are a nicety; never fail the UI
        return {"suggestions": []}
