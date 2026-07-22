"""Golden eval set: schema + YAML loader.

expected.value fields are derived, not hand-typed: `python -m app.evals golden --write`
recomputes them from data/samples via derivations.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

GOLDEN_DIR = Path(__file__).parent / "golden"
REPO_ROOT = Path(__file__).resolve().parents[3]


class GoldenExpected(BaseModel):
    kind: Literal["numeric", "categorical", "statistical", "narrative"]
    value: int | float | str | None = None
    tolerance: float = 0.0  # relative band for floats; 0.0 means exact
    direction: Literal["higher", "lower", ""] = ""
    significant: bool | None = None
    method_family: str = ""


class GoldenQuestion(BaseModel):
    id: str
    question: str
    expected: GoldenExpected
    tags: list[str] = Field(default_factory=list)


class GoldenDataset(BaseModel):
    name: str
    csv: str  # repo-root-relative, e.g. data/samples/taxi_trips.csv
    questions: list[GoldenQuestion]


def load_golden(golden_dir: Path = GOLDEN_DIR) -> list[GoldenDataset]:
    return [
        GoldenDataset.model_validate(yaml.safe_load(p.read_text()))
        for p in sorted(golden_dir.glob("*.yaml"))
    ]
