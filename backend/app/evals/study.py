"""Multi-config tradeoff study: which roles deserve the strong model?"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from app.evals.golden import REPO_ROOT

CONFIGS_DIR = REPO_ROOT / "backend" / "configs"
ROLES = ("router", "planner", "analyst", "critic", "composer")


class StudyConfig(BaseModel):
    name: str
    description: str = ""
    models: dict[str, str]
    judge_model: str = "gpt-4o"


def load_study_config(path: Path) -> StudyConfig:
    cfg = StudyConfig.model_validate(yaml.safe_load(path.read_text()))
    missing = set(ROLES) - set(cfg.models)
    if missing:
        raise ValueError(f"{path.name} missing model for roles: {sorted(missing)}")
    return cfg
