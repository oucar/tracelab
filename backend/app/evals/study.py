"""Multi-config tradeoff study: which roles deserve the strong model?"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import yaml
from pydantic import BaseModel

from app.evals.golden import REPO_ROOT, GoldenDataset

if TYPE_CHECKING:
    # app.evals.harness imports ROLES from this module at import time, so a
    # module-level import of run_eval here would be circular — see run_study.
    from app.agents.llm import GraphDeps
    from app.evals.judge import JudgeFn
    from app.tracing.store import Store

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


def run_study(
    st: "Store",
    configs: list[StudyConfig],
    deps_factory_for: Callable[[StudyConfig], Callable[[], "GraphDeps"]],
    judge_for: Callable[[StudyConfig], "JudgeFn | None"],
    *,
    golden_sets: list[GoldenDataset],
    repo_root: Path = REPO_ROOT,
    enforce_budget: bool = True,
) -> list[tuple[str, str]]:
    """Sweep the golden set once per `StudyConfig`, recording an eval run labeled
    `study:<config name>` for each. The `deps_factory_for`/`judge_for` indirection
    lets callers (tests, CLI) build stubbed or live deps/judge per config without
    `run_study` itself knowing about live models or API keys.
    """
    from app.evals.harness import run_eval

    pairs: list[tuple[str, str]] = []
    for cfg in configs:
        eval_id = run_eval(
            st, golden_sets, deps_factory_for(cfg),
            judge=judge_for(cfg), label=f"study:{cfg.name}",
            models=cfg.models, judge_model=cfg.judge_model,
            repo_root=repo_root, enforce_budget=enforce_budget,
        )
        pairs.append((cfg.name, eval_id))
        print(f"[study] {cfg.name}: eval {eval_id}")
    return pairs
