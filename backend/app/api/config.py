"""Read-only config surface for the UI: models, CHEAP_MODE, budget state."""

from fastapi import APIRouter

from app.config import settings
from app.deps import store
from app.tracing.store import utc_midnight

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("")
def get_config() -> dict:
    cfg = settings()
    return {
        "cheap_mode": cfg.cheap_mode,
        "daily_budget_usd": cfg.daily_budget_usd,
        "spent_today": store().cost_since(utc_midnight()),
        "models": {role: cfg.model_for(role) for role in ("planner", "analyst", "critic", "composer")},
    }
