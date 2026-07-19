"""Central settings. Everything tunable lives here, driven by environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""

    # CHEAP_MODE=1 forces mini models everywhere (day-to-day dev default).
    cheap_mode: bool = True
    analyst_model: str = "gpt-4o-mini"
    composer_model: str = "gpt-4o-mini"
    strong_model: str = "gpt-4o"

    # Budgets — hard stops, not suggestions.
    max_analyst_iterations: int = 3
    sandbox_timeout_seconds: int = 30  # cold scipy imports under parallel load need headroom
    sandbox_memory_mb: int = 1024
    sandbox_cpu_seconds: int = 15
    max_upload_mb: int = 25

    # M2 — planning, verification, budgets.
    planner_model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o-mini"
    max_plan_steps: int = 4

    # M4 — tier-2 scoring.
    judge_model: str = "gpt-4o"
    max_retries: int = 1  # bounded retry after a critic discrepancy
    max_critic_iterations: int = 3  # critic sandbox executions
    numeric_rel_tolerance: float = 0.01
    alpha: float = 0.05
    max_tokens_per_agent: int = 24_000

    # M3 — observability.
    daily_budget_usd: float = 2.0  # hard cap on real-run spend per UTC day

    db_path: Path = DATA_DIR / "tracelab.sqlite3"
    checkpoints_db_path: Path = DATA_DIR / "checkpoints.sqlite3"
    uploads_dir: Path = DATA_DIR / "uploads"

    def model_for(self, role: str) -> str:
        if self.cheap_mode:
            # In cheap mode, collapse all roles to analyst_model (including judge).
            return self.analyst_model
        return {
            "planner": self.planner_model,
            "analyst": self.analyst_model,
            "critic": self.critic_model,
            "composer": self.composer_model,
            "judge": self.judge_model,
        }.get(role, self.analyst_model)


@lru_cache
def settings() -> Settings:
    s = Settings()
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    return s
