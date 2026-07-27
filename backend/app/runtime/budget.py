"""Budgets — hard stops, not suggestions.

Two layers, because they fail differently:

**Per-agent** (`AgentBudget`) — call, tool-call, and token caps scoped to one
agent instance. Parallel analyst branches budget independently ("per agent"
means per agent instance). This bounds a single agent looping forever.

**Per-run** (`RunBudget`) — a dollar cap on the whole run, plus the remaining
headroom under today's `daily_budget_usd`. This is the layer that was missing:
per-agent token caps say nothing about dollars, and a daily cap checked only
at admission time is not a cap. Four parallel analysts at 24k tokens each,
plus a critic, a retry, and a composer, can cost real money after the
admission check has already said yes.

`RunBudget` lives in a module-level registry keyed by `run_id`, the same shape
as the event bus, because analyst branches run in separate threads and cannot
share mutable graph state (LangGraph channels exist precisely to stop that).
Charging happens at the single choke point where cost is already computed
(`graph._emit`); enforcement happens where every node already handles
`BudgetExceeded` — immediately before its next model call. That ordering means
a run can overshoot by at most one in-flight call, which is a bounded, honest
overshoot rather than an unbounded one.

Unpriced models (see `tracing.pricing.is_unpriced`) are counted separately.
A run that spent real money on a model we cannot price has an unenforceable
dollar budget, and `RunBudget.unpriced_calls` is what lets callers say so out
loud instead of reporting a reassuring $0.00.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

from app.config import settings
from app.tracing import pricing


class BudgetExceeded(Exception):
    def __init__(self, role: str, kind: str, detail: str = "") -> None:
        self.role = role
        self.kind = kind  # "llm_calls" | "tool_calls" | "tokens" | "run_cost" | "daily_cost"
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{role} budget exhausted ({kind}){suffix}")


# ── per-run dollar budget ────────────────────────────────────────────────────


@dataclass
class RunBudget:
    """Dollar ceiling for one run. `charge` accumulates, `check` enforces."""

    run_id: str
    max_run_usd: float
    #: Dollars left under today's cap when this run started. `inf` disables the
    #: daily layer (unit tests, and any caller that has no store to ask).
    daily_headroom_usd: float = math.inf
    spent_usd: float = 0.0
    unpriced_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def charge(self, model: str, cost_usd: float) -> None:
        with self._lock:
            self.spent_usd += cost_usd
            if pricing.is_unpriced(model):
                self.unpriced_calls += 1

    def check(self, role: str) -> None:
        """Raise if this run has already spent past either ceiling."""
        with self._lock:
            spent, unpriced = self.spent_usd, self.unpriced_calls
        if spent >= self.max_run_usd:
            raise BudgetExceeded(
                role, "run_cost", f"${spent:.4f} of ${self.max_run_usd:.2f} per-run cap"
            )
        if spent >= self.daily_headroom_usd:
            raise BudgetExceeded(
                role,
                "daily_cost",
                f"${spent:.4f} spent this run exhausts today's remaining "
                f"${self.daily_headroom_usd:.4f}",
            )
        if unpriced:
            raise BudgetExceeded(
                role,
                "run_cost",
                f"{unpriced} call(s) used a model with no entry in the price table, "
                "so spend cannot be measured and the dollar cap cannot be enforced",
            )


_registry: dict[str, RunBudget] = {}
_registry_lock = threading.Lock()


def open_run_budget(run_id: str, daily_headroom_usd: float | None = None) -> RunBudget:
    """Register a dollar budget for `run_id`. `None` headroom means unlimited."""
    budget = RunBudget(
        run_id=run_id,
        max_run_usd=settings().max_cost_per_run_usd,
        daily_headroom_usd=math.inf if daily_headroom_usd is None else daily_headroom_usd,
    )
    with _registry_lock:
        _registry[run_id] = budget
    return budget


def get_run_budget(run_id: str) -> RunBudget | None:
    with _registry_lock:
        return _registry.get(run_id)


def close_run_budget(run_id: str) -> RunBudget | None:
    with _registry_lock:
        return _registry.pop(run_id, None)


# ── per-agent call/token budget ──────────────────────────────────────────────


@dataclass
class AgentBudget:
    role: str
    max_llm_calls: int
    max_tool_calls: int
    max_tokens: int
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    #: Set by `for_role` when the caller knows which run it belongs to.
    run_id: str | None = None

    @classmethod
    def for_role(cls, role: str, run_id: str | None = None) -> "AgentBudget":
        cfg = settings()
        llm_caps = {
            "planner": 2,
            "analyst": cfg.max_analyst_iterations + 1,
            "critic": cfg.max_critic_iterations + 1,
            "composer": 1,
            "router": 1,
        }
        tool_caps = {"analyst": cfg.max_analyst_iterations, "critic": cfg.max_critic_iterations}
        return cls(
            role=role,
            max_llm_calls=llm_caps.get(role, 1),
            max_tool_calls=tool_caps.get(role, 0),
            max_tokens=cfg.max_tokens_per_agent,
            run_id=run_id,
        )

    def check_run_cost(self) -> None:
        """Enforce the run's dollar ceiling. No-op when the run has no budget."""
        if self.run_id is None:
            return
        run_budget = get_run_budget(self.run_id)
        if run_budget is not None:
            run_budget.check(self.role)

    def spend_llm(self, tokens_in: int, tokens_out: int) -> None:
        self.llm_calls += 1
        self.tokens += tokens_in + tokens_out
        if self.llm_calls > self.max_llm_calls:
            raise BudgetExceeded(self.role, "llm_calls")
        if self.tokens > self.max_tokens:
            raise BudgetExceeded(self.role, "tokens")

    def spend_tool(self) -> None:
        self.tool_calls += 1
        if self.tool_calls > self.max_tool_calls:
            raise BudgetExceeded(self.role, "tool_calls")
