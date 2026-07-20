"""Per-agent budgets — hard stops, not suggestions.

Each node instantiates its own tracker; parallel analyst branches therefore
budget independently ("per agent" means per agent instance). Exhaustion raises
BudgetExceeded, which nodes convert into a typed failure the composer must
surface honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings


class BudgetExceeded(Exception):
    def __init__(self, role: str, kind: str) -> None:
        self.role = role
        self.kind = kind  # "llm_calls" | "tool_calls" | "tokens"
        super().__init__(f"{role} budget exhausted ({kind})")


@dataclass
class AgentBudget:
    role: str
    max_llm_calls: int
    max_tool_calls: int
    max_tokens: int
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0

    @classmethod
    def for_role(cls, role: str) -> "AgentBudget":
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
        )

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
