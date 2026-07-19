# M2 — Multi-agent + Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the M1 analyst→composer graph with the full multi-agent pipeline: planner (structured plans) → parallel analysts (`Send` fan-out) → independent critic with reconciliation gates → composer, plus per-agent budgets, SqliteSaver checkpointing, first stats methods (mean comparison + correlation) with methodology chips, and ChartSpec → MUI X Charts rendering with claim badges in the UI.

**Architecture:** The graph's routing lives in conditional edges: planner fans out one analyst branch per independent plan step via `Send`; the critic re-derives every claim with independently written code (it never sees analyst code), deterministic reconciliation in `runtime/reconcile.py` produces per-claim verdicts, and a conditional edge after the critic routes to composer (all verified), one bounded retry (discrepancy, critic findings injected), or honest-failure composition. Claims/verdicts/charts flow to the UI as a structured `FinalAnswer` in the `run_finished` event payload and persist in a new `runs.result` column.

**Tech Stack:** LangGraph 1.2.9 (`langgraph.types.Send`, `input_schema=` on `add_node`, `SqliteSaver` from `langgraph-checkpoint-sqlite`), Pydantic v2, FastAPI, scipy (sandbox), React + MUI + @mui/x-charts (already installed) + zod (new).

## Global Constraints

- Python 3.11+, line length 100 (ruff), tests must pass with **no API key** (all LLMs stubbed via `GraphDeps`).
- LangGraph is the only agent framework; reconciliation/budgets/chart validation are deliberately custom plain Python.
- `AgentEvent` stays the single event shape; new agents emit through it (planner/critic get `agent` values already reserved in the frontend types).
- MUI X community tier only (no Pro components).
- Frontend must pass `npm run typecheck`.
- Commit after every task; never push.
- Sandbox libraries in M2: pandas, numpy, scipy (statsmodels/sklearn arrive M3).

**Files overview** (Create/Modify):

| File | Task |
|---|---|
| M `backend/app/config.py` | 1 |
| C `backend/app/runtime/budget.py`, C `backend/tests/test_budget.py` | 1 |
| M `backend/app/runtime/state.py` | 2 |
| C `backend/app/runtime/chartspec.py`, C `backend/tests/test_chartspec.py` | 3 |
| C `backend/app/runtime/reconcile.py`, C `backend/tests/test_reconcile.py` | 4 |
| M `backend/app/agents/schemas.py`, M `backend/app/agents/llm.py`, C `backend/tests/test_llm.py` | 5 |
| C `backend/app/agents/prompts/planner.md`, C `.../critic.md`, M `.../analyst.md`, M `.../composer.md` | 6 |
| M `backend/app/runtime/graph.py`, M `backend/tests/test_graph.py` | 6 |
| M `backend/pyproject.toml`, C `backend/tests/test_checkpoint.py` | 7 |
| M `backend/app/tracing/store.py`, M `backend/app/api/runs.py`, M `backend/tests/test_store.py` | 8 |
| M `frontend/package.json` (zod), C `frontend/src/lib/chartSpec.ts`, M `frontend/src/lib/types.ts` | 9 |
| C `frontend/src/components/ChartSpecRenderer.tsx`, C `.../ClaimBadge.tssx→tsx`, C `.../MethodologyChip.tsx`, C `.../AnswerPanel.tsx` | 9 |
| M `frontend/src/store/runStore.ts`, M `frontend/src/pages/Workbench.tsx`, M `frontend/src/components/EventLog.tsx` | 10 |
| M `MILESTONES.md` | 11 |

---

### Task 1: Config + per-agent budgets

**Files:**
- Modify: `backend/app/config.py`
- Create: `backend/app/runtime/budget.py`
- Test: `backend/tests/test_budget.py`

**Interfaces:**
- Produces: `AgentBudget.for_role(role: str) -> AgentBudget` with `spend_llm(tokens_in: int, tokens_out: int)` and `spend_tool()`, both raising `BudgetExceeded(role, kind)`. New settings: `max_plan_steps=4`, `max_retries=1`, `max_critic_iterations=3`, `numeric_rel_tolerance=0.01`, `alpha=0.05`, `max_tokens_per_agent=24_000`, `planner_model`, `critic_model`, `checkpoints_db_path`.

- [ ] **Step 1: Write the failing test** (`backend/tests/test_budget.py`)

```python
"""Per-agent budgets are hard stops, not suggestions."""

import pytest

from app.runtime.budget import AgentBudget, BudgetExceeded


def test_llm_call_budget_exhausts():
    b = AgentBudget(role="planner", max_llm_calls=2, max_tool_calls=0, max_tokens=10_000)
    b.spend_llm(100, 50)
    b.spend_llm(100, 50)
    with pytest.raises(BudgetExceeded) as exc:
        b.spend_llm(100, 50)
    assert exc.value.role == "planner" and exc.value.kind == "llm_calls"


def test_token_budget_exhausts():
    b = AgentBudget(role="analyst", max_llm_calls=100, max_tool_calls=10, max_tokens=500)
    b.spend_llm(300, 100)
    with pytest.raises(BudgetExceeded) as exc:
        b.spend_llm(300, 100)
    assert exc.value.kind == "tokens"


def test_tool_call_budget_exhausts():
    b = AgentBudget(role="critic", max_llm_calls=10, max_tool_calls=1, max_tokens=10_000)
    b.spend_tool()
    with pytest.raises(BudgetExceeded) as exc:
        b.spend_tool()
    assert exc.value.kind == "tool_calls"


def test_for_role_reads_settings():
    b = AgentBudget.for_role("analyst")
    assert b.max_tool_calls == 3          # cfg.max_analyst_iterations
    assert b.max_llm_calls == 4           # iterations + 1 finishing turn
    assert b.max_tokens == 24_000
    assert AgentBudget.for_role("composer").max_tool_calls == 0
```

- [ ] **Step 2: Run to verify failure:** `cd backend && .venv/bin/pytest tests/test_budget.py -q` → ImportError.
- [ ] **Step 3: Implement.**

`backend/app/config.py` — add after `max_upload_mb`:

```python
    # M2 — planning, verification, budgets.
    planner_model: str = "gpt-4o-mini"
    critic_model: str = "gpt-4o-mini"
    max_plan_steps: int = 4
    max_retries: int = 1               # bounded retry after a critic discrepancy
    max_critic_iterations: int = 3     # critic sandbox executions
    numeric_rel_tolerance: float = 0.01
    alpha: float = 0.05
    max_tokens_per_agent: int = 24_000

    checkpoints_db_path: Path = DATA_DIR / "checkpoints.sqlite3"
```

and extend `model_for`:

```python
    def model_for(self, role: str) -> str:
        if self.cheap_mode:
            return self.analyst_model
        return {
            "planner": self.planner_model,
            "analyst": self.analyst_model,
            "critic": self.critic_model,
            "composer": self.composer_model,
        }.get(role, self.analyst_model)
```

`backend/app/runtime/budget.py`:

```python
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
```

- [ ] **Step 4: Run:** `pytest tests/test_budget.py -q` → 4 passed. Full suite still green: `pytest -q`.
- [ ] **Step 5: Commit:** `git add -A && git commit -m "feat(m2): per-agent budgets + M2 settings"`

---

### Task 2: State models v2

**Files:**
- Modify: `backend/app/runtime/state.py`

**Interfaces:**
- Produces (all Pydantic, imported everywhere downstream): `Methodology`, `Claim`, `PlanStep`, `Plan`, `Verdict`, `AnalystResult`, `VerifiedClaim`, `FinalAnswer`, `AnalystTask` and the reshaped `RunState`. `SandboxResult`/`AnalysisStep` unchanged.
- Removes: `RunState.steps`, `.analyst_findings`, `.analyst_failed`, `.failure_reason` (replaced by `analyst_results`; test_graph.py is rewritten in Task 6 — expect it to fail from this task until then, that's the only allowed intermediate red).

- [ ] **Step 1: Implement** — replace the `RunState` section of `state.py` (keep `SandboxResult`, `AnalysisStep` as-is; note `ChartSpec` imports from Task 3's module, so within this task reference it lazily):

Actually to avoid a forward dependency, `ChartSpec` lives in `runtime/chartspec.py` (Task 3). Do Task 3 **before** wiring `chart_specs` — in this task use `list[dict]` placeholder? **No placeholders.** Instead: reorder — this task creates every model that does NOT need ChartSpec, and Task 3 adds `chart_specs: list[ChartSpec]` to `AnalystResult` and `charts: list[ChartSpec]` to `FinalAnswer` when ChartSpec exists. Code for this task:

```python
"""Typed graph state. LangGraph nodes read and return partial updates of this."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class SandboxResult(BaseModel):
    code: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisStep(BaseModel):
    iteration: int
    result: SandboxResult


class Methodology(BaseModel):
    """How a statistical claim was derived — rendered as a methodology chip."""

    method: str  # e.g. "Welch t-test", "Mann-Whitney U", "Pearson correlation"
    n: int
    p_value: float | None = None
    effect_size: float | None = None
    effect_size_name: str = ""  # "Cohen's d", "r", "rank-biserial"
    assumptions: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    """One atomic, checkable statement an analyst makes."""

    id: str = ""  # assigned by the analyst node: "<step_id>-<n>"
    step_id: int = 0
    text: str
    kind: Literal["numeric", "categorical", "statistical"]
    value: float | str | None = None  # numeric value or category label
    direction: Literal["higher", "lower", "none"] | None = None  # statistical only
    significant: bool | None = None  # statistical only, at alpha
    methodology: Methodology | None = None


class PlanStep(BaseModel):
    id: int = 0
    description: str
    method: Literal["descriptive", "mean_comparison", "correlation"] = "descriptive"


class Plan(BaseModel):
    steps: list[PlanStep]
    rationale: str = ""


class Verdict(BaseModel):
    """The critic's judgment of one claim after deterministic reconciliation."""

    claim_id: str
    status: Literal["verified", "discrepancy", "unverifiable"]
    critic_value: float | str | None = None
    reason: str = ""
    methodology_ok: bool | None = None  # statistical claims only
```

then (after ChartSpec exists, final form shown here for reference — Task 3 finalizes):

```python
class AnalystResult(BaseModel):
    step_id: int
    findings: str = ""
    claims: list[Claim] = Field(default_factory=list)
    chart_specs: list[ChartSpec] = Field(default_factory=list)   # Task 3
    chart_rejections: list[str] = Field(default_factory=list)
    iterations: list[AnalysisStep] = Field(default_factory=list)
    failed: bool = False
    failure_reason: str = ""


class VerifiedClaim(BaseModel):
    claim: Claim
    status: Literal["verified", "unverified"]
    detail: str = ""  # discrepancy/unchecked explanation


class FinalAnswer(BaseModel):
    narrative: str
    claims: list[VerifiedClaim] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)        # Task 3
    failed: bool = False


class AnalystTask(BaseModel):
    """Send-payload for one analyst branch (also the retry payload)."""

    run_id: str
    question: str
    dataset_path: str
    dataset_profile: dict[str, Any] = Field(default_factory=dict)
    step: PlanStep
    critic_feedback: str = ""


class RunState(BaseModel):
    """State flowing through the M2 graph: planner → analysts (Send) → critic → composer."""

    run_id: str
    question: str
    dataset_path: str
    dataset_profile: dict[str, Any] = Field(default_factory=dict)

    plan: Plan | None = None
    planner_failed: bool = False
    planner_failure_reason: str = ""

    # Parallel analyst branches append here; last result per step_id wins.
    analyst_results: Annotated[list[AnalystResult], operator.add] = Field(default_factory=list)

    verdicts: list[Verdict] = Field(default_factory=list)
    retry_count: int = 0
    retry_steps: list[int] = Field(default_factory=list)  # set by critic when retrying

    final_answer: str = ""
    final: FinalAnswer | None = None
```

To keep Task 2 green on its own, do Tasks 2+3 as one commit if preferred; the checklist keeps them separate for review clarity but Task 2's step 2 runs only `pytest tests/test_budget.py tests/test_events.py tests/test_sandbox.py tests/test_store.py` (test_graph is legitimately red until Task 6).

- [ ] **Step 2: Run:** `pytest tests/test_budget.py tests/test_events.py tests/test_sandbox.py tests/test_store.py -q` → green.
- [ ] **Step 3: Commit:** `git commit -am "feat(m2): run state v2 — plans, claims, verdicts"`

---

### Task 3: ChartSpec + column validation

**Files:**
- Create: `backend/app/runtime/chartspec.py`
- Modify: `backend/app/runtime/state.py` (add `chart_specs`/`charts` fields per Task 2 note)
- Test: `backend/tests/test_chartspec.py`

**Interfaces:**
- Produces: `ChartSpec` (kind: line|bar|scatter|pie|histogram, x: str, y: list[str] min 1, data ≤500 rows, source_columns), `extract_chart_specs(artifacts: list[dict], columns: list[str]) -> tuple[list[ChartSpec], list[str]]` (valid specs, rejection reasons). Sandbox artifact convention: analysts write `./artifacts/chart_*.json`.

- [ ] **Step 1: Failing test** (`backend/tests/test_chartspec.py`):

```python
"""ChartSpecs are verified structured output: bad columns are rejected like bad numbers."""

from app.runtime.chartspec import extract_chart_specs

COLUMNS = ["fare", "tip", "day"]


def art(name: str, data: dict) -> dict:
    return {"kind": "json", "name": name, "data": data}


def valid_spec() -> dict:
    return {
        "kind": "bar",
        "title": "Mean fare by day",
        "x": "day",
        "y": ["mean_fare"],
        "data": [{"day": "Mon", "mean_fare": 12.1}, {"day": "Sat", "mean_fare": 15.3}],
        "source_columns": ["day", "fare"],
    }


def test_valid_spec_is_accepted():
    specs, rejections = extract_chart_specs([art("chart_fare.json", valid_spec())], COLUMNS)
    assert len(specs) == 1 and rejections == []
    assert specs[0].kind == "bar" and specs[0].y == ["mean_fare"]


def test_unknown_source_column_is_rejected():
    bad = valid_spec() | {"source_columns": ["day", "surge_multiplier"]}
    specs, rejections = extract_chart_specs([art("chart_x.json", bad)], COLUMNS)
    assert specs == [] and "surge_multiplier" in rejections[0]


def test_data_keys_must_match_axes():
    bad = valid_spec() | {"y": ["median_fare"]}
    specs, rejections = extract_chart_specs([art("chart_x.json", bad)], COLUMNS)
    assert specs == [] and "median_fare" in rejections[0]


def test_schema_violation_is_rejected_not_raised():
    specs, rejections = extract_chart_specs([art("chart_x.json", {"kind": "sunburst"})], COLUMNS)
    assert specs == [] and len(rejections) == 1


def test_non_chart_artifacts_are_ignored():
    specs, rejections = extract_chart_specs([art("result.json", {"whatever": 1})], COLUMNS)
    assert specs == [] and rejections == []


def test_empty_data_is_rejected():
    specs, rejections = extract_chart_specs([art("chart_x.json", valid_spec() | {"data": []})], COLUMNS)
    assert specs == [] and "data" in rejections[0].lower()
```

- [ ] **Step 2: Run** → ImportError.
- [ ] **Step 3: Implement** `backend/app/runtime/chartspec.py`:

```python
"""ChartSpec — charts as verified structured output, not code.

Analysts write `./artifacts/chart_*.json` in the sandbox; each is validated
(schema + dataset-column check) exactly the way numeric claims are verified.
An invalid chart is rejected with a reason, never rendered.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

MAX_DATA_ROWS = 500


class ChartSpec(BaseModel):
    kind: Literal["line", "bar", "scatter", "pie", "histogram"]
    title: str = ""
    x: str
    y: list[str] = Field(min_length=1)
    data: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_DATA_ROWS)
    x_label: str = ""
    y_label: str = ""
    source_columns: list[str] = Field(min_length=1)  # dataset columns this derives from


def extract_chart_specs(
    artifacts: list[dict], columns: list[str]
) -> tuple[list[ChartSpec], list[str]]:
    """Validate chart_*.json sandbox artifacts. Returns (valid specs, rejection reasons)."""
    specs: list[ChartSpec] = []
    rejections: list[str] = []
    for artifact in artifacts:
        name = artifact.get("name", "")
        if not name.startswith("chart_"):
            continue
        if artifact.get("kind") != "json":
            rejections.append(f"{name}: not valid JSON")
            continue
        try:
            spec = ChartSpec.model_validate(artifact.get("data"))
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"]) or "spec"
            rejections.append(f"{name}: invalid spec — {loc}: {first['msg']}")
            continue
        unknown = [c for c in spec.source_columns if c not in columns]
        if unknown:
            rejections.append(f"{name}: references nonexistent column(s): {', '.join(unknown)}")
            continue
        fields = [spec.x, *spec.y]
        missing = [f for f in fields if f not in spec.data[0]]
        if missing:
            rejections.append(f"{name}: data rows missing field(s): {', '.join(missing)}")
            continue
        specs.append(spec)
    return specs, rejections
```

Then in `state.py`: `from app.runtime.chartspec import ChartSpec` and add the two fields from Task 2's reference block (`AnalystResult.chart_specs`, `AnalystResult.chart_rejections`, `FinalAnswer.charts`).

- [ ] **Step 4: Run:** `pytest tests/test_chartspec.py -q` → 6 passed.
- [ ] **Step 5: Commit:** `git commit -am "feat(m2): ChartSpec with column validation"`

---

### Task 4: Reconciliation tolerances

**Files:**
- Create: `backend/app/runtime/reconcile.py`
- Test: `backend/tests/test_reconcile.py`

**Interfaces:**
- Consumes: `Claim`, `Verdict` from `app.runtime.state`; `CriticFinding` — defined **here** (not in agents/schemas) to keep this module dependency-light; agents/schemas re-exports it in Task 5.
- Produces: `CriticFinding` model; `reconcile_claims(claims: list[Claim], findings: list[CriticFinding], rel_tol: float) -> list[Verdict]`; `numbers_match(a: float, b: float, rel_tol: float) -> bool`.

Tolerance policy (from plan §7 "Critic false alarms"):
- floats: relative epsilon (`rel_tol`, default 0.01), with near-zero guard;
- integral values (both sides `float.is_integer()` or int): exact;
- categorical: case-insensitive, whitespace-stripped string equality;
- statistical: `direction` and `significant` must both match; a `methodology_ok is False` from the critic is a discrepancy even when numbers agree;
- claim with no matching finding → `unverifiable`; finding with `could_not_verify` → `unverifiable`.

- [ ] **Step 1: Failing test** (`backend/tests/test_reconcile.py`):

```python
"""Tolerance policy: design the rules, test the rules (plan §7 — critic false alarms)."""

from app.runtime.reconcile import CriticFinding, numbers_match, reconcile_claims
from app.runtime.state import Claim


def claim_num(value, id="1-1") -> Claim:
    return Claim(id=id, step_id=1, text=f"x = {value}", kind="numeric", value=value)


def finding(id="1-1", **kw) -> CriticFinding:
    return CriticFinding(claim_id=id, **kw)


def one(claims, findings):
    return reconcile_claims(claims, findings, rel_tol=0.01)[0]


def test_float_within_relative_tolerance_verifies():
    assert numbers_match(100.0, 100.9, 0.01)
    assert not numbers_match(100.0, 102.0, 0.01)
    assert numbers_match(0.0, 0.0, 0.01)


def test_numeric_claim_verified_and_discrepant():
    assert one([claim_num(100.0)], [finding(value=100.5)]).status == "verified"
    v = one([claim_num(100.0)], [finding(value=90.0)])
    assert v.status == "discrepancy" and v.critic_value == 90.0


def test_integral_values_require_exact_match():
    assert one([claim_num(17)], [finding(value=17.0)]).status == "verified"
    assert one([claim_num(17)], [finding(value=18.0)]).status == "discrepancy"


def test_categorical_matching_is_case_insensitive():
    c = Claim(id="1-1", step_id=1, text="busiest day", kind="categorical", value="Saturday")
    assert one([c], [finding(value="  saturday ")]).status == "verified"
    assert one([c], [finding(value="Sunday")]).status == "discrepancy"


def stat_claim(direction="higher", significant=True) -> Claim:
    return Claim(
        id="1-1", step_id=1, text="weekend fares higher", kind="statistical",
        direction=direction, significant=significant,
    )


def test_statistical_direction_and_significance_must_match():
    ok = finding(direction="higher", significant=True, methodology_ok=True)
    assert one([stat_claim()], [ok]).status == "verified"
    assert one([stat_claim()], [finding(direction="lower", significant=True)]).status == "discrepancy"
    assert one([stat_claim()], [finding(direction="higher", significant=False)]).status == "discrepancy"


def test_bad_methodology_is_a_discrepancy_even_when_numbers_agree():
    v = one(
        [stat_claim()],
        [finding(direction="higher", significant=True, methodology_ok=False,
                 notes="t-test on heavily skewed data; Mann-Whitney was appropriate")],
    )
    assert v.status == "discrepancy" and "methodology" in v.reason.lower()


def test_unmatched_and_unverifiable_claims():
    assert one([claim_num(5.0)], []).status == "unverifiable"
    assert one([claim_num(5.0)], [finding(could_not_verify=True)]).status == "unverifiable"
```

- [ ] **Step 2: Run** → ImportError.
- [ ] **Step 3: Implement** `backend/app/runtime/reconcile.py`:

```python
"""Deterministic reconciliation of analyst claims against critic findings.

The tolerance policy lives HERE, in plain testable Python — the LLM critic only
derives values; it never decides whether a mismatch is within tolerance.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.runtime.state import Claim, Verdict


class CriticFinding(BaseModel):
    """The critic's independently derived counterpart of one claim."""

    claim_id: str
    could_not_verify: bool = False
    value: float | str | None = None
    direction: Literal["higher", "lower", "none"] | None = None
    significant: bool | None = None
    methodology_ok: bool | None = None  # statistical claims: was the method appropriate?
    notes: str = ""


def numbers_match(a: float, b: float, rel_tol: float) -> bool:
    """Relative-epsilon comparison; integral values must match exactly."""
    if float(a).is_integer() and float(b).is_integer():
        return a == b
    denom = max(abs(a), abs(b))
    if denom < 1e-9:
        return True
    return abs(a - b) / denom <= rel_tol


def _reconcile_one(claim: Claim, finding: CriticFinding, rel_tol: float) -> Verdict:
    if finding.could_not_verify:
        return Verdict(
            claim_id=claim.id, status="unverifiable",
            reason=finding.notes or "critic could not verify this claim",
        )

    if claim.kind == "statistical":
        if finding.methodology_ok is False:
            return Verdict(
                claim_id=claim.id, status="discrepancy", methodology_ok=False,
                reason=f"methodology: {finding.notes or 'critic judged the method inappropriate'}",
            )
        if claim.direction != finding.direction or claim.significant != finding.significant:
            return Verdict(
                claim_id=claim.id, status="discrepancy", methodology_ok=finding.methodology_ok,
                reason=(
                    f"critic found direction={finding.direction}, "
                    f"significant={finding.significant}; "
                    f"claimed direction={claim.direction}, significant={claim.significant}"
                ),
            )
        return Verdict(claim_id=claim.id, status="verified", methodology_ok=finding.methodology_ok)

    if isinstance(claim.value, (int, float)) and isinstance(finding.value, (int, float)):
        if numbers_match(float(claim.value), float(finding.value), rel_tol):
            return Verdict(claim_id=claim.id, status="verified", critic_value=finding.value)
        return Verdict(
            claim_id=claim.id, status="discrepancy", critic_value=finding.value,
            reason=f"claimed {claim.value}, critic derived {finding.value}",
        )

    if isinstance(claim.value, str) and isinstance(finding.value, str):
        if claim.value.strip().casefold() == finding.value.strip().casefold():
            return Verdict(claim_id=claim.id, status="verified", critic_value=finding.value)
        return Verdict(
            claim_id=claim.id, status="discrepancy", critic_value=finding.value,
            reason=f"claimed {claim.value!r}, critic derived {finding.value!r}",
        )

    return Verdict(
        claim_id=claim.id, status="unverifiable",
        reason="claim and critic values are not comparable",
    )


def reconcile_claims(
    claims: list[Claim], findings: list[CriticFinding], rel_tol: float
) -> list[Verdict]:
    by_id = {f.claim_id: f for f in findings}
    verdicts: list[Verdict] = []
    for claim in claims:
        finding = by_id.get(claim.id)
        if finding is None:
            verdicts.append(
                Verdict(claim_id=claim.id, status="unverifiable",
                        reason="critic did not address this claim")
            )
        else:
            verdicts.append(_reconcile_one(claim, finding, rel_tol))
    return verdicts
```

- [ ] **Step 4: Run:** `pytest tests/test_reconcile.py -q` → 8 passed.
- [ ] **Step 5: Commit:** `git commit -am "feat(m2): reconciliation tolerances + critic findings"`

---

### Task 5: Structured-output schemas + LLM deps v2 (usage + repair retry)

**Files:**
- Modify: `backend/app/agents/schemas.py`, `backend/app/agents/llm.py`
- Test: `backend/tests/test_llm.py`

**Interfaces:**
- Produces:
  - schemas: `AnalystTurn` (+ `claims: list[Claim]`), `PlannerTurn(steps: list[PlanStep], rationale: str)`, `CriticTurn(action, code, findings: list[CriticFinding])`.
  - llm: `LLMUsage(tokens_in, tokens_out)`, `MalformedOutputError(role, detail)`, `invoke_structured(structured_invoke, messages, role) -> tuple[BaseModel, LLMUsage]` (one repair retry), and `GraphDeps(planner, analyst_turn, critic_turn, compose)` where every callable returns `(output, LLMUsage)`.

- [ ] **Step 1: Failing test** (`backend/tests/test_llm.py`):

```python
"""Malformed structured output gets exactly one repair retry (plan §3.1)."""

import pytest

from app.agents.llm import LLMUsage, MalformedOutputError, invoke_structured
from app.agents.schemas import AnalystTurn


def raw(tokens=(10, 5)):
    class Raw:
        usage_metadata = {"input_tokens": tokens[0], "output_tokens": tokens[1]}
    return Raw()


def test_valid_output_passes_through_with_usage():
    out = {"raw": raw(), "parsed": AnalystTurn(action="finish", findings="ok"), "parsing_error": None}
    turn, usage = invoke_structured(lambda messages: out, [], "analyst")
    assert turn.findings == "ok"
    assert usage == LLMUsage(tokens_in=10, tokens_out=5)


def test_malformed_output_is_repaired_once():
    outputs = [
        {"raw": raw(), "parsed": None, "parsing_error": ValueError("bad json")},
        {"raw": raw(), "parsed": AnalystTurn(action="finish", findings="fixed"), "parsing_error": None},
    ]
    calls: list[list] = []

    def invoke(messages):
        calls.append(list(messages))
        return outputs[len(calls) - 1]

    turn, usage = invoke_structured(invoke, [], "analyst")
    assert turn.findings == "fixed"
    assert len(calls) == 2
    assert "failed validation" in calls[1][-1].content  # repair instruction appended
    assert usage == LLMUsage(tokens_in=20, tokens_out=10)  # both calls billed


def test_second_malformed_output_raises():
    bad = {"raw": raw(), "parsed": None, "parsing_error": ValueError("nope")}
    with pytest.raises(MalformedOutputError):
        invoke_structured(lambda messages: bad, [], "analyst")
```

- [ ] **Step 2: Run** → ImportError.
- [ ] **Step 3: Implement.**

`backend/app/agents/schemas.py` (full replacement):

```python
"""Structured-output schemas for agent turns."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.runtime.reconcile import CriticFinding
from app.runtime.state import Claim, PlanStep

__all__ = ["AnalystTurn", "PlannerTurn", "CriticTurn", "CriticFinding"]


class PlannerTurn(BaseModel):
    """The planner's structured plan: 1..N independent analysis steps."""

    steps: list[PlanStep] = Field(
        description="Independent analysis steps; they run in parallel and must not depend on each other."
    )
    rationale: str = Field(default="", description="One short paragraph: why this decomposition.")


class AnalystTurn(BaseModel):
    """One analyst decision: either run code against the dataset, or finish."""

    action: Literal["run_code", "finish"]
    code: str = Field(
        default="",
        description="Python to execute in the sandbox (required when action='run_code').",
    )
    findings: str = Field(
        default="",
        description="Findings summary with concrete numbers (required when action='finish').",
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="Atomic checkable claims backing the findings (required when action='finish').",
    )


class CriticTurn(BaseModel):
    """One critic decision: either run verification code, or finish with findings."""

    action: Literal["run_code", "finish"]
    code: str = Field(
        default="",
        description="Python to execute in the sandbox (required when action='run_code').",
    )
    findings: list[CriticFinding] = Field(
        default_factory=list,
        description="One finding per claim (required when action='finish').",
    )
```

`backend/app/agents/llm.py` (full replacement):

```python
"""LLM access, isolated so tests can inject stubs.

The graph never imports an OpenAI client directly; it receives callables via
`GraphDeps`, each returning `(output, LLMUsage)`. Tests provide plain Python
functions — the whole graph runs without an API key. `GraphDeps.default()`
wires the real models with structured output and one repair retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from app.agents.schemas import AnalystTurn, CriticTurn, PlannerTurn
from app.config import settings


@dataclass(frozen=True)
class LLMUsage:
    tokens_in: int = 0
    tokens_out: int = 0


class MalformedOutputError(Exception):
    def __init__(self, role: str, detail: str) -> None:
        self.role = role
        super().__init__(f"{role} produced malformed structured output after repair retry: {detail}")


def _usage_of(raw: object) -> LLMUsage:
    meta = getattr(raw, "usage_metadata", None) or {}
    return LLMUsage(tokens_in=meta.get("input_tokens", 0), tokens_out=meta.get("output_tokens", 0))


def invoke_structured(
    structured_invoke: Callable[[list[BaseMessage]], dict],
    messages: list[BaseMessage],
    role: str,
) -> tuple[BaseModel, LLMUsage]:
    """Invoke a `with_structured_output(include_raw=True)` runnable with ONE repair retry."""
    out = structured_invoke(messages)
    usage = _usage_of(out["raw"])
    if out["parsed"] is not None:
        return out["parsed"], usage

    repair = list(messages) + [
        HumanMessage(
            content=(
                f"Your previous response failed validation: {out['parsing_error']}. "
                "Respond again, matching the required schema exactly."
            )
        )
    ]
    out = structured_invoke(repair)
    second = _usage_of(out["raw"])
    usage = LLMUsage(usage.tokens_in + second.tokens_in, usage.tokens_out + second.tokens_out)
    if out["parsed"] is None:
        raise MalformedOutputError(role, str(out["parsing_error"]))
    return out["parsed"], usage


def _structured_fn(role: str, schema: type[BaseModel]):
    from langchain_openai import ChatOpenAI

    cfg = settings()
    model = ChatOpenAI(model=cfg.model_for(role), temperature=0, api_key=cfg.openai_api_key)
    runnable = model.with_structured_output(schema, include_raw=True)

    def call(messages: list[BaseMessage]):
        return invoke_structured(runnable.invoke, messages, role)

    return call


def _real_compose() -> "ComposeFn":
    from langchain_openai import ChatOpenAI

    cfg = settings()
    model = ChatOpenAI(model=cfg.model_for("composer"), temperature=0, api_key=cfg.openai_api_key)

    def compose(messages: list[BaseMessage]) -> tuple[str, LLMUsage]:
        response = model.invoke(messages)
        return response.content, _usage_of(response)

    return compose


PlannerFn = Callable[[list[BaseMessage]], tuple[PlannerTurn, LLMUsage]]
AnalystFn = Callable[[list[BaseMessage]], tuple[AnalystTurn, LLMUsage]]
CriticFn = Callable[[list[BaseMessage]], tuple[CriticTurn, LLMUsage]]
ComposeFn = Callable[[list[BaseMessage]], tuple[str, LLMUsage]]


@dataclass
class GraphDeps:
    """Injected model callables. Tests pass stubs; production uses `default()`."""

    planner: PlannerFn
    analyst_turn: AnalystFn
    critic_turn: CriticFn
    compose: ComposeFn

    @classmethod
    def default(cls) -> "GraphDeps":
        return cls(
            planner=_structured_fn("planner", PlannerTurn),
            analyst_turn=_structured_fn("analyst", AnalystTurn),
            critic_turn=_structured_fn("critic", CriticTurn),
            compose=_real_compose(),
        )
```

- [ ] **Step 4: Run:** `pytest tests/test_llm.py -q` → 3 passed (test_graph still red — rewritten next task).
- [ ] **Step 5: Commit:** `git commit -am "feat(m2): planner/critic schemas, LLM usage tracking, repair retry"`

---

### Task 6: The M2 graph — planner, Send fan-out, critic gate, bounded retry, composer

**Files:**
- Create: `backend/app/agents/prompts/planner.md`, `backend/app/agents/prompts/critic.md`
- Modify: `backend/app/agents/prompts/analyst.md`, `backend/app/agents/prompts/composer.md`
- Modify: `backend/app/runtime/graph.py`
- Test: `backend/tests/test_graph.py` (full rewrite)

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `build_graph(deps, checkpointer=None)`, `execute_run(state, deps=None) -> RunState` (same signature as M1; checkpointing wired in Task 7). Analyst node registered with `input_schema=AnalystTask`; `Send("analyst", task.model_dump())`.

- [ ] **Step 1: Failing tests** — full rewrite of `backend/tests/test_graph.py`:

```python
"""M2 graph routing with stubbed LLMs — the whole graph runs without an API key."""

from pathlib import Path

import pytest

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticFinding, CriticTurn, PlannerTurn
from app.runtime.graph import execute_run
from app.runtime.state import Claim, PlanStep, RunState

U = LLMUsage(tokens_in=10, tokens_out=5)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    csv = tmp_path / "d.csv"
    csv.write_text("fare,tip,day\n10,2,Sat\n20,5,Sun\n30,9,Mon\n")
    return csv


def make_state(dataset: Path, run_id: str) -> RunState:
    return RunState(
        run_id=run_id,
        question="What is the total fare and which day is busiest?",
        dataset_path=str(dataset),
        dataset_profile={"rows": 3, "columns": [{"name": "fare"}, {"name": "tip"}, {"name": "day"}]},
    )


def two_step_planner(messages):
    return (
        PlannerTurn(
            steps=[
                PlanStep(description="Compute the total fare", method="descriptive"),
                PlanStep(description="Find the busiest day", method="descriptive"),
            ]
        ),
        U,
    )


def finishing_analyst(findings_by_step: dict[int, tuple[str, list[Claim]]]):
    """Analyst stub that finishes immediately with per-step findings and claims."""

    def turn(messages):
        text = "\n".join(m.content for m in messages)
        for step_id, (findings, claims) in findings_by_step.items():
            if f"[step {step_id}]" in text:
                return AnalystTurn(action="finish", findings=findings, claims=claims), U
        raise AssertionError(f"no step marker in prompt: {text[-200:]}")

    return turn


def verifying_critic(messages):
    """Critic that confirms every claim it is shown."""
    import json as _json
    text = "\n".join(m.content for m in messages)
    payload = text[text.index("[") : text.rindex("]") + 1]
    claims = _json.loads(payload)
    return (
        CriticTurn(
            action="finish",
            findings=[CriticFinding(claim_id=c["id"], value=c["value"]) for c in claims],
        ),
        U,
    )


def test_planner_fans_out_and_verified_claims_compose(dataset):
    deps = GraphDeps(
        planner=two_step_planner,
        analyst_turn=finishing_analyst({
            1: ("Total fare is 60.", [Claim(text="total fare", kind="numeric", value=60.0)]),
            2: ("Busiest day is Sat.", [Claim(text="busiest day", kind="categorical", value="Sat")]),
        }),
        critic_turn=verifying_critic,
        compose=lambda m: ("Total fare is 60; Saturday is busiest.", U),
    )
    final = execute_run(make_state(dataset, "run-fanout"), deps)

    assert len(final.analyst_results) == 2
    assert {r.step_id for r in final.analyst_results} == {1, 2}
    assert all(v.status == "verified" for v in final.verdicts) and len(final.verdicts) == 2
    assert final.retry_count == 0
    assert final.final is not None
    assert all(vc.status == "verified" for vc in final.final.claims)


def test_discrepancy_triggers_one_bounded_retry_with_feedback(dataset):
    analyst_calls = []

    def analyst(messages):
        analyst_calls.append("\n".join(m.content for m in messages))
        return AnalystTurn(
            action="finish", findings="Total fare is 55.",
            claims=[Claim(text="total fare", kind="numeric", value=55.0)],
        ), U

    def critic(messages):
        return CriticTurn(
            action="finish",
            findings=[CriticFinding(claim_id="1-1", value=60.0)],
        ), U

    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=analyst,
        critic_turn=critic,
        compose=lambda m: ("Composed with caveats.", U),
    )
    final = execute_run(make_state(dataset, "run-retry"), deps)

    assert final.retry_count == 1
    assert len(analyst_calls) == 2                       # original + exactly one retry
    assert "critic" in analyst_calls[1].lower()          # feedback injected
    assert final.verdicts[0].status == "discrepancy"     # still wrong after retry
    assert final.final.claims[0].status == "unverified"  # shipped with explicit flag
    assert "60" in final.final.claims[0].detail


def test_planner_failure_routes_to_honest_composer(dataset):
    def bad_planner(messages):
        return PlannerTurn(steps=[]), U

    deps = GraphDeps(
        planner=bad_planner,
        analyst_turn=lambda m: (_ for _ in ()).throw(AssertionError("analyst must not run")),
        critic_turn=lambda m: (_ for _ in ()).throw(AssertionError("critic must not run")),
        compose=lambda m: ("I could not analyze this question.", U),
    )
    final = execute_run(make_state(dataset, "run-noplan"), deps)
    assert final.planner_failed
    assert final.final.failed
    assert "could not" in final.final_answer.lower()


def test_analyst_budget_exhaustion_is_a_typed_failure(dataset):
    big = LLMUsage(tokens_in=30_000, tokens_out=0)  # over max_tokens_per_agent in one call

    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=lambda m: (AnalystTurn(action="run_code", code="print(1)"), big),
        critic_turn=lambda m: (CriticTurn(action="finish", findings=[]), U),
        compose=lambda m: ("The analysis could not be completed: budget exhausted.", U),
    )
    final = execute_run(make_state(dataset, "run-budget"), deps)
    assert final.analyst_results[0].failed
    assert "budget" in final.analyst_results[0].failure_reason
    assert final.final.failed


def test_analyst_code_loop_still_works_end_to_end(dataset):
    calls = {"n": 0}

    def analyst(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return AnalystTurn(
                action="run_code",
                code="import pandas as pd\nprint('total', pd.read_csv('data.csv')['fare'].sum())",
            ), U
        assert "total 60" in messages[-1].content
        return AnalystTurn(
            action="finish", findings="Total fare is 60.",
            claims=[Claim(text="total fare", kind="numeric", value=60.0)],
        ), U

    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=analyst,
        critic_turn=verifying_critic,
        compose=lambda m: ("The total fare is 60.", U),
    )
    final = execute_run(make_state(dataset, "run-loop"), deps)
    assert final.final_answer == "The total fare is 60."
    assert final.analyst_results[0].iterations[0].result.exit_code == 0
    assert final.verdicts[0].status == "verified"


def test_events_flow_for_all_agents(dataset):
    from app.runtime.events import bus

    deps = GraphDeps(
        planner=two_step_planner,
        analyst_turn=finishing_analyst({
            1: ("A", [Claim(text="total fare", kind="numeric", value=60.0)]),
            2: ("B", [Claim(text="busiest day", kind="categorical", value="Sat")]),
        }),
        critic_turn=verifying_critic,
        compose=lambda m: ("done", U),
    )
    execute_run(make_state(dataset, "run-events"), deps)
    events = bus.history("run-events")
    agents = {e.agent for e in events}
    types = [e.type.value for e in events]
    assert {"system", "planner", "analyst", "critic", "composer"} <= agents
    assert types[0] == "run_started" and types[-1] == "run_finished"
    assert "verdict" in types and "handoff" in types
    assert any(e.tokens_in > 0 for e in events)  # usage flows onto events
```

- [ ] **Step 2: Run** → failures (old graph).
- [ ] **Step 3: Write the prompts.**

`backend/app/agents/prompts/planner.md`:

```markdown
You are the planner agent of tracelab, a data analysis system. Decompose the user's
question about a CSV dataset into the smallest set of INDEPENDENT analysis steps
(1 to {max_steps}). Steps run in parallel: no step may read another step's output.

For each step pick the method an analyst should use:

- `descriptive` — aggregations, group-bys, rankings, counts, distributions, derived metrics.
- `mean_comparison` — comparing a numeric quantity between exactly two groups
  ("is fare higher on weekends?"). Triggers a significance test with effect size.
- `correlation` — the relationship between two numeric columns. Triggers a
  correlation test with significance.

Rules:

- Prefer ONE step. Only split when the question genuinely contains multiple
  independent analyses.
- Use `mean_comparison`/`correlation` whenever the question implies difference,
  relationship, or significance — they trigger statistical rigor downstream.
- Each description must be self-contained (name the exact columns involved);
  the analyst executing it sees only that description, not the other steps.

Dataset profile (precomputed):

{profile}
```

`backend/app/agents/prompts/critic.md`:

```markdown
You are the critic agent of tracelab. Analysts have answered a question about a CSV
dataset and made specific claims. Your job is to verify every claim INDEPENDENTLY.
You have deliberately NOT been shown the analysts' code — write your own analysis
from scratch. Independent derivation is the whole point.

Rules:

- The dataset is at `./data.csv`. Libraries: pandas, numpy, scipy.
- Re-derive each claimed quantity with your own approach and print it to stdout.
  Only stdout comes back to you.
- For statistical claims, run the test YOU judge appropriate, then also review the
  claimed methodology: was the method right for the data types, sample size, and
  distribution? Are assumptions violated? Is the claim's strength proportionate to
  the effect size? Set `methodology_ok=false` (with notes) if the method was
  inappropriate — a statistically shaky answer is flagged exactly like a wrong number.
- Report the value/direction/significance YOU derived — never echo the claimed one.
  Tolerance is applied downstream by deterministic code, not by you.
- If a claim cannot be derived from this dataset, set `could_not_verify=true` with notes.
- You have at most {max_iterations} code executions. Verify all claims in as few
  scripts as possible — one combined script is ideal.

When done, finish with exactly one finding per claim (match `claim_id`).

Dataset profile (precomputed):

{profile}
```

`backend/app/agents/prompts/analyst.md` (full replacement):

```markdown
You are an analyst agent of tracelab, a data analysis system. You are assigned ONE
analysis step of a larger plan. You answer it by writing Python executed in a sandbox.

Your step: {step_description}
Method: {method}

Rules:

- The dataset is at `./data.csv`. Load it with pandas.
- Libraries available: pandas, numpy, scipy.
- Print every finding you rely on to stdout. Only stdout comes back to you.
- Never fabricate a number. If code fails, you will see stderr and may revise.
- Round presented floats sensibly, but compute at full precision.
- You have at most {max_iterations} code executions. One well-planned script beats
  three exploratory ones.

Method playbooks (follow the one for your assigned method):

- `descriptive`: compute the aggregation/ranking directly. State the exact numbers.
- `mean_comparison`: comparing a numeric column between two groups. Check group sizes
  and distribution shape (normality via `scipy.stats.shapiro` on samples ≤ 5000, else
  skewness; variance equality via `scipy.stats.levene`). Choose Welch's t-test
  (`scipy.stats.ttest_ind(equal_var=False)`) for roughly normal data, otherwise
  Mann-Whitney U. Report: n per group, both group means/medians, p-value, effect size
  (Cohen's d, or rank-biserial for Mann-Whitney), direction, and which assumptions you
  checked. Conclude significance at alpha = {alpha}.
- `correlation`: check linearity/outliers first (describe or quantiles). Report Pearson r
  with p-value (`scipy.stats.pearsonr`); if the relationship is monotonic but not linear
  or outlier-driven, use Spearman instead and say why. Effect size is the coefficient
  itself. Conclude significance at alpha = {alpha}.

Charts: if a chart genuinely helps answer the step, write a JSON file to
`./artifacts/chart_<name>.json` with EXACTLY this shape:

    {{"kind": "line|bar|scatter|pie|histogram", "title": "...",
      "x": "<field in data rows>", "y": ["<field in data rows>"],
      "data": [{{...}}, ...],  // ≤ 500 aggregated rows you computed
      "x_label": "...", "y_label": "...",
      "source_columns": ["<dataset columns this chart derives from>"]}}

Charts referencing nonexistent dataset columns are rejected like wrong numbers.

Finishing: when you have enough evidence, respond with `action="finish"`, a findings
summary, and a list of atomic `claims` — every number or category your findings rely
on, each independently checkable:

- numeric claim: kind="numeric", value=<the number>
- categorical claim: kind="categorical", value="<the label>"
- statistical claim: kind="statistical", direction ("higher"/"lower"/"none"),
  significant (true/false at alpha = {alpha}), and a full `methodology`
  (method, n, p_value, effect_size, effect_size_name, assumptions checked).

A separate critic will re-derive every claim from scratch; claims you cannot back
with executed code will be flagged.

Dataset profile (precomputed):

{profile}
```

`backend/app/agents/prompts/composer.md` (full replacement):

```markdown
You are the composer agent of tracelab. You turn verified findings into a final
answer for the user.

Rules:

- Use ONLY the findings and numbers provided. Never introduce new numbers.
- Claims arrive with a verification status. Present `verified` claims plainly.
  Present `unverified` claims ONLY with an explicit caveat stating what the critic
  found (e.g. "the analyst computed X, but this could not be confirmed — the critic
  derived Y"). Never present an unverified number as settled fact.
- Be direct and concrete: lead with the answer, then one short paragraph of context.
- For statistical findings, state the conclusion in plain language (test, p-value,
  effect size are shown separately in the UI — do not repeat raw statistics tables).
- If the analysis failed or is incomplete, say so plainly and state what could not
  be computed and why. An honest "could not determine X" is a correct answer.
- Plain prose. No headers, no bullet lists unless the user's question is itself a list.
```

- [ ] **Step 4: Implement `backend/app/runtime/graph.py`** (full replacement):

```python
"""The M2 graph: planner → parallel analysts (Send) → critic gate → composer.

Orchestration logic lives in the conditional edges:
  - after planner: fan out one analyst branch per independent plan step (Send API),
    or straight to composer on planner failure (honest failure path);
  - after critic: all verified → composer; discrepancy → ONE bounded retry of the
    disputed steps with the critic's findings injected; else composer ships the
    answer with explicit unverified flags.

The critic never sees analyst code — only the question, the dataset profile, and
the claims. Independent derivation is the point (see docs/architecture.md).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.llm import GraphDeps, LLMUsage, MalformedOutputError
from app.config import settings
from app.runtime.budget import AgentBudget, BudgetExceeded
from app.runtime.chartspec import ChartSpec, extract_chart_specs
from app.runtime.events import AgentEvent, EventType, bus
from app.runtime.reconcile import reconcile_claims
from app.runtime.state import (
    AnalysisStep,
    AnalystResult,
    AnalystTask,
    Claim,
    FinalAnswer,
    Plan,
    RunState,
    VerifiedClaim,
)
from app.sandbox.executor import run_code

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agents" / "prompts"


def _prompt(name: str, **kwargs: object) -> str:
    text = (PROMPTS_DIR / f"{name}.md").read_text()
    return text.format(**kwargs) if kwargs else text


def _emit(
    run_id: str,
    agent: str,
    type_: EventType,
    payload: dict,
    started: float,
    usage: LLMUsage | None = None,
) -> None:
    bus.emit(
        AgentEvent(
            run_id=run_id,
            agent=agent,
            type=type_,
            payload=payload,
            tokens_in=usage.tokens_in if usage else 0,
            tokens_out=usage.tokens_out if usage else 0,
            started_at=started,
            duration_ms=int((time.time() - started) * 1000),
        )
    )


def _latest_results(results: list[AnalystResult]) -> dict[int, AnalystResult]:
    """Last result per step wins — a retried step's result supersedes the original."""
    latest: dict[int, AnalystResult] = {}
    for r in results:
        latest[r.step_id] = r
    return dict(sorted(latest.items()))


# ── planner ──────────────────────────────────────────────────────────────────


def planner_node(state: RunState, deps: GraphDeps) -> dict:
    cfg = settings()
    budget = AgentBudget.for_role("planner")
    t0 = time.time()
    try:
        turn, usage = deps.planner(
            [
                SystemMessage(
                    content=_prompt(
                        "planner", max_steps=cfg.max_plan_steps, profile=state.dataset_profile
                    )
                ),
                HumanMessage(content=state.question),
            ]
        )
        budget.spend_llm(usage.tokens_in, usage.tokens_out)
    except (BudgetExceeded, MalformedOutputError) as exc:
        _emit(state.run_id, "planner", EventType.ERROR, {"error": str(exc)}, t0)
        return {"planner_failed": True, "planner_failure_reason": str(exc)}

    steps = turn.steps[: cfg.max_plan_steps]
    for i, step in enumerate(steps, start=1):
        step.id = i
    _emit(
        state.run_id,
        "planner",
        EventType.LLM_CALL,
        {"plan": [s.model_dump() for s in steps], "rationale": turn.rationale},
        t0,
        usage,
    )
    if not steps:
        return {"planner_failed": True, "planner_failure_reason": "planner produced an empty plan"}
    _emit(
        state.run_id,
        "planner",
        EventType.HANDOFF,
        {"to": "analyst", "steps": [s.id for s in steps]},
        time.time(),
    )
    return {"plan": Plan(steps=steps, rationale=turn.rationale)}


def fan_out(state: RunState):
    if state.planner_failed or state.plan is None or not state.plan.steps:
        return "composer"
    return [
        Send(
            "analyst",
            AnalystTask(
                run_id=state.run_id,
                question=state.question,
                dataset_path=state.dataset_path,
                dataset_profile=state.dataset_profile,
                step=step,
            ).model_dump(),
        )
        for step in state.plan.steps
    ]


# ── analyst (Send target; also the retry target) ─────────────────────────────


def _assign_claim_ids(step_id: int, claims: list[Claim]) -> list[Claim]:
    for i, claim in enumerate(claims, start=1):
        claim.id = f"{step_id}-{i}"
        claim.step_id = step_id
    return claims


def analyst_node(task: AnalystTask, deps: GraphDeps) -> dict:
    cfg = settings()
    budget = AgentBudget.for_role("analyst")
    step = task.step
    columns = [c.get("name", "") for c in task.dataset_profile.get("columns", [])]

    messages: list = [
        SystemMessage(
            content=_prompt(
                "analyst",
                max_iterations=cfg.max_analyst_iterations,
                profile=task.dataset_profile,
                step_description=step.description,
                method=step.method,
                alpha=cfg.alpha,
            )
        ),
        HumanMessage(content=f"Overall question: {task.question}\n[step {step.id}] {step.description}"),
    ]
    if task.critic_feedback:
        messages.append(
            HumanMessage(
                content=(
                    "A previous attempt at this step was disputed by the critic:\n"
                    f"{task.critic_feedback}\n"
                    "Re-derive the result carefully and resolve the discrepancy."
                )
            )
        )

    iterations: list[AnalysisStep] = []
    chart_specs: list[ChartSpec] = []
    chart_rejections: list[str] = []

    def failure(reason: str) -> dict:
        return {
            "analyst_results": [
                AnalystResult(
                    step_id=step.id, iterations=iterations, failed=True, failure_reason=reason
                )
            ]
        }

    for iteration in range(1, cfg.max_analyst_iterations + 1):
        t0 = time.time()
        try:
            turn, usage = deps.analyst_turn(messages)
            budget.spend_llm(usage.tokens_in, usage.tokens_out)
        except (BudgetExceeded, MalformedOutputError) as exc:
            _emit(task.run_id, "analyst", EventType.ERROR, {"step_id": step.id, "error": str(exc)}, t0)
            return failure(str(exc))
        _emit(
            task.run_id,
            "analyst",
            EventType.LLM_CALL,
            {"step_id": step.id, "iteration": iteration, "action": turn.action},
            t0,
            usage,
        )

        if turn.action == "finish":
            return {
                "analyst_results": [
                    AnalystResult(
                        step_id=step.id,
                        findings=turn.findings,
                        claims=_assign_claim_ids(step.id, turn.claims),
                        chart_specs=chart_specs,
                        chart_rejections=chart_rejections,
                        iterations=iterations,
                    )
                ]
            }

        try:
            budget.spend_tool()
        except BudgetExceeded as exc:
            return failure(str(exc))
        t0 = time.time()
        result = run_code(turn.code, task.dataset_path)
        iterations.append(AnalysisStep(iteration=iteration, result=result))
        specs, rejections = extract_chart_specs(result.artifacts, columns)
        chart_specs.extend(specs)
        chart_rejections.extend(rejections)
        _emit(
            task.run_id,
            "analyst",
            EventType.TOOL_CALL,
            {
                "step_id": step.id,
                "iteration": iteration,
                "code": turn.code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "charts_accepted": len(specs),
                "charts_rejected": rejections,
            },
            t0,
        )
        feedback = (
            f"Execution result (exit={result.exit_code}, timed_out={result.timed_out})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        if rejections:
            feedback += "\nRejected charts:\n" + "\n".join(rejections)
        messages.append(HumanMessage(content=feedback))

    return failure(
        f"analyst exhausted {cfg.max_analyst_iterations} iterations without findings"
    )


# ── critic ───────────────────────────────────────────────────────────────────


def critic_node(state: RunState, deps: GraphDeps) -> dict:
    cfg = settings()
    latest = _latest_results(state.analyst_results)
    claims = [c for r in latest.values() if not r.failed for c in r.claims]
    if not claims:
        return {"verdicts": [], "retry_steps": []}

    budget = AgentBudget.for_role("critic")
    claims_json = json.dumps([c.model_dump() for c in claims], indent=2, default=str)
    messages: list = [
        SystemMessage(
            content=_prompt(
                "critic", max_iterations=cfg.max_critic_iterations, profile=state.dataset_profile
            )
        ),
        HumanMessage(content=f"Question: {state.question}\n\nClaims to verify:\n{claims_json}"),
    ]

    findings = []
    for iteration in range(1, cfg.max_critic_iterations + 2):
        t0 = time.time()
        try:
            turn, usage = deps.critic_turn(messages)
            budget.spend_llm(usage.tokens_in, usage.tokens_out)
        except (BudgetExceeded, MalformedOutputError) as exc:
            _emit(state.run_id, "critic", EventType.ERROR, {"error": str(exc)}, t0)
            break  # claims fall through as unverifiable
        _emit(
            state.run_id,
            "critic",
            EventType.LLM_CALL,
            {"iteration": iteration, "action": turn.action},
            t0,
            usage,
        )
        if turn.action == "finish":
            findings = turn.findings
            break
        try:
            budget.spend_tool()
        except BudgetExceeded:
            break
        t0 = time.time()
        result = run_code(turn.code, state.dataset_path)
        _emit(
            state.run_id,
            "critic",
            EventType.TOOL_CALL,
            {
                "iteration": iteration,
                "code": turn.code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
            },
            t0,
        )
        messages.append(
            HumanMessage(
                content=(
                    f"Execution result (exit={result.exit_code}, timed_out={result.timed_out})\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
            )
        )

    verdicts = reconcile_claims(claims, findings, cfg.numeric_rel_tolerance)
    for verdict in verdicts:
        _emit(state.run_id, "critic", EventType.VERDICT, verdict.model_dump(), time.time())

    claim_step = {c.id: c.step_id for c in claims}
    disputed = sorted({claim_step[v.claim_id] for v in verdicts if v.status == "discrepancy"})
    if disputed and state.retry_count < cfg.max_retries:
        _emit(
            state.run_id,
            "critic",
            EventType.HANDOFF,
            {"to": "analyst", "retry_steps": disputed},
            time.time(),
        )
        return {
            "verdicts": verdicts,
            "retry_count": state.retry_count + 1,
            "retry_steps": disputed,
        }
    return {"verdicts": verdicts, "retry_steps": []}


def route_after_critic(state: RunState):
    if not state.retry_steps or state.plan is None:
        return "composer"
    step_map = {s.id: s for s in state.plan.steps}
    claim_step = {c.id: c.step_id for r in state.analyst_results for c in r.claims}
    feedback: dict[int, list[str]] = {sid: [] for sid in state.retry_steps}
    for v in state.verdicts:
        sid = claim_step.get(v.claim_id)
        if sid in feedback and v.status == "discrepancy":
            feedback[sid].append(f"- {v.claim_id}: {v.reason}")
    return [
        Send(
            "analyst",
            AnalystTask(
                run_id=state.run_id,
                question=state.question,
                dataset_path=state.dataset_path,
                dataset_profile=state.dataset_profile,
                step=step_map[sid],
                critic_feedback="\n".join(feedback[sid]) or "the critic disputed this step",
            ).model_dump(),
        )
        for sid in state.retry_steps
        if sid in step_map
    ]


# ── composer ─────────────────────────────────────────────────────────────────


def composer_node(state: RunState, deps: GraphDeps) -> dict:
    t0 = time.time()
    latest = _latest_results(state.analyst_results)
    verdict_map = {v.claim_id: v for v in state.verdicts}

    verified_claims: list[VerifiedClaim] = []
    for result in latest.values():
        for claim in result.claims:
            verdict = verdict_map.get(claim.id)
            if verdict is not None and verdict.status == "verified":
                verified_claims.append(VerifiedClaim(claim=claim, status="verified"))
            else:
                detail = verdict.reason if verdict else "not checked by the critic"
                verified_claims.append(
                    VerifiedClaim(claim=claim, status="unverified", detail=detail)
                )

    charts = [spec for r in latest.values() for spec in r.chart_specs]
    all_failed = bool(latest) and all(r.failed for r in latest.values())
    failed = state.planner_failed or not latest or all_failed

    if state.planner_failed:
        context = f"Planning FAILED: {state.planner_failure_reason}. Compose an honest failure answer."
    else:
        parts: list[str] = []
        for result in latest.values():
            if result.failed:
                parts.append(f"Step {result.step_id} FAILED: {result.failure_reason}")
            else:
                parts.append(f"Step {result.step_id} findings:\n{result.findings}")
        parts.append("Claims with verification status:")
        for vc in verified_claims:
            line = f"- [{vc.status}] {vc.claim.text}"
            if vc.claim.value is not None:
                line += f" = {vc.claim.value}"
            if vc.detail:
                line += f" ({vc.detail})"
            parts.append(line)
        context = "\n\n".join(parts)

    answer, usage = deps.compose(
        [
            SystemMessage(content=_prompt("composer")),
            HumanMessage(content=f"Question: {state.question}\n\n{context}"),
        ]
    )
    _emit(state.run_id, "composer", EventType.LLM_CALL, {"answer": answer}, t0, usage)
    final = FinalAnswer(narrative=answer, claims=verified_claims, charts=charts, failed=failed)
    return {"final_answer": answer, "final": final}


# ── graph assembly ───────────────────────────────────────────────────────────


def build_graph(deps: GraphDeps, checkpointer=None):
    g = StateGraph(RunState)
    g.add_node("planner", lambda s: planner_node(s, deps))
    g.add_node("analyst", lambda t: analyst_node(t, deps), input_schema=AnalystTask)
    g.add_node("critic", lambda s: critic_node(s, deps))
    g.add_node("composer", lambda s: composer_node(s, deps))
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out, ["analyst", "composer"])
    g.add_edge("analyst", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["analyst", "composer"])
    g.add_edge("composer", END)
    return g.compile(checkpointer=checkpointer)


def execute_run(state: RunState, deps: GraphDeps | None = None) -> RunState:
    """Run the graph for one question, emitting lifecycle events."""
    deps = deps or GraphDeps.default()
    t0 = time.time()
    bus.emit(
        AgentEvent(
            run_id=state.run_id,
            agent="system",
            type=EventType.RUN_STARTED,
            payload={"question": state.question},
        )
    )
    try:
        final = build_graph(deps).invoke(state)
        final_state = RunState.model_validate(final)
        _emit(
            final_state.run_id,
            "system",
            EventType.RUN_FINISHED,
            {
                "answer": final_state.final_answer,
                "final": final_state.final.model_dump() if final_state.final else None,
            },
            t0,
        )
        return final_state
    except Exception as exc:  # surface, don't swallow — the UI shows honest errors
        bus.emit(
            AgentEvent(
                run_id=state.run_id,
                agent="system",
                type=EventType.ERROR,
                payload={"error": str(exc)},
            )
        )
        raise
```

- [ ] **Step 5: Run:** `pytest -q` → all green (test_llm, test_budget, test_reconcile, test_chartspec, test_graph, test_events, test_sandbox, test_store). Also `ruff check .`.
- [ ] **Step 6: Commit:** `git commit -am "feat(m2): full graph — planner, Send fan-out, critic gate, bounded retry"`

---

### Task 7: Checkpointing (SqliteSaver) + new dependencies

**Files:**
- Modify: `backend/pyproject.toml` (add `langgraph-checkpoint-sqlite>=2.0`, `scipy>=1.13`)
- Modify: `backend/app/runtime/graph.py` (`execute_run` opens a saver)
- Test: `backend/tests/test_checkpoint.py`

**Interfaces:**
- Produces: `execute_run` checkpoints every superstep under `thread_id=run_id` into `settings().checkpoints_db_path`; `build_graph(deps, checkpointer=...)` already accepts a saver (Task 6).

- [ ] **Step 1: Install deps:** add to `pyproject.toml` dependencies:

```toml
    "langgraph-checkpoint-sqlite>=2.0",
    "scipy>=1.13",
```

Run: `.venv/bin/pip install -e ".[dev]"` (from `backend/`).

- [ ] **Step 2: Failing test** (`backend/tests/test_checkpoint.py`):

```python
"""Checkpointing: every superstep persists; a run's state is recoverable by thread_id."""

from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticFinding, CriticTurn, PlannerTurn
from app.runtime.graph import build_graph
from app.runtime.state import Claim, PlanStep, RunState

U = LLMUsage(tokens_in=1, tokens_out=1)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    csv = tmp_path / "d.csv"
    csv.write_text("fare\n10\n20\n30\n")
    return csv


def test_run_state_is_checkpointed_per_superstep(dataset, tmp_path):
    deps = GraphDeps(
        planner=lambda m: (PlannerTurn(steps=[PlanStep(description="total fare")]), U),
        analyst_turn=lambda m: (
            AnalystTurn(
                action="finish", findings="Total is 60.",
                claims=[Claim(text="total", kind="numeric", value=60.0)],
            ),
            U,
        ),
        critic_turn=lambda m: (
            CriticTurn(action="finish", findings=[CriticFinding(claim_id="1-1", value=60.0)]),
            U,
        ),
        compose=lambda m: ("Total is 60.", U),
    )
    with SqliteSaver.from_conn_string(str(tmp_path / "ckpt.sqlite3")) as saver:
        graph = build_graph(deps, checkpointer=saver)
        config = {"configurable": {"thread_id": "run-ckpt"}}
        graph.invoke(
            RunState(
                run_id="run-ckpt", question="total?", dataset_path=str(dataset),
                dataset_profile={"columns": [{"name": "fare"}]},
            ),
            config,
        )
        checkpoints = list(saver.list(config))
        assert len(checkpoints) >= 4  # input + planner + analyst + critic + composer supersteps
        latest = graph.get_state(config)
        assert latest.values["final_answer"] == "Total is 60."
        assert latest.next == ()  # run completed; nothing pending
```

- [ ] **Step 3: Run** → fails (only if `execute_run` change pending — this test targets `build_graph` directly, so it may pass immediately; the red step here is the import failing before `pip install`). Then wire `execute_run`:

In `graph.py`, replace the body of `execute_run`'s try block:

```python
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(settings().checkpoints_db_path), check_same_thread=False)
        try:
            graph = build_graph(deps, checkpointer=SqliteSaver(conn))
            final = graph.invoke(state, {"configurable": {"thread_id": state.run_id}})
        finally:
            conn.close()
        final_state = RunState.model_validate(final)
        ...
```

(imports stay local so unit tests that call `build_graph` directly never touch the checkpoint DB; module import remains light).

- [ ] **Step 4: Run:** `pytest tests/test_checkpoint.py -q` → 1 passed; full suite green.
- [ ] **Step 5: Commit:** `git commit -am "feat(m2): SqliteSaver checkpointing per run"`

---

### Task 8: Persist + serve the structured result

**Files:**
- Modify: `backend/app/tracing/store.py`, `backend/app/api/runs.py`
- Test: `backend/tests/test_store.py` (add cases; keep existing)

**Interfaces:**
- Produces: `runs.result` TEXT column (JSON of `FinalAnswer`), migration for existing DBs; `Store.finish_run(run_id, answer, status="finished", result="")`; `GET /api/runs/{id}` returns `result` as parsed JSON (or `None`).

- [ ] **Step 1: Failing test** — append to `backend/tests/test_store.py`:

```python
def test_finish_run_persists_structured_result(tmp_path):
    store = Store(tmp_path / "t.sqlite3")
    ds = store.add_dataset("d", "/tmp/d.csv", {"rows": 1})
    run_id = store.create_run(ds, "q")
    store.finish_run(run_id, "answer", "finished", result='{"narrative": "answer", "failed": false}')
    run = store.get_run(run_id)
    assert run["result"] == '{"narrative": "answer", "failed": false}'


def test_existing_database_is_migrated_with_result_column(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, question TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running', answer TEXT DEFAULT '',
            created_at REAL NOT NULL
        );
        INSERT INTO runs VALUES ('r1', 'd1', 'q', 'finished', 'a', 0);
        """
    )
    conn.commit()
    conn.close()

    store = Store(db)  # must migrate, not crash
    assert store.get_run("r1")["result"] == ""
```

- [ ] **Step 2: Run** → fails.
- [ ] **Step 3: Implement.** In `store.py`: add `result TEXT NOT NULL DEFAULT ''` to the `runs` CREATE TABLE, and after `executescript(_SCHEMA)` in `__init__`:

```python
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Additive migrations for databases created before a column existed."""
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
        if "result" not in columns:
            conn.execute("ALTER TABLE runs ADD COLUMN result TEXT NOT NULL DEFAULT ''")
```

and:

```python
    def finish_run(
        self, run_id: str, answer: str, status: str = "finished", result: str = ""
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, answer = ?, result = ? WHERE id = ?",
                (status, answer, result, run_id),
            )
```

In `api/runs.py` `_execute`:

```python
    try:
        final = execute_run(state)
        result = final.final.model_dump_json() if final.final else ""
        store().finish_run(run_id, final.final_answer, "finished", result)
    except Exception as exc:
        store().finish_run(run_id, f"error: {exc}", "error")
```

and in `get_run`:

```python
@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    run = store().get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    run["result"] = json.loads(run["result"]) if run.get("result") else None
    return {**run, "spans": store().spans_for_run(run_id)}
```

(add `import json` at top of `api/runs.py`).

- [ ] **Step 4: Run:** `pytest -q` green; `ruff check .` clean.
- [ ] **Step 5: Commit:** `git commit -am "feat(m2): persist structured FinalAnswer on runs"`

---

### Task 9: Frontend — types, Zod ChartSpec, renderer, badges, chips, answer panel

**Files:**
- Modify: `frontend/package.json` (`npm install zod`)
- Create: `frontend/src/lib/chartSpec.ts`
- Modify: `frontend/src/lib/types.ts`
- Create: `frontend/src/components/ChartSpecRenderer.tsx`, `frontend/src/components/ClaimBadge.tsx`, `frontend/src/components/MethodologyChip.tsx`, `frontend/src/components/AnswerPanel.tsx`

**Interfaces:**
- Consumes: `FinalAnswer` JSON shape from Task 6 (`{narrative, claims: [{claim, status, detail}], charts, failed}`).
- Produces: `<AnswerPanel final={FinalAnswer} />` used by Workbench (Task 10); `chartSpecSchema` Zod mirror.

- [ ] **Step 1:** `cd frontend && npm install zod`

- [ ] **Step 2:** `frontend/src/lib/chartSpec.ts`:

```ts
import { z } from "zod";

/** Zod mirror of the backend ChartSpec (backend/app/runtime/chartspec.py). */
export const chartSpecSchema = z.object({
  kind: z.enum(["line", "bar", "scatter", "pie", "histogram"]),
  title: z.string().default(""),
  x: z.string(),
  y: z.array(z.string()).min(1),
  data: z.array(z.record(z.string(), z.unknown())).min(1).max(500),
  x_label: z.string().default(""),
  y_label: z.string().default(""),
  source_columns: z.array(z.string()).default([]),
});

export type ChartSpec = z.infer<typeof chartSpecSchema>;
```

- [ ] **Step 3:** extend `frontend/src/lib/types.ts` (append):

```ts
export interface Methodology {
  method: string;
  n: number;
  p_value: number | null;
  effect_size: number | null;
  effect_size_name: string;
  assumptions: string[];
}

export interface Claim {
  id: string;
  step_id: number;
  text: string;
  kind: "numeric" | "categorical" | "statistical";
  value: number | string | null;
  direction: "higher" | "lower" | "none" | null;
  significant: boolean | null;
  methodology: Methodology | null;
}

export interface VerifiedClaim {
  claim: Claim;
  status: "verified" | "unverified";
  detail: string;
}

export interface FinalAnswer {
  narrative: string;
  claims: VerifiedClaim[];
  charts: unknown[]; // validated at render time by the Zod chartSpecSchema
  failed: boolean;
}
```

- [ ] **Step 4:** `frontend/src/components/ClaimBadge.tsx`:

```tsx
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { Chip, Tooltip } from "@mui/material";
import type { VerifiedClaim } from "../lib/types";

/** verified (critic reconciled it) or unverified (with the discrepancy shown). */
export function ClaimBadge({ vc }: { vc: VerifiedClaim }) {
  const verified = vc.status === "verified";
  const chip = (
    <Chip
      size="small"
      variant="outlined"
      color={verified ? "success" : "warning"}
      icon={verified ? <CheckCircleOutlineIcon /> : <WarningAmberIcon />}
      label={vc.status}
    />
  );
  return verified ? chip : <Tooltip title={vc.detail || "unverified"}>{chip}</Tooltip>;
}
```

`frontend/src/components/MethodologyChip.tsx`:

```tsx
import ScienceOutlinedIcon from "@mui/icons-material/ScienceOutlined";
import { Chip, Tooltip } from "@mui/material";
import type { Methodology } from "../lib/types";

const fmtP = (p: number) => (p < 0.001 ? "p<0.001" : `p=${p.toFixed(3)}`);

/** test used, n, p-value, effect size — the rigor receipt on a statistical claim. */
export function MethodologyChip({ m }: { m: Methodology }) {
  const parts = [m.method, `n=${m.n}`];
  if (m.p_value !== null) parts.push(fmtP(m.p_value));
  if (m.effect_size !== null)
    parts.push(`${m.effect_size_name || "effect"}=${m.effect_size.toFixed(2)}`);
  const chip = (
    <Chip size="small" variant="outlined" icon={<ScienceOutlinedIcon />} label={parts.join(" · ")} />
  );
  return m.assumptions.length > 0 ? (
    <Tooltip title={`Assumptions checked: ${m.assumptions.join("; ")}`}>{chip}</Tooltip>
  ) : (
    chip
  );
}
```

`frontend/src/components/ChartSpecRenderer.tsx`:

```tsx
import { Paper, Typography } from "@mui/material";
import { BarChart } from "@mui/x-charts/BarChart";
import { LineChart } from "@mui/x-charts/LineChart";
import { PieChart } from "@mui/x-charts/PieChart";
import { ScatterChart } from "@mui/x-charts/ScatterChart";
import { chartSpecSchema } from "../lib/chartSpec";

const HEIGHT = 300;

/** Renders a validated ChartSpec with MUI X Charts. Invalid specs render nothing. */
export function ChartSpecRenderer({ spec: raw }: { spec: unknown }) {
  const parsed = chartSpecSchema.safeParse(raw);
  if (!parsed.success) return null;
  const spec = parsed.data;
  const data = spec.data as Record<string, number | string>[];
  const series = spec.y.map((key) => ({ dataKey: key, label: key }));

  let chart: React.ReactNode;
  switch (spec.kind) {
    case "bar":
    case "histogram":
      chart = (
        <BarChart
          dataset={data}
          xAxis={[{ scaleType: "band", dataKey: spec.x, label: spec.x_label || undefined }]}
          yAxis={[{ label: spec.y_label || undefined }]}
          series={series}
          height={HEIGHT}
        />
      );
      break;
    case "line":
      chart = (
        <LineChart
          dataset={data}
          xAxis={[{ scaleType: "point", dataKey: spec.x, label: spec.x_label || undefined }]}
          yAxis={[{ label: spec.y_label || undefined }]}
          series={series}
          height={HEIGHT}
        />
      );
      break;
    case "scatter":
      chart = (
        <ScatterChart
          series={[
            {
              label: spec.y[0],
              data: data.map((row, i) => ({
                id: i,
                x: Number(row[spec.x]),
                y: Number(row[spec.y[0]]),
              })),
            },
          ]}
          xAxis={[{ label: spec.x_label || spec.x }]}
          yAxis={[{ label: spec.y_label || spec.y[0] }]}
          height={HEIGHT}
        />
      );
      break;
    case "pie":
      chart = (
        <PieChart
          series={[
            {
              data: data.map((row, i) => ({
                id: i,
                label: String(row[spec.x]),
                value: Number(row[spec.y[0]]),
              })),
            },
          ]}
          height={HEIGHT}
        />
      );
      break;
  }

  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      {spec.title && (
        <Typography variant="subtitle2" gutterBottom>
          {spec.title}
        </Typography>
      )}
      {chart}
    </Paper>
  );
}
```

`frontend/src/components/AnswerPanel.tsx`:

```tsx
import { Alert, Paper, Stack, Typography } from "@mui/material";
import type { FinalAnswer } from "../lib/types";
import { ChartSpecRenderer } from "./ChartSpecRenderer";
import { ClaimBadge } from "./ClaimBadge";
import { MethodologyChip } from "./MethodologyChip";

/** The composed answer: narrative, per-claim verification badges, and charts. */
export function AnswerPanel({ final }: { final: FinalAnswer }) {
  return (
    <Stack spacing={2}>
      <Alert severity={final.failed ? "warning" : "success"}>{final.narrative}</Alert>

      {final.claims.length > 0 && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Claims
          </Typography>
          <Stack spacing={1}>
            {final.claims.map((vc) => (
              <Stack
                key={vc.claim.id}
                direction="row"
                spacing={1}
                alignItems="center"
                flexWrap="wrap"
                useFlexGap
              >
                <Typography variant="body2" sx={{ flex: "1 1 240px" }}>
                  {vc.claim.text}
                  {vc.claim.value !== null && ` — ${vc.claim.value}`}
                </Typography>
                {vc.claim.methodology && <MethodologyChip m={vc.claim.methodology} />}
                <ClaimBadge vc={vc} />
              </Stack>
            ))}
          </Stack>
        </Paper>
      )}

      {final.charts.map((spec, i) => (
        <ChartSpecRenderer key={i} spec={spec} />
      ))}
    </Stack>
  );
}
```

- [ ] **Step 5: Verify:** `npm run typecheck` → clean.
- [ ] **Step 6: Commit:** `git commit -am "feat(m2): claim badges, methodology chips, ChartSpec renderer"`

---

### Task 10: Frontend wiring — store, Workbench, EventLog

**Files:**
- Modify: `frontend/src/store/runStore.ts`, `frontend/src/pages/Workbench.tsx`, `frontend/src/components/EventLog.tsx`

- [ ] **Step 1:** `runStore.ts` — add `final` to the store:

```ts
import type { AgentEvent, FinalAnswer } from "../lib/types";
```

state interface gains `final: FinalAnswer | null;`, initial value `final: null`, `start` resets it (`final: null` in the set), and in `ingest`:

```ts
      if (event.type === "run_finished") {
        next.status = "finished";
        next.answer = String(event.payload.answer ?? "");
        next.final = (event.payload.final ?? null) as FinalAnswer | null;
      }
```

- [ ] **Step 2:** `Workbench.tsx` — pull `final` from the store and render the panel:

```ts
const { runId, status, events, answer, error, final, start } = useRunStore();
```

replace the finished line with:

```tsx
{status === "finished" &&
  (final ? <AnswerPanel final={final} /> : <Alert severity="success">{answer}</Alert>)}
```

adding `import { AnswerPanel } from "../components/AnswerPanel";`.

- [ ] **Step 3:** `EventLog.tsx` — teach `summary()` the new events (replace the `llm_call` case and add `verdict`/`handoff`):

```ts
    case "llm_call":
      if (e.payload.plan !== undefined)
        return `planned ${(e.payload.plan as unknown[]).length} step(s)`;
      if (e.payload.answer !== undefined) return "composed final answer";
      return e.payload.action === "run_code"
        ? `${step(e)}iteration ${e.payload.iteration}: decided to run code`
        : `${step(e)}iteration ${e.payload.iteration}: finished analysis`;
    case "tool_call":
      return `${step(e)}sandbox exec (exit=${e.payload.exit_code}${e.payload.timed_out ? ", TIMED OUT" : ""})`;
    case "verdict":
      return `${e.payload.status} — claim ${e.payload.claim_id}${
        e.payload.reason ? ` (${e.payload.reason})` : ""
      }`;
    case "handoff":
      return e.payload.retry_steps !== undefined
        ? `retry → step(s) ${(e.payload.retry_steps as number[]).join(", ")}`
        : `fan-out → ${(e.payload.steps as number[]).length} analyst(s)`;
```

with helper above `summary`:

```ts
const step = (e: AgentEvent) =>
  e.payload.step_id !== undefined ? `[step ${e.payload.step_id}] ` : "";
```

- [ ] **Step 4: Verify:** `npm run typecheck && npm run build` → clean.
- [ ] **Step 5: Commit:** `git commit -am "feat(m2): wire structured answers + verdict events into the UI"`

---

### Task 11: Finalize — full verification + milestone bookkeeping

- [ ] **Step 1:** Backend: `cd backend && .venv/bin/pytest -q && .venv/bin/ruff check .` → all green.
- [ ] **Step 2:** Frontend: `cd frontend && npm run typecheck && npm run build` → clean.
- [ ] **Step 3:** Tick every M2 box in `MILESTONES.md` (planner, Send fan-out, critic, conditional edges, badges, budgets, checkpointing, stats methods, ChartSpec).
- [ ] **Step 4:** Commit: `git commit -am "chore: tick M2 milestones"`.

**Out of scope (explicitly deferred):** M2 exit-criteria *live* demo runs (needs an OpenAI key — also still open on the M1 checklist); span `parent_span_id` tree structure, cost meter (M3); remaining stats toolkit (M3/M4); sample datasets + suggested questions (M4 golden sets).
