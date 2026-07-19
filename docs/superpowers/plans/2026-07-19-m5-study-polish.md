# M5 — Tradeoff Study + Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the golden set across 3-4 model configs, publish the quality vs cost vs latency chart + conclusions, finalize README/docs, draft the blog post, and hand the owner a recording/publishing checklist.

**Architecture:** Builds directly on M4's harness: a `StudyConfig` (per-role model map, judge pinned to `gpt-4o` for comparability) threads through `GraphDeps.default(models=...)` into `_structured_fn`. The `study` CLI runs `run_eval` once per config; `report` renders markdown tables + a matplotlib PNG for the README; the Evals screen gets a TradeoffChart over the same eval-run rows.

**Tech Stack:** As M4, plus `matplotlib` (backend dev/report use only).

## Global Constraints

- Depends on the completed M4 plan (`2026-07-19-m4-evals-adaptive-routing.md`) — same branch family; work on `m5-study-polish` created from the M4 branch tip.
- No pushes; commits stay local. `MILESTONES.md` stays untracked.
- All pytest stays keyless. Live study runs are explicitly gated: **mini-everywhere config may run without asking (~$1 with judge); any config using `gpt-4o` for agents requires the owner's go-ahead first** (estimate: $4-8 per strong config sweep, $10-20 for the full study).
- The judge model is pinned per-config-file to `gpt-4o` — never varied with the config under test, or the quality axis measures the judge, not the agents. `CHEAP_MODE` must be OFF for study runs (explicit models bypass it — verify, don't assume).
- README/docs numbers come from `python -m app.evals report` output — never hand-typed.

---

### Task 1: Study configs + per-role model threading

**Files:**
- Create: `backend/configs/mini.yaml`, `backend/configs/strong-planner.yaml`, `backend/configs/strong-critic.yaml`, `backend/configs/strong.yaml`
- Create: `backend/app/evals/study.py` (config loading; the runner comes in Task 2)
- Modify: `backend/app/agents/llm.py` (`GraphDeps.default(models=None)`, `_structured_fn(role, schema, model=None)`, compose fn likewise)
- Test: `backend/tests/test_study.py`

**Interfaces:**
- Config file shape:

```yaml
# backend/configs/mini.yaml
name: mini
description: gpt-4o-mini everywhere — the cost floor
models:
  router: gpt-4o-mini
  planner: gpt-4o-mini
  analyst: gpt-4o-mini
  critic: gpt-4o-mini
  composer: gpt-4o-mini
judge_model: gpt-4o
```

`strong-planner.yaml`: planner `gpt-4o`, rest mini ("strong brain, cheap hands"). `strong-critic.yaml`: critic `gpt-4o`, rest mini ("cheap work, strong verification"). `strong.yaml`: all five roles `gpt-4o`. All four keep `judge_model: gpt-4o`.
- Produces: `StudyConfig(name: str, description: str, models: dict[str, str], judge_model: str)` (Pydantic), `load_study_config(path: Path) -> StudyConfig`, `CONFIGS_DIR = REPO_ROOT / "backend" / "configs"`.
- `GraphDeps.default(models: dict[str, str] | None = None)` — when `models` is given, each `_structured_fn(role, schema, model=models[role])` uses that exact model id, bypassing `cheap_mode`'s collapse (`model or cfg.model_for(role)` inside `_structured_fn`; same for the compose ChatOpenAI and the router). M4's harness already accepts `models=` and records it in `config_json`; now also pass it: `run_eval(..., models=...)` must call `GraphDeps.default` with it when the caller passes the default factory — implement by giving the harness an explicit `deps_factory=lambda: GraphDeps.default(models=cfg.models)` at the call site in Task 2 (no harness change needed).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_study.py
from pathlib import Path

from app.evals.study import CONFIGS_DIR, load_study_config


def test_all_four_configs_load():
    configs = [load_study_config(p) for p in sorted(CONFIGS_DIR.glob("*.yaml"))]
    names = {c.name for c in configs}
    assert names == {"mini", "strong-planner", "strong-critic", "strong"}
    for c in configs:
        assert set(c.models) == {"router", "planner", "analyst", "critic", "composer"}
        assert c.judge_model == "gpt-4o"  # judge never varies with the config under test
    strong = next(c for c in configs if c.name == "strong")
    mini = next(c for c in configs if c.name == "mini")
    assert strong.models["analyst"] == "gpt-4o"
    assert mini.models["analyst"] == "gpt-4o-mini"


def test_default_deps_accept_model_override(monkeypatch):
    """GraphDeps.default(models=...) must not raise and must not read cheap_mode."""
    from app.agents.llm import GraphDeps
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    from app.config import settings
    settings.cache_clear()
    deps = GraphDeps.default(models={"router": "gpt-4o-mini", "planner": "gpt-4o",
                                    "analyst": "gpt-4o-mini", "critic": "gpt-4o",
                                    "composer": "gpt-4o-mini"})
    assert deps.planner is not None and deps.router is not None
    settings.cache_clear()
```

(Constructing `ChatOpenAI` with a fake key doesn't call the network; if `GraphDeps.default` eagerly validates the key beyond construction, relax the test to assert `load_study_config` only and note it.)

- [ ] **Step 2: Run to verify it fails** → `cd backend && .venv/bin/pytest tests/test_study.py -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# backend/app/evals/study.py
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
```

`llm.py` changes: `_structured_fn(role, schema, model: str | None = None)` → `ChatOpenAI(model=model or cfg.model_for(role), ...)`; `GraphDeps.default(cls, models=None)` threads `models.get(role)` into every `_structured_fn` call, the compose model, and the router. Write the four YAML config files.

- [ ] **Step 4: Run tests** → `cd backend && .venv/bin/pytest tests/test_study.py -v && .venv/bin/pytest` → PASS.

- [ ] **Step 5: Commit**

```bash
git checkout -b m5-study-polish
git add backend/configs backend/app/evals/study.py backend/app/agents/llm.py backend/tests/test_study.py
git commit -m "feat(m5): study configs + per-role model override through GraphDeps"
```

---

### Task 2: `study` CLI subcommand

**Files:**
- Modify: `backend/app/evals/study.py` (add `run_study`)
- Modify: `backend/app/evals/__main__.py` (add `study` subcommand)
- Test: extend `backend/tests/test_study.py`

**Interfaces:**
- Produces: `run_study(st: Store, configs: list[StudyConfig], deps_factory_for: Callable[[StudyConfig], Callable[[], GraphDeps]], judge_for: Callable[[StudyConfig], JudgeFn | None], *, repo_root: Path = REPO_ROOT, enforce_budget: bool = True) -> list[tuple[str, str]]` — `(config_name, eval_run_id)` pairs; each eval run labeled `study:<config name>` with `models` recorded. The factory indirection keeps tests keyless.
- CLI: `python -m app.evals study [--configs mini,strong] [--no-judge]` — resolves names to `backend/configs/<name>.yaml`, defaults to all four, judge on by default (pinned per config's `judge_model`; build it with the same `_structured_fn("judge", JudgeTurn, model=cfg.judge_model)` override).

- [ ] **Step 1: Write the failing test** (append to `test_study.py`; reuse `_deps`/`_golden` helpers from `tests/test_harness.py` — import them or copy the two helpers into a shared `tests/helpers.py` if importing test modules is awkward)

```python
def test_run_study_records_one_eval_run_per_config(tmp_path):
    from app.evals.study import StudyConfig, run_study
    from app.tracing.store import Store
    from tests.test_harness import _deps, _golden  # or tests.helpers if extracted

    st = Store(tmp_path / "t.db")
    configs = [
        StudyConfig(name="a", models={r: "m" for r in
                                      ("router", "planner", "analyst", "critic", "composer")}),
        StudyConfig(name="b", models={r: "m" for r in
                                      ("router", "planner", "analyst", "critic", "composer")}),
    ]
    golden = _golden(tmp_path, 3)
    pairs = run_study(st, configs, lambda cfg: (lambda: _deps(3)), lambda cfg: None,
                      repo_root=tmp_path, enforce_budget=False, golden_sets=golden)
    assert [name for name, _ in pairs] == ["a", "b"]
    labels = {r["label"] for r in st.list_eval_runs()}
    assert labels == {"study:a", "study:b"}
```

(Adjust `run_study`'s signature to take `golden_sets` explicitly — cleaner than a hidden global; update the Interfaces block accordingly: `run_study(st, configs, deps_factory_for, judge_for, *, golden_sets, repo_root=REPO_ROOT, enforce_budget=True)`.)

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement**

```python
# append to backend/app/evals/study.py
from typing import Callable

from app.agents.llm import GraphDeps
from app.evals.golden import GoldenDataset
from app.evals.harness import run_eval
from app.evals.judge import JudgeFn
from app.tracing.store import Store


def run_study(st: Store, configs: list[StudyConfig],
              deps_factory_for: Callable[[StudyConfig], Callable[[], GraphDeps]],
              judge_for: Callable[[StudyConfig], JudgeFn | None], *,
              golden_sets: list[GoldenDataset],
              repo_root: Path = REPO_ROOT, enforce_budget: bool = True
              ) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for cfg in configs:
        eval_id = run_eval(st, golden_sets, deps_factory_for(cfg),
                           judge=judge_for(cfg), label=f"study:{cfg.name}",
                           models=cfg.models, repo_root=repo_root,
                           enforce_budget=enforce_budget)
        pairs.append((cfg.name, eval_id))
        print(f"[study] {cfg.name}: eval {eval_id}")
    return pairs
```

CLI wiring in `__main__.py`:

```python
    study = sub.add_parser("study", help="run the golden set across model configs")
    study.add_argument("--configs", default="mini,strong-planner,strong-critic,strong")
    study.add_argument("--no-judge", action="store_true")
```

```python
    if args.cmd == "study":
        from app.agents.llm import GraphDeps, _structured_fn
        from app.agents.schemas import JudgeTurn
        from app.deps import store
        from app.evals.study import CONFIGS_DIR, load_study_config, run_study

        configs = [load_study_config(CONFIGS_DIR / f"{n.strip()}.yaml")
                   for n in args.configs.split(",")]
        run_study(
            store(), configs,
            deps_factory_for=lambda cfg: (lambda: GraphDeps.default(models=cfg.models)),
            judge_for=lambda cfg: None if args.no_judge
            else _structured_fn("judge", JudgeTurn, model=cfg.judge_model),
            golden_sets=load_golden(GOLDEN_DIR))
        return
```

- [ ] **Step 4: Run tests** → suite green; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add backend/app/evals backend/tests/test_study.py
git commit -m "feat(m5): study runner — golden sweep per config with pinned judge"
```

---

### Task 3: Report generator — markdown tables + tradeoff PNG

**Files:**
- Create: `backend/app/evals/report.py`
- Modify: `backend/app/evals/__main__.py` (add `report` subcommand)
- Modify: `backend/pyproject.toml` (add `matplotlib>=3.9`)
- Test: `backend/tests/test_report.py`

**Interfaces:**
- Produces:
  - `study_rows(st: Store) -> list[dict]` — for each eval run labeled `study:*` (latest per config name): `{"config", "tier1_pass_rate", "judge_avg", "cost_per_question", "latency_s_per_question", "cost_usd", "eval_run_id"}`.
  - `study_markdown(rows) -> str` — a GitHub table (Config | Tier-1 | Judge | $/question | s/question).
  - `calibration_markdown(report: dict) -> str` — the README calibration table from M4's `calibration_report` shape.
  - `tradeoff_png(rows, out_path: Path) -> None` — matplotlib scatter: x = cost/question (log if spread >10×), y = judge_avg (fallback tier1%), point size ∝ latency, each point annotated with config name; dark-friendly styling is NOT needed (README renders on white).
  - CLI: `python -m app.evals report [--png docs/assets/tradeoff.png]` — prints both markdown blocks, writes the PNG when `--png` given.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_report.py
from app.evals.report import study_markdown, study_rows, tradeoff_png
from app.tracing.store import Store


def _seed(st: Store) -> None:
    rows = [("study:mini", 100.0, 0.30, 4, 0.9, 3.6), ("study:mini", 400.0, 0.35, 4, 0.9, 3.7),
            ("study:strong", 200.0, 4.20, 4, 1.0, 4.5)]
    for i, (label, ts, cost, total, rate, judge) in enumerate(rows):
        st.add_eval_run(id=f"ev{i}", created_at=ts, label=label, git_sha="s",
                        config_hash="h", config_json="{}", questions_total=total,
                        tier1_scorable=total, tier1_passed=int(total * rate),
                        judge_avg=judge, cost_usd=cost, duration_ms=total * 8000)


def test_study_rows_latest_per_config(tmp_path):
    st = Store(tmp_path / "t.db")
    _seed(st)
    rows = study_rows(st)
    assert {r["config"] for r in rows} == {"mini", "strong"}
    mini = next(r for r in rows if r["config"] == "mini")
    assert mini["eval_run_id"] == "ev1"  # the newer of the two mini runs
    assert mini["cost_per_question"] == round(0.35 / 4, 4)
    assert mini["latency_s_per_question"] == 8.0


def test_markdown_and_png(tmp_path):
    st = Store(tmp_path / "t.db")
    _seed(st)
    rows = study_rows(st)
    md = study_markdown(rows)
    assert "| Config |" in md and "mini" in md and "strong" in md
    out = tmp_path / "tradeoff.png"
    tradeoff_png(rows, out)
    assert out.stat().st_size > 1000
```

- [ ] **Step 2: Run to verify it fails** → FAIL (after `pip install -e ".[dev]"` with matplotlib added).

- [ ] **Step 3: Implement**

```python
# backend/app/evals/report.py
"""Study + calibration artifacts for the README: markdown tables and the tradeoff chart."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.tracing.store import Store


def study_rows(st: Store) -> list[dict]:
    latest: dict[str, dict] = {}
    for r in st.list_eval_runs():  # newest first
        if not r["label"].startswith("study:"):
            continue
        latest.setdefault(r["label"].removeprefix("study:"), r)
    out = []
    for config, r in sorted(latest.items()):
        n = max(r["questions_total"], 1)
        out.append({
            "config": config, "eval_run_id": r["id"],
            "tier1_pass_rate": (r["tier1_passed"] / r["tier1_scorable"]
                                if r["tier1_scorable"] else None),
            "judge_avg": r["judge_avg"],
            "cost_per_question": round(r["cost_usd"] / n, 4),
            "latency_s_per_question": round(r["duration_ms"] / n / 1000, 1),
            "cost_usd": r["cost_usd"],
        })
    return out


def study_markdown(rows: list[dict]) -> str:
    lines = ["| Config | Tier-1 pass | Judge avg (1-5) | $/question | s/question |",
             "|---|---|---|---|---|"]
    for r in rows:
        t1 = "—" if r["tier1_pass_rate"] is None else f"{r['tier1_pass_rate']:.0%}"
        judge = "—" if r["judge_avg"] is None else f"{r['judge_avg']:.2f}"
        lines.append(f"| {r['config']} | {t1} | {judge} "
                     f"| ${r['cost_per_question']:.4f} | {r['latency_s_per_question']} |")
    return "\n".join(lines)


def calibration_markdown(report: dict) -> str:
    if not report.get("available"):
        return "_No calibration labels yet._"
    lines = ["| Dimension | n | Exact % | Within-1 % | Cohen's κ |", "|---|---|---|---|---|"]
    for d in report["dimensions"]:
        lines.append(f"| {d['dimension'].replace('_', ' ')} | {d['n']} "
                     f"| {d['exact_pct']} | {d['within1_pct']} | {d['kappa']} |")
    o = report["overall"]
    lines.append(f"| **overall (pooled)** | {o['n']} | {o['exact_pct']} "
                 f"| {o['within1_pct']} | {o['kappa']} |")
    return "\n".join(lines)


def tradeoff_png(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    xs = [r["cost_per_question"] for r in rows]
    ys = [r["judge_avg"] if r["judge_avg"] is not None
          else (r["tier1_pass_rate"] or 0) * 5 for r in rows]
    sizes = [40 + 12 * r["latency_s_per_question"] for r in rows]
    ax.scatter(xs, ys, s=sizes, alpha=0.75)
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(r["config"], (x, y), xytext=(6, 6), textcoords="offset points")
    if max(xs) / max(min(xs), 1e-9) > 10:
        ax.set_xscale("log")
    ax.set_xlabel("cost per question (USD)")
    ax.set_ylabel("quality (judge avg, 1-5)")
    ax.set_title("Quality vs cost vs latency (point size = latency)")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
```

CLI wiring: `report` subparser with `--png` (default empty); handler prints `## Tradeoff study`, `study_markdown(...)`, `## Judge calibration`, `calibration_markdown(calibration_report(...) if labels else {})`, writes PNG when `--png` set.

- [ ] **Step 4: Run tests** → suite green; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add backend/app/evals backend/pyproject.toml backend/tests/test_report.py
git commit -m "feat(m5): study report — markdown tables + tradeoff PNG"
```

---

### Task 4: TradeoffChart on the Evals screen

**Files:**
- Create: `frontend/src/components/TradeoffChart.tsx`
- Modify: `frontend/src/pages/EvalsScreen.tsx` (render it above "Score over time")

**Interfaces:**
- Consumes: `EvalRunSummary[]` already fetched by the page (`listEvalRuns`). No new endpoint: filter `label.startsWith("study:")`, keep the newest per config label.
- Produces: `TradeoffChart({ runs }: { runs: EvalRunSummary[] })` — MUI X `ScatterChart` following `ChartSpecRenderer`'s exact scatter pattern (`{id, x, y}` points); x = cost/question, y = judge_avg (fallback tier-1 % × 5); one series per config so the legend names configs. Hidden entirely (`return null`) when no study runs exist.

- [ ] **Step 1: Implement**

```tsx
// frontend/src/components/TradeoffChart.tsx
import { Paper, Typography } from "@mui/material";
import { ScatterChart } from "@mui/x-charts/ScatterChart";
import type { EvalRunSummary } from "../lib/types";

export function TradeoffChart({ runs }: { runs: EvalRunSummary[] }) {
  const latest = new Map<string, EvalRunSummary>();
  for (const run of [...runs].sort((a, b) => b.created_at - a.created_at)) {
    if (!run.label.startsWith("study:")) continue;
    const config = run.label.slice("study:".length);
    if (!latest.has(config)) latest.set(config, run);
  }
  if (latest.size === 0) return null;

  const series = [...latest.entries()].map(([config, run]) => {
    const perQ = run.cost_usd / Math.max(run.questions_total, 1);
    const quality = run.judge_avg ?? (run.tier1_pass_rate ?? 0) * 5;
    return { label: config, data: [{ id: run.id, x: perQ, y: quality }] };
  });

  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Quality vs cost (latest study run per config)
      </Typography>
      <ScatterChart
        height={300}
        series={series}
        xAxis={[{ label: "cost per question (USD)" }]}
        yAxis={[{ label: "quality (judge avg 1-5)", min: 1, max: 5 }]}
      />
    </Paper>
  );
}
```

In `EvalsScreen.tsx`: `<TradeoffChart runs={runs.data ?? []} />` as the first card in the Stack.

- [ ] **Step 2: Verify + commit** → `cd frontend && npm run typecheck && npm run build` → clean.

```bash
git add frontend/src
git commit -m "feat(m5): tradeoff scatter on the Evals screen"
```

---

### Task 5: Run the study (live spend — GATED)

**Files:** none (data + `docs/assets/tradeoff.png` output)

- [ ] **Step 1: Mini config first (no gate, ~$1 with judge)**

```bash
cd backend && .venv/bin/python -m app.evals study --configs mini
```

Sanity-check the eval run on the Evals screen (tier-1 rate should be within a few points of the M4 baseline — routing landed before it, so material drops are a routing regression, not noise).

- [ ] **Step 2: OWNER GATE — strong configs.** STOP and get an explicit go-ahead before running: `--configs strong-planner,strong-critic,strong` with judge ≈ **$10-20 total** (33 questions × 3 configs, gpt-4o roles + gpt-4o judge). Note: the daily budget cap (`daily_budget_usd = 2.0`) will abort mid-sweep — raise `DAILY_BUDGET_USD` in `backend/.env` for the study session (e.g. 25) and restore it after; say so in the go-ahead request.

- [ ] **Step 3: After approval and runs**

```bash
cd backend && .venv/bin/python -m app.evals report --png ../docs/assets/tradeoff.png
```

Save the printed markdown (it feeds Task 6). Commit the PNG:

```bash
git add docs/assets/tradeoff.png
git commit -m "feat(m5): tradeoff study chart from live multi-config runs"
```

---

### Task 6: README + docs finalization

**Files:**
- Modify: `README.md` (full rewrite to the build-plan §6 skeleton)
- Create: `docs/architecture.md`, `docs/evals.md`

Follow the skeleton in `tracelab-plan.md` §6 exactly (Why / How it works / The orchestration / Verification / Evals / Tradeoffs & limits / Run locally). Non-negotiables, all sourced from the codebase and the plan's stated theses:

- Top: one-line pitch, `[demo GIF placeholder — see owner checklist]`, links to the video/blog placeholders, and the one honest no-deploy sentence (build plan §3, "Deployment" paragraph — paraphrase it, keep the judgment-call framing).
- "The orchestration": rendered graph diagram (ASCII from the build plan §3.1 updated with the router at entry), the typed-state/conditional-edges/Send/checkpointing walkthrough, a "what LangGraph does under the hood" subsection (Pregel-style supersteps, channels, checkpoint semantics), the AgentEvent model, and the routing line: "the orchestration scales with the question; most questions don't need orchestration."
- "Verification": critic independence (never sees analyst code), tolerance policy from `reconcile.py`, methodology gate + chip UI.
- "Evals": methodology, golden-set self-verification via derivations, two-tier scoring, **the calibration table** (paste `calibration_markdown` output — requires the owner's labels; if still unlabeled, insert the table placeholder with the exact regeneration command and flag it in the handoff), **the tradeoff table + `docs/assets/tradeoff.png` + 3 written conclusions**. Conclusions must be drawn from the actual numbers (e.g. does strong-planner beat mini on quality-per-dollar? does strong-everywhere justify its cost?) — write them AFTER Task 5, not from priors.
- "Tradeoffs & limits": sandbox isolation level and the gVisor/Firecracker production path, in-process orchestration vs durable queues, anti-goals list from build-plan §1, what-I'd-build-next.
- `docs/architecture.md`: expanded diagrams + design decisions (event model as load-bearing decision, replay layers, router). `docs/evals.md`: full methodology + calibration + study details beyond README depth.

- [ ] Write all three documents; verify every command in "Run locally" actually works (`make dev`, `make test`); commit:

```bash
git add README.md docs/architecture.md docs/evals.md
git commit -m "docs(m5): final README, architecture and evals deep-dives"
```

---

### Task 7: Blog post draft

**Files:**
- Create: `docs/blog-draft-calibrated-judge.md`

Angle (from the build plan): **"I calibrated my LLM judge"** — the strongest differentiator. Outline: (1) everyone ships LLM-as-judge, almost nobody checks the judge; (2) the rubric and why 4 dimensions; (3) hand-labeling 40 answers — what disagreements revealed; (4) the numbers: agreement/within-1/kappa table, per-dimension; (5) iterating the judge prompt against κ; (6) what the tradeoff study could then say *because* the quality axis was trusted; (7) takeaway: a judge you haven't calibrated is a random number generator with good vibes. Pull real numbers from `report` output; owner's voice, first person, publishable on blog.ucaronur.com after their edit pass. ~1200-1600 words.

Commit: `git add docs/blog-draft-calibrated-judge.md && git commit -m "docs(m5): blog draft — I calibrated my LLM judge"`.

---

### Task 8: Owner handoff checklist (recording + publishing — human-only)

Deliver at the end as the final message, not a file commit:

1. **Hand-label calibration** (if not done in M4) → `python -m app.evals calibration`, then paste `report`'s calibration markdown into README §Evals and docs/evals.md.
2. **Strong-config study go-ahead** (Task 5 gate) if not yet approved/run.
3. **Record the demo GIF + 60-second video** from build-plan §2's storyboard: sample dataset → suggested question → live graph (show a simple question folding vs a statistical question fanning out — the M4 money shot) → click critic node → methodology chip → Runs cost. Insert GIF at README top.
4. **Publish the blog post** on blog.ucaronur.com after an edit pass; link it from the README.
5. **Portfolio grid**: add tracelab to the portfolio site (`/Users/onurucar/Developer/portfolio` — follow its Daylight/Ember design-system conventions; can be delegated to an agent in that repo with the README as source material).
6. **Merge/push decision**: `m3-observability` → `m4-evals-routing` → `m5-study-polish` are stacked locally; merging to master and pushing is the owner's call (never pushed by agents).
7. Check off the remaining M5 boxes in `MILESTONES.md` as each lands.

## Self-review notes (already applied)

- Spec coverage: study across 3-4 configs (Tasks 1-2, 5), quality/cost/latency chart + conclusions (3-6), README finalized w/ diagram + tradeoffs + calibration + study chart (6), GIF/video (8, owner), blog (7-8), portfolio grid (8). 
- The judge-pinning constraint (judge model constant across configs) is the study's validity linchpin — it appears in Global Constraints, Task 1's test, and Task 2's CLI.
- Numbers discipline: every README/blog number traces to `python -m app.evals report`; both docs tasks explicitly depend on Task 5's live data.
