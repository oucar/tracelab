"""Run the golden set through the graph, score both tiers, persist to the store."""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import pandas as pd

from app.agents.llm import GraphDeps
from app.api.datasets import profile_dataframe
from app.config import settings
from app.evals.golden import REPO_ROOT, GoldenDataset
from app.evals.judge import DIMENSIONS, JudgeFn, judge_answer
from app.evals.scoring import score_tier1
from app.evals.study import ROLES
from app.runtime.events import bus
from app.runtime.graph import execute_run
from app.runtime.state import RunState
from app.tracing import pricing
from app.tracing.store import Store, utc_midnight


def _ensure_sink(st: Store) -> None:
    """Wire the bus to `st`; EventBus.add_sink dedup handles the rest.

    EventBus.add_sink checks if the bound method is already registered (`if sink
    not in self._sinks`), so re-registering the same store's sink is safe.
    Notably, stale sinks from earlier Stores may still receive spans (writes to
    old test DBs), but correctness only requires the CURRENT store to persist
    them; orphan writes are harmless.
    """
    bus.add_sink(st.add_span)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


#: Bumped whenever the shape of the snapshot changes. Without it, hashes from
#: two different snapshot schemas are indistinguishable strings, so an old run
#: silently looks reproducible under new code when it isn't.
CONFIG_SCHEMA_VERSION = 2

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "agents" / "prompts"


def prompt_digests(prompts_dir: Path | None = None) -> dict[str, str]:
    """Content hash per agent prompt file.

    The prompts are the single biggest lever on answer quality in this system,
    and until they were part of the recorded config a pass-rate move could not
    be attributed: swapping a model and rewording `analyst.md` produced the same
    `config_hash`, so regression tracking could tell you *that* quality changed
    but never *what changed*. Hashing per file rather than over the whole
    directory means a diff of two snapshots names the prompt that moved.

    `prompts_dir` resolves at call time rather than as a default argument so
    that pointing `PROMPTS_DIR` elsewhere actually takes effect.
    """
    return {
        path.stem: sha256(path.read_bytes()).hexdigest()[:12]
        for path in sorted((prompts_dir or PROMPTS_DIR).glob("*.md"))
    }


def effective_judge_model(judge_model: str | None) -> str:
    """The model the judge will ACTUALLY use, not the one configured for it.

    `Settings.model_for("judge")` collapses to `analyst_model` under
    `cheap_mode`, so recording `cfg.judge_model` verbatim can claim a gpt-4o
    judge on a run that was in fact judged by gpt-4o-mini — the config under
    test grading its own homework, which is exactly what pinning the judge
    exists to prevent. An explicit `judge_model` (the study path) is honoured
    as-is; otherwise record what resolution actually returns.
    """
    return judge_model or settings().model_for("judge")


def config_snapshot(
    models: dict[str, str] | None,
    judge_model: str | None = None,
    judge_ran: bool = True,
) -> tuple[str, str]:
    """Snapshot the config that actually drove a run, for `config_hash` regression tracking.

    When `models` is passed it is the source of truth (exactly what `deps_factory`
    was built with) and is recorded verbatim — never re-read from `settings()`,
    which could silently diverge from what the run actually used. `judge_model`
    is recorded alongside since it varies independently of the five agent roles
    (`ROLES` = router/planner/analyst/critic/composer) and must also be part of
    the hash so two configs differing only in judge model don't collide.

    `judge_ran=False` records `judge_model: null`. A tier-1-only sweep has no
    judge at all, and naming one implies a tier-2 number that was never
    produced — the same class of mistake as recording a judge model the run
    did not actually use.
    """
    cfg = settings()
    snap = {
        "version": CONFIG_SCHEMA_VERSION,
        "models": models if models is not None else {r: cfg.model_for(r) for r in ROLES},
        "judge_model": effective_judge_model(judge_model) if judge_ran else None,
        "prompts": prompt_digests(),
        "alpha": cfg.alpha,
        "numeric_rel_tolerance": cfg.numeric_rel_tolerance,
        "cheap_mode": cfg.cheap_mode,
    }
    blob = json.dumps(snap, sort_keys=True)
    return blob, sha256(blob.encode()).hexdigest()[:12]


def run_eval(
    st: Store,
    golden_sets: list[GoldenDataset],
    deps_factory: Callable[[], GraphDeps],
    *,
    judge: JudgeFn | None = None,
    label: str = "",
    models: dict[str, str] | None = None,
    judge_model: str | None = None,
    repo_root: Path = REPO_ROOT,
    enforce_budget: bool = True,
) -> str:
    """Sweep every question in `golden_sets` through the graph and record a regression row.

    One question crashing (a graph exception) records a failed, tier-1-scored
    result and the sweep continues — a single bad question must never take
    down the whole eval run.
    """
    _ensure_sink(st)
    eval_run_id = uuid.uuid4().hex[:12]
    config_json, config_hash = config_snapshot(models, judge_model, judge_ran=judge is not None)
    t_eval = time.time()
    scorable = passed = 0
    judge_totals: list[float] = []
    total_cost = 0.0

    for gs in golden_sets:
        csv_path = repo_root / gs.csv
        df = pd.read_csv(csv_path)
        profile = profile_dataframe(df)
        existing = [d for d in st.list_datasets() if d["name"] == gs.name]
        dataset_id = (
            existing[0]["id"] if existing else st.add_dataset(gs.name, str(csv_path), profile)
        )

        for q in gs.questions:
            if enforce_budget and st.cost_since(utc_midnight()) >= settings().daily_budget_usd:
                raise RuntimeError("daily budget exhausted — aborting eval run")
            run_id = st.create_run(dataset_id, q.question)
            state = RunState(
                run_id=run_id, question=q.question,
                dataset_path=str(csv_path), dataset_profile=profile,
            )
            t0 = time.time()
            final = None
            crash_detail = ""
            headroom = (
                max(settings().daily_budget_usd - st.cost_since(utc_midnight()), 0.0)
                if enforce_budget
                else None
            )
            try:
                out = execute_run(state, deps_factory(), daily_headroom_usd=headroom)
                final = out.final
                result = final.model_dump_json() if final else ""
                st.finish_run(run_id, out.final_answer, "finished", result)
            except Exception as exc:  # noqa: BLE001  # one bad question must not kill the sweep
                st.finish_run(run_id, f"error: {exc}", "error")
                crash_detail = f"run crashed: {exc} — "
            duration_ms = int((time.time() - t0) * 1000)
            cost = sum(s["cost_usd"] for s in st.spans_for_run(run_id))

            tier1 = score_tier1(q.expected, final)
            if crash_detail:
                tier1.detail = crash_detail + tier1.detail
            scorable += int(tier1.scorable)
            passed += int(tier1.scorable and tier1.passed)

            judge_json = rationale = None
            if judge is not None and final is not None:
                turn, judge_usage = judge_answer(q.question, final, judge)
                # The judge is an LLM call like any other, but it runs outside
                # the graph so it emits no span — which meant its cost was
                # invisible to `spans_for_run` above and every $/question figure
                # excluded it. With the judge pinned to a strong model it is
                # frequently the largest single line item in a study run.
                cost += pricing.cost_usd(
                    judge_usage.model, judge_usage.tokens_in, judge_usage.tokens_out
                )
                judge_json = turn.model_dump_json()
                rationale = turn.rationale
                judge_totals.append(sum(getattr(turn, d) for d in DIMENSIONS) / len(DIMENSIONS))

            total_cost += cost  # after the judge, so the eval total includes it

            st.add_eval_result(
                eval_run_id=eval_run_id, question_id=q.id, run_id=run_id,
                dataset=gs.name, tags_json=json.dumps(q.tags),
                tier1_scorable=tier1.scorable, tier1_passed=tier1.passed,
                tier1_detail=tier1.detail, judge_json=judge_json,
                judge_rationale=rationale or "", cost_usd=cost, duration_ms=duration_ms,
            )

    st.add_eval_run(
        id=eval_run_id, created_at=t_eval, label=label, git_sha=git_sha(),
        config_hash=config_hash, config_json=config_json,
        questions_total=sum(len(g.questions) for g in golden_sets),
        tier1_scorable=scorable, tier1_passed=passed,
        judge_avg=(sum(judge_totals) / len(judge_totals)) if judge_totals else None,
        cost_usd=total_cost, duration_ms=int((time.time() - t_eval) * 1000),
    )
    return eval_run_id
