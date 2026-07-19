"""Record/replay of the nondeterministic boundary: LLM calls and the sandbox.

Recording wraps GraphDeps — the same seam tests use for stubs. Every call is
keyed by a sha256 of its request content, so parallel analyst branches replay
correctly no matter which order they complete in. Identical requests within a
run replay in first-recorded order via a per-key sequence number. Replay never
touches the network or the sandbox: a recorded run replays on a plane.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict

from langchain_core.messages import BaseMessage

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticTurn, PlannerTurn
from app.runtime.state import SandboxResult
from app.tracing.store import Store


def request_key(role: str, messages: list[BaseMessage]) -> str:
    content = json.dumps([[m.__class__.__name__, str(m.content)] for m in messages])
    return hashlib.sha256(f"{role}\n{content}".encode()).hexdigest()


def sandbox_key(code: str) -> str:
    return hashlib.sha256(f"sandbox\n{code}".encode()).hexdigest()


def _usage_dict(usage: LLMUsage) -> dict:
    return {"tokens_in": usage.tokens_in, "tokens_out": usage.tokens_out, "model": usage.model}


class Recorder:
    """Appends recordings with thread-safe per-key sequence numbers."""

    def __init__(self, store: Store, run_id: str) -> None:
        self.store = store
        self.run_id = run_id
        self._seq: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(self, kind: str, key: str, response: dict) -> None:
        with self._lock:
            seq = self._seq[key]
            self._seq[key] += 1
        self.store.add_recording(self.run_id, key, seq, kind, response)


def recording_deps(inner: GraphDeps, recorder: Recorder) -> GraphDeps:
    """Wrap deps so every call passes through unchanged but is recorded."""

    def structured(role: str, fn):
        def call(messages):
            turn, usage = fn(messages)
            recorder.record(
                "llm",
                request_key(role, messages),
                {"turn": turn.model_dump(), "usage": _usage_dict(usage)},
            )
            return turn, usage

        return call

    def compose(messages):
        text, usage = inner.compose(messages)
        recorder.record(
            "llm", request_key("composer", messages), {"text": text, "usage": _usage_dict(usage)}
        )
        return text, usage

    def run_code(code: str, dataset_path: str) -> SandboxResult:
        result = inner.run_code(code, dataset_path)
        recorder.record("sandbox", sandbox_key(code), result.model_dump())
        return result

    return GraphDeps(
        planner=structured("planner", inner.planner),
        analyst_turn=structured("analyst", inner.analyst_turn),
        critic_turn=structured("critic", inner.critic_turn),
        compose=compose,
        run_code=run_code,
    )


class ReplayMiss(Exception):
    """A replayed run issued a request that was never recorded."""

    def __init__(self, kind: str, key: str) -> None:
        super().__init__(
            f"replay miss: no recorded {kind} response for request {key[:12]}… — "
            "the replayed graph diverged from the original run"
        )


def replay_deps(recordings: list[dict]) -> GraphDeps:
    """GraphDeps that answer every request from recordings. Fully offline."""

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(recordings, key=lambda r: r["seq"]):
        buckets[row["key"]].append(row["response"])
    lock = threading.Lock()

    def pop(kind: str, key: str) -> dict:
        with lock:
            bucket = buckets.get(key)
            if not bucket:
                raise ReplayMiss(kind, key)
            return bucket.pop(0)

    def replay_usage(rec: dict) -> LLMUsage:
        usage = rec.get("usage", {})
        # model="replay" is deliberately unpriced: replays keep tokens, cost $0
        return LLMUsage(usage.get("tokens_in", 0), usage.get("tokens_out", 0), model="replay")

    def structured(role: str, schema):
        def call(messages):
            rec = pop("llm", request_key(role, messages))
            return schema.model_validate(rec["turn"]), replay_usage(rec)

        return call

    def compose(messages):
        rec = pop("llm", request_key("composer", messages))
        return rec["text"], replay_usage(rec)

    def run_code(code: str, dataset_path: str) -> SandboxResult:
        return SandboxResult.model_validate(pop("sandbox", sandbox_key(code)))

    return GraphDeps(
        planner=structured("planner", PlannerTurn),
        analyst_turn=structured("analyst", AnalystTurn),
        critic_turn=structured("critic", CriticTurn),
        compose=compose,
        run_code=run_code,
    )
