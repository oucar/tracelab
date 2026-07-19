# M4 — Evals + Adaptive Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Golden-dataset eval harness (programmatic tier + calibrated LLM judge), regression tracking with an Evals screen, CI gate, and an adaptive complexity router that lets simple questions skip the planner.

**Architecture:** Everything is testable keyless via the existing `GraphDeps` stub pattern. Golden expected answers are *derived* (pandas/scipy functions in `derivations.py` regenerate the YAML via `python -m app.evals golden --write`), so the golden set is self-verifying and survives dataset regeneration. Eval results land in two new SQLite tables in the existing `Store`. The router is a new first node in the LangGraph graph emitting the already-existing `HANDOFF` event type; the composer folds (skips its LLM call) for single-verified-finding runs.

**Tech Stack:** Python 3.12+ (venv is 3.14), FastAPI, LangGraph, pandas/scipy/sklearn (already deps), PyYAML (new dep), React 18 + MUI 6 + MUI X Charts/DataGrid 7, TanStack Query 5.

## Global Constraints

- Work on branch `m4-evals-routing`, created from `m3-observability` (master is 13 commits behind m3; do NOT merge to master, do NOT push — commits stay local per owner rule).
- All tests run keyless: LLMs injected through `GraphDeps`; never call OpenAI in pytest.
- Backend: ruff line-length 100, target py311; run `cd backend && .venv/bin/pytest` (or `make test` at repo root) and `.venv/bin/python -m ruff check .`.
- Frontend: strict TS with `noUnusedLocals`; the only gate is `npm run typecheck` + `npm run build` (no test runner exists — do not add one in M4).
- No new frontend dependencies. Backend gains only `pyyaml`.
- `MILESTONES.md` at repo root is gitignored — update checkboxes as tasks land, never `git add` it.
- Dates in per-day logic: not applicable here; eval timestamps are epoch floats like the rest of the store.
- `bus` is a process-global singleton; the API registers `bus.add_sink(store().add_span)` in `main.py` — the eval CLI must register its own sink exactly once.
- Existing `GraphDeps` constructions in tests use keyword args (`planner=`, `analyst_turn=`, `critic_turn=`, `compose=`, optional `run_code=`). New fields must default (e.g. `router: RouterFn | None = None`) so every existing test still constructs.
- `execute_run(state, deps)` is the single entry point for running the graph (it checkpoints via SqliteSaver keyed by `thread_id=run_id`); tests rely on conftest's `_isolated_checkpoints` autouse fixture.

---

### Task 1: Sample dataset generator + the three bundled CSVs

**Files:**
- Create: `backend/scripts/make_samples.py`
- Create: `data/samples/ATTRIBUTION.md`
- Create (generated): `data/samples/taxi_trips.csv`, `data/samples/retail_sales.csv`, `data/samples/weather.csv`
- Test: `backend/tests/test_samples.py`

**Interfaces:**
- Consumes: nothing (pure numpy/pandas).
- Produces: three CSVs at `data/samples/` with fixed columns (below) and deterministic content (seed 42). Later tasks reference them by repo-root-relative path.
  - `taxi_trips.csv`: `pickup_hour:int, day:str(Mon..Sun), distance_km:float, passenger_count:int, payment_type:str(card|cash), fare:float, tip:float` (900 rows)
  - `retail_sales.csv`: `date:YYYY-MM-DD, day:str, units:int, revenue:float, promo:int(0|1)` (730 rows)
  - `weather.csv`: `date:YYYY-MM-DD, temp_c:float, humidity:float, wind_kmh:float, precip_mm:float, condition:str(sunny|cloudy|rain)` (730 rows)

The data is synthetic but seeded with real effects the golden questions probe: weekend fares +3, rush-hour +2, fare ~ 2.2×distance (taxi); upward trend, weekend lift, promo lift, promo-more-likely-on-weekends (retail); seasonal temperature with 6 injected ±15° anomalies, humidity anti-correlated with temp, precip driven by humidity (weather).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_samples.py
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "data" / "samples"

EXPECTED_COLUMNS = {
    "taxi_trips.csv": ["pickup_hour", "day", "distance_km", "passenger_count",
                       "payment_type", "fare", "tip"],
    "retail_sales.csv": ["date", "day", "units", "revenue", "promo"],
    "weather.csv": ["date", "temp_c", "humidity", "wind_kmh", "precip_mm", "condition"],
}


def test_generator_is_deterministic(tmp_path):
    script = REPO / "backend" / "scripts" / "make_samples.py"
    for out in (tmp_path / "a", tmp_path / "b"):
        subprocess.run([sys.executable, str(script), "--out", str(out)], check=True)
    for name in EXPECTED_COLUMNS:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_committed_samples_match_schema():
    for name, cols in EXPECTED_COLUMNS.items():
        df = pd.read_csv(SAMPLES / name)
        assert list(df.columns) == cols
        assert len(df) in (900, 730)


def test_taxi_has_designed_effects():
    df = pd.read_csv(SAMPLES / "taxi_trips.csv")
    weekend = df[df["day"].isin(["Sat", "Sun"])]["fare"].mean()
    weekday = df[~df["day"].isin(["Sat", "Sun"])]["fare"].mean()
    assert weekend > weekday
    assert df["fare"].corr(df["distance_km"]) > 0.6
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_samples.py -v`
Expected: FAIL (script and CSVs missing).

- [ ] **Step 3: Write the generator**

```python
# backend/scripts/make_samples.py
"""Generate the three bundled sample datasets deterministically (seed 42).

Synthetic, modeled on real-world shapes (see data/samples/ATTRIBUTION.md).
Regenerate: python backend/scripts/make_samples.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "samples"


def _hourly_profile() -> np.ndarray:
    w = np.ones(24)
    w[[7, 8, 9, 17, 18, 19]] = 3.0
    w[[0, 1, 2, 3, 4]] = 0.4
    return w / w.sum()


def make_taxi(rng: np.random.Generator) -> pd.DataFrame:
    n = 900
    hour = rng.choice(24, size=n, p=_hourly_profile())
    day = rng.choice(DAYS, size=n)
    distance = np.round(rng.lognormal(1.1, 0.55, n), 2)
    passengers = rng.choice([1, 2, 3, 4], size=n, p=[0.62, 0.22, 0.10, 0.06])
    payment = rng.choice(["card", "cash"], size=n, p=[0.7, 0.3])
    weekend = np.isin(day, ["Sat", "Sun"])
    rush = np.isin(hour, [17, 18, 19])
    fare = 3.0 + 2.2 * distance + 2.0 * rush + 3.0 * weekend + rng.normal(0, 1.2, n)
    fare = np.round(np.clip(fare, 3.0, None), 2)
    tip = np.where(payment == "card", fare * rng.uniform(0.10, 0.25, n), rng.uniform(0, 1, n))
    return pd.DataFrame({
        "pickup_hour": hour, "day": day, "distance_km": distance,
        "passenger_count": passengers, "payment_type": payment,
        "fare": fare, "tip": np.round(tip, 2),
    })


def make_retail(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    t = np.arange(730)
    weekend = np.isin(dates.dayofweek, [5, 6])
    yearly = 60 * np.sin(2 * np.pi * (t % 365) / 365 - np.pi / 2) + 60
    promo = (rng.random(730) < np.where(weekend, 0.25, 0.10)).astype(int)
    units = 200 + 0.25 * t + 40 * weekend + yearly + 90 * promo + rng.normal(0, 25, 730)
    units = np.round(units).clip(20).astype(int)
    revenue = np.round(units * 4.5 * rng.uniform(0.95, 1.05, 730), 2)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "day": [DAYS[d] for d in dates.dayofweek],
        "units": units, "revenue": revenue, "promo": promo,
    })


def make_weather(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    t = np.arange(730)
    temp = 12 + 10 * np.sin(2 * np.pi * (t - 105) / 365.25) + rng.normal(0, 2.5, 730)
    anomaly_idx = rng.choice(730, 6, replace=False)
    temp[anomaly_idx] += rng.choice([-1.0, 1.0], 6) * 15
    humidity = np.clip(85 - 1.8 * temp + rng.normal(0, 7, 730), 20, 100)
    wind = np.round(np.abs(rng.normal(14, 6, 730)), 1)
    precip = np.round(np.where(humidity > 70, rng.exponential(4, 730),
                               rng.exponential(0.6, 730)), 1)
    condition = np.select([precip > 5, humidity > 70], ["rain", "cloudy"], default="sunny")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "temp_c": np.round(temp, 1),
        "humidity": np.round(humidity, 1), "wind_kmh": wind,
        "precip_mm": precip, "condition": condition,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    make_taxi(rng).to_csv(args.out / "taxi_trips.csv", index=False)
    make_retail(rng).to_csv(args.out / "retail_sales.csv", index=False)
    make_weather(rng).to_csv(args.out / "weather.csv", index=False)
    print(f"wrote 3 datasets to {args.out}")


if __name__ == "__main__":
    main()
```

Note: a single `rng` is threaded through all three generators in order — do not reseed per dataset, and never reorder the calls (that changes every dataset and invalidates golden answers).

- [ ] **Step 4: Generate the CSVs + attribution, run tests**

Run: `cd backend && .venv/bin/python scripts/make_samples.py`

```markdown
<!-- data/samples/ATTRIBUTION.md -->
# Sample datasets

All three datasets are **synthetic**, generated deterministically by
`backend/scripts/make_samples.py` (numpy seed 42) and modeled on familiar
real-world shapes (NYC-style taxi trips, daily retail sales, daily weather).
Synthetic data keeps the repo license-clean and makes the golden-eval answers
exactly derivable — see `backend/app/evals/derivations.py`.
```

Run: `cd backend && .venv/bin/pytest tests/test_samples.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git checkout -b m4-evals-routing m3-observability
git add backend/scripts/make_samples.py data/samples/ backend/tests/test_samples.py
git commit -m "feat(m4): seeded sample dataset generator + 3 bundled golden datasets"
```

---

### Task 2: Golden set schema, loader, question YAMLs

**Files:**
- Create: `backend/app/evals/__init__.py` (empty)
- Create: `backend/app/evals/golden.py`
- Create: `backend/app/evals/golden/taxi.yaml`, `retail.yaml`, `weather.yaml`
- Modify: `backend/pyproject.toml` (add `pyyaml>=6.0` to dependencies)
- Test: `backend/tests/test_golden.py`

**Interfaces:**
- Produces: `GoldenExpected(kind, value, tolerance, direction, significant, method_family)`, `GoldenQuestion(id, question, expected, tags)`, `GoldenDataset(name, csv, questions)`, `load_golden(golden_dir: Path) -> list[GoldenDataset]`, `GOLDEN_DIR` (module constant `Path(__file__).parent / "golden"`), `REPO_ROOT` (`Path(__file__).resolve().parents[3]`).
- `expected.value` may be `null` in the YAML until Task 3's `golden --write` fills it.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_golden.py
from app.evals.golden import GOLDEN_DIR, REPO_ROOT, load_golden


def test_golden_sets_load_and_are_wellformed():
    sets = load_golden(GOLDEN_DIR)
    assert {s.name for s in sets} == {"taxi", "retail", "weather"}
    ids = [q.id for s in sets for q in s.questions]
    assert len(ids) == len(set(ids))
    for s in sets:
        assert (REPO_ROOT / s.csv).exists()
        assert 10 <= len(s.questions) <= 15
        kinds = {q.expected.kind for q in s.questions}
        assert "statistical" in kinds and "narrative" in kinds
        for q in s.questions:
            if q.expected.kind == "statistical":
                assert q.expected.method_family
                assert "stats" in q.tags
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_golden.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement schema + loader**

Add `"pyyaml>=6.0",` to `[project] dependencies` in `backend/pyproject.toml`, then `cd backend && .venv/bin/pip install -e ".[dev]" -q`.

```python
# backend/app/evals/golden.py
"""Golden eval set: schema + YAML loader.

expected.value fields are derived, not hand-typed: `python -m app.evals golden --write`
recomputes them from data/samples via derivations.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

GOLDEN_DIR = Path(__file__).parent / "golden"
REPO_ROOT = Path(__file__).resolve().parents[3]


class GoldenExpected(BaseModel):
    kind: Literal["numeric", "categorical", "statistical", "narrative"]
    value: float | str | None = None
    tolerance: float = 0.0  # relative band for floats; 0.0 means exact
    direction: Literal["higher", "lower", ""] = ""
    significant: bool | None = None
    method_family: str = ""


class GoldenQuestion(BaseModel):
    id: str
    question: str
    expected: GoldenExpected
    tags: list[str] = Field(default_factory=list)


class GoldenDataset(BaseModel):
    name: str
    csv: str  # repo-root-relative, e.g. data/samples/taxi_trips.csv
    questions: list[GoldenQuestion]


def load_golden(golden_dir: Path = GOLDEN_DIR) -> list[GoldenDataset]:
    return [
        GoldenDataset.model_validate(yaml.safe_load(p.read_text()))
        for p in sorted(golden_dir.glob("*.yaml"))
    ]
```

- [ ] **Step 4: Write the three question YAMLs**

`expected.value: null` everywhere below is deliberate — Task 3 fills them. Statistical `direction`/`significant` are also filled by Task 3; author them as `direction: ""` / `significant: null` here.

```yaml
# backend/app/evals/golden/taxi.yaml
name: taxi
csv: data/samples/taxi_trips.csv
questions:
  - id: taxi-001
    question: "How many trips are in the dataset?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.0}
  - id: taxi-002
    question: "What is the average fare?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.02}
  - id: taxi-003
    question: "Which pickup hour (0-23) has the highest average fare?"
    tags: [aggregation, groupby]
    expected: {kind: numeric, value: null, tolerance: 0.0}
  - id: taxi-004
    question: "Which day of the week has the most trips?"
    tags: [aggregation, groupby]
    expected: {kind: categorical, value: null}
  - id: taxi-005
    question: "What is the maximum trip distance in km?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.0}
  - id: taxi-006
    question: "How many trips were paid in cash?"
    tags: [aggregation, filter]
    expected: {kind: numeric, value: null, tolerance: 0.0}
  - id: taxi-007
    question: "Is the average fare significantly higher on weekends than on weekdays?"
    tags: [stats, hypothesis-testing]
    expected: {kind: statistical, direction: "", significant: null, method_family: mean-comparison}
  - id: taxi-008
    question: "Is there a significant correlation between trip distance and fare?"
    tags: [stats, correlation]
    expected: {kind: statistical, direction: "", significant: null, method_family: correlation}
  - id: taxi-009
    question: "Do card-paying passengers tip significantly more than cash-paying passengers?"
    tags: [stats, hypothesis-testing]
    expected: {kind: statistical, direction: "", significant: null, method_family: mean-comparison}
  - id: taxi-010
    question: "What is the median fare?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.02}
  - id: taxi-011
    question: "Summarize the main drivers of fare in this dataset."
    tags: [narrative]
    expected: {kind: narrative}
```

```yaml
# backend/app/evals/golden/retail.yaml
name: retail
csv: data/samples/retail_sales.csv
questions:
  - id: retail-001
    question: "How many days of sales does the dataset cover?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.0}
  - id: retail-002
    question: "What was the total revenue over the whole period?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.01}
  - id: retail-003
    question: "What is the average number of units sold per day?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.02}
  - id: retail-004
    question: "Which month of the year (1-12) has the highest average daily revenue?"
    tags: [aggregation, groupby, dates]
    expected: {kind: numeric, value: null, tolerance: 0.0}
  - id: retail-005
    question: "How many promo days are in the dataset?"
    tags: [aggregation, filter]
    expected: {kind: numeric, value: null, tolerance: 0.0}
  - id: retail-006
    question: "Do promo days sell significantly more units than non-promo days?"
    tags: [stats, hypothesis-testing]
    expected: {kind: statistical, direction: "", significant: null, method_family: mean-comparison}
  - id: retail-007
    question: "Is running a promotion significantly associated with weekends?"
    tags: [stats, association]
    expected: {kind: statistical, direction: "", significant: null, method_family: chi-square}
  - id: retail-008
    question: "Is there a significant upward trend in units sold over time?"
    tags: [stats, trend]
    expected: {kind: statistical, direction: "", significant: null, method_family: trend}
  - id: retail-009
    question: "What was the highest single-day revenue?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.01}
  - id: retail-010
    question: "What is the average number of units sold on weekend days?"
    tags: [aggregation, filter]
    expected: {kind: numeric, value: null, tolerance: 0.02}
  - id: retail-011
    question: "Describe the seasonality of sales in this dataset."
    tags: [narrative]
    expected: {kind: narrative}
```

```yaml
# backend/app/evals/golden/weather.yaml
name: weather
csv: data/samples/weather.csv
questions:
  - id: weather-001
    question: "How many daily observations are in the dataset?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.0}
  - id: weather-002
    question: "What was the highest temperature recorded?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.01}
  - id: weather-003
    question: "What is the mean temperature over the whole period?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.02}
  - id: weather-004
    question: "Which weather condition occurs most often?"
    tags: [aggregation, groupby]
    expected: {kind: categorical, value: null}
  - id: weather-005
    question: "Is there a significant correlation between temperature and humidity?"
    tags: [stats, correlation]
    expected: {kind: statistical, direction: "", significant: null, method_family: correlation}
  - id: weather-006
    question: "Is it significantly windier on rain days than on sunny days?"
    tags: [stats, hypothesis-testing]
    expected: {kind: statistical, direction: "", significant: null, method_family: mean-comparison}
  - id: weather-007
    question: "How many days had precipitation above 10 mm?"
    tags: [aggregation, filter]
    expected: {kind: numeric, value: null, tolerance: 0.0}
  - id: weather-008
    question: "Does humidity significantly predict precipitation?"
    tags: [stats, regression]
    expected: {kind: statistical, direction: "", significant: null, method_family: regression}
  - id: weather-009
    question: "What was the average temperature in July 2024?"
    tags: [aggregation, dates]
    expected: {kind: numeric, value: null, tolerance: 0.02}
  - id: weather-010
    question: "What is the average wind speed?"
    tags: [aggregation]
    expected: {kind: numeric, value: null, tolerance: 0.02}
  - id: weather-011
    question: "Are there any anomalous temperature readings? Describe them."
    tags: [narrative, anomaly]
    expected: {kind: narrative}
```

- [ ] **Step 5: Run tests, lint, commit**

Run: `cd backend && .venv/bin/pytest tests/test_golden.py tests/test_samples.py -v` → PASS.
Run: `cd backend && .venv/bin/python -m ruff check .` → clean.

```bash
git add backend/app/evals backend/tests/test_golden.py backend/pyproject.toml
git commit -m "feat(m4): golden eval set schema, loader, and 33 questions across 3 datasets"
```

---

### Task 3: Derivations — self-verifying expected answers + `golden --write`

**Files:**
- Create: `backend/app/evals/derivations.py`
- Create: `backend/app/evals/__main__.py` (CLI skeleton with the `golden` subcommand; later tasks add `run`, `label-template`, `calibration`)
- Modify (generated): the three golden YAMLs (values filled in)
- Test: `backend/tests/test_derivations.py`

**Interfaces:**
- Consumes: `GoldenExpected`, `load_golden`, `GOLDEN_DIR`, `REPO_ROOT` from Task 2.
- Produces: `DERIVATIONS: dict[str, Callable[[pd.DataFrame], GoldenExpected]]`, `derive_all() -> dict[str, GoldenExpected]` (loads each CSV once), `write_golden(golden_dir: Path) -> None` (rewrites YAMLs preserving question text/tags/tolerance, replacing `value`/`direction`/`significant`), and the CLI entry `python -m app.evals golden --write`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_derivations.py
from app.evals.derivations import DERIVATIONS, derive_all
from app.evals.golden import GOLDEN_DIR, load_golden


def test_every_nonnarrative_question_has_a_derivation():
    sets = load_golden(GOLDEN_DIR)
    for s in sets:
        for q in s.questions:
            if q.expected.kind != "narrative":
                assert q.id in DERIVATIONS, f"no derivation for {q.id}"


def test_golden_yaml_matches_derivations():
    """The committed YAML values must equal what the derivations compute.

    If this fails after changing make_samples.py, run:
        python -m app.evals golden --write
    """
    derived = derive_all()
    for s in load_golden(GOLDEN_DIR):
        for q in s.questions:
            if q.expected.kind == "narrative":
                continue
            exp, got = q.expected, derived[q.id]
            assert exp.kind == got.kind, q.id
            if exp.kind in ("numeric", "categorical"):
                assert exp.value == got.value, q.id
            else:
                assert (exp.direction, exp.significant, exp.method_family) == (
                    got.direction, got.significant, got.method_family), q.id
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_derivations.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement derivations + writer + CLI**

```python
# backend/app/evals/derivations.py
"""One derivation function per golden question — the machine-checkable ground truth.

Numeric floats are rounded to 4 decimals so YAML round-trips exactly.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd
import yaml
from scipy import stats

from app.evals.golden import GOLDEN_DIR, REPO_ROOT, GoldenExpected, load_golden

Derivation = Callable[[pd.DataFrame], GoldenExpected]
DERIVATIONS: dict[str, Derivation] = {}
ALPHA = 0.05
WEEKEND = ["Sat", "Sun"]


def derivation(qid: str) -> Callable[[Derivation], Derivation]:
    def register(fn: Derivation) -> Derivation:
        DERIVATIONS[qid] = fn
        return fn
    return register


def numeric(value: float, tolerance: float = 0.0) -> GoldenExpected:
    v = float(value)
    return GoldenExpected(kind="numeric", value=int(v) if v == int(v) else round(v, 4),
                         tolerance=tolerance)


def categorical(value: str) -> GoldenExpected:
    return GoldenExpected(kind="categorical", value=str(value))


def statistical(direction: str, p_value: float, method_family: str) -> GoldenExpected:
    return GoldenExpected(kind="statistical", direction=direction,
                         significant=bool(p_value < ALPHA), method_family=method_family)


# --- taxi ---------------------------------------------------------------
@derivation("taxi-001")
def _taxi_count(df: pd.DataFrame) -> GoldenExpected:
    return numeric(len(df))


@derivation("taxi-002")
def _taxi_avg_fare(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["fare"].mean(), 0.02)


@derivation("taxi-003")
def _taxi_top_hour(df: pd.DataFrame) -> GoldenExpected:
    return numeric(int(df.groupby("pickup_hour")["fare"].mean().idxmax()))


@derivation("taxi-004")
def _taxi_busiest_day(df: pd.DataFrame) -> GoldenExpected:
    return categorical(df["day"].value_counts().idxmax())


@derivation("taxi-005")
def _taxi_max_distance(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["distance_km"].max())


@derivation("taxi-006")
def _taxi_cash_trips(df: pd.DataFrame) -> GoldenExpected:
    return numeric(int((df["payment_type"] == "cash").sum()))


@derivation("taxi-007")
def _taxi_weekend_fare(df: pd.DataFrame) -> GoldenExpected:
    wk = df[df["day"].isin(WEEKEND)]["fare"]
    wd = df[~df["day"].isin(WEEKEND)]["fare"]
    _, p = stats.ttest_ind(wk, wd, equal_var=False)
    return statistical("higher" if wk.mean() > wd.mean() else "lower", p, "mean-comparison")


@derivation("taxi-008")
def _taxi_fare_distance_corr(df: pd.DataFrame) -> GoldenExpected:
    r, p = stats.pearsonr(df["distance_km"], df["fare"])
    return statistical("higher" if r > 0 else "lower", p, "correlation")


@derivation("taxi-009")
def _taxi_tip_by_payment(df: pd.DataFrame) -> GoldenExpected:
    card = df[df["payment_type"] == "card"]["tip"]
    cash = df[df["payment_type"] == "cash"]["tip"]
    _, p = stats.mannwhitneyu(card, cash)
    return statistical("higher" if card.mean() > cash.mean() else "lower", p, "mean-comparison")


@derivation("taxi-010")
def _taxi_median_fare(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["fare"].median(), 0.02)


# --- retail -------------------------------------------------------------
@derivation("retail-001")
def _retail_days(df: pd.DataFrame) -> GoldenExpected:
    return numeric(len(df))


@derivation("retail-002")
def _retail_total_revenue(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["revenue"].sum(), 0.01)


@derivation("retail-003")
def _retail_avg_units(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["units"].mean(), 0.02)


@derivation("retail-004")
def _retail_best_month(df: pd.DataFrame) -> GoldenExpected:
    months = pd.to_datetime(df["date"]).dt.month
    return numeric(int(df.groupby(months)["revenue"].mean().idxmax()))


@derivation("retail-005")
def _retail_promo_days(df: pd.DataFrame) -> GoldenExpected:
    return numeric(int(df["promo"].sum()))


@derivation("retail-006")
def _retail_promo_units(df: pd.DataFrame) -> GoldenExpected:
    promo = df[df["promo"] == 1]["units"]
    rest = df[df["promo"] == 0]["units"]
    _, p = stats.ttest_ind(promo, rest, equal_var=False)
    return statistical("higher" if promo.mean() > rest.mean() else "lower", p, "mean-comparison")


@derivation("retail-007")
def _retail_promo_weekend(df: pd.DataFrame) -> GoldenExpected:
    weekend = df["day"].isin(WEEKEND)
    table = pd.crosstab(weekend, df["promo"])
    _, p, _, _ = stats.chi2_contingency(table)
    rate_wk = df[weekend]["promo"].mean()
    rate_wd = df[~weekend]["promo"].mean()
    return statistical("higher" if rate_wk > rate_wd else "lower", p, "chi-square")


@derivation("retail-008")
def _retail_trend(df: pd.DataFrame) -> GoldenExpected:
    res = stats.linregress(range(len(df)), df["units"])
    return statistical("higher" if res.slope > 0 else "lower", res.pvalue, "trend")


@derivation("retail-009")
def _retail_best_day(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["revenue"].max(), 0.01)


@derivation("retail-010")
def _retail_weekend_units(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df[df["day"].isin(WEEKEND)]["units"].mean(), 0.02)


# --- weather ------------------------------------------------------------
@derivation("weather-001")
def _weather_days(df: pd.DataFrame) -> GoldenExpected:
    return numeric(len(df))


@derivation("weather-002")
def _weather_max_temp(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["temp_c"].max(), 0.01)


@derivation("weather-003")
def _weather_mean_temp(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["temp_c"].mean(), 0.02)


@derivation("weather-004")
def _weather_top_condition(df: pd.DataFrame) -> GoldenExpected:
    return categorical(df["condition"].value_counts().idxmax())


@derivation("weather-005")
def _weather_temp_humidity(df: pd.DataFrame) -> GoldenExpected:
    r, p = stats.pearsonr(df["temp_c"], df["humidity"])
    return statistical("higher" if r > 0 else "lower", p, "correlation")


@derivation("weather-006")
def _weather_wind_rain(df: pd.DataFrame) -> GoldenExpected:
    rain = df[df["condition"] == "rain"]["wind_kmh"]
    sunny = df[df["condition"] == "sunny"]["wind_kmh"]
    _, p = stats.ttest_ind(rain, sunny, equal_var=False)
    return statistical("higher" if rain.mean() > sunny.mean() else "lower", p, "mean-comparison")


@derivation("weather-007")
def _weather_wet_days(df: pd.DataFrame) -> GoldenExpected:
    return numeric(int((df["precip_mm"] > 10).sum()))


@derivation("weather-008")
def _weather_humidity_precip(df: pd.DataFrame) -> GoldenExpected:
    res = stats.linregress(df["humidity"], df["precip_mm"])
    return statistical("higher" if res.slope > 0 else "lower", res.pvalue, "regression")


@derivation("weather-009")
def _weather_july_temp(df: pd.DataFrame) -> GoldenExpected:
    dates = pd.to_datetime(df["date"])
    july = df[(dates.dt.year == 2024) & (dates.dt.month == 7)]
    return numeric(july["temp_c"].mean(), 0.02)


@derivation("weather-010")
def _weather_avg_wind(df: pd.DataFrame) -> GoldenExpected:
    return numeric(df["wind_kmh"].mean(), 0.02)


# --- writer -------------------------------------------------------------
def derive_all() -> dict[str, GoldenExpected]:
    out: dict[str, GoldenExpected] = {}
    for s in load_golden(GOLDEN_DIR):
        df = pd.read_csv(REPO_ROOT / s.csv)
        for q in s.questions:
            if q.id in DERIVATIONS:
                derived = DERIVATIONS[q.id](df)
                derived.tolerance = q.expected.tolerance  # tolerance is authored, not derived
                out[q.id] = derived
    return out


def write_golden(golden_dir: Path = GOLDEN_DIR) -> None:
    derived = derive_all()
    for path in sorted(golden_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        for q in raw["questions"]:
            if q["id"] in derived:
                q["expected"] = derived[q["id"]].model_dump(exclude_defaults=False)
        path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
```

Add the missing `from pathlib import Path` import at the top of `derivations.py`.

```python
# backend/app/evals/__main__.py
"""CLI: python -m app.evals <subcommand>. Subcommands grow over M4/M5."""
import argparse

from app.evals.derivations import write_golden


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.evals")
    sub = parser.add_subparsers(dest="cmd", required=True)
    golden = sub.add_parser("golden", help="golden-set maintenance")
    golden.add_argument("--write", action="store_true",
                        help="recompute expected values from data/samples")
    args = parser.parse_args()
    if args.cmd == "golden" and args.write:
        write_golden()
        print("golden YAMLs updated from derivations")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Fill the YAMLs, run tests**

Run: `cd backend && .venv/bin/python -m app.evals golden --write`
Then inspect the diff (`git diff backend/app/evals/golden/`) — every non-narrative `value`/`direction`/`significant` should now be concrete; spot-check 2-3 against a quick pandas one-liner.
Run: `cd backend && .venv/bin/pytest tests/test_derivations.py tests/test_golden.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/evals backend/tests/test_derivations.py
git commit -m "feat(m4): derivations make the golden set self-verifying; golden --write CLI"
```

---

### Task 4: Programmatic (tier-1) scoring

**Files:**
- Create: `backend/app/evals/scoring.py`
- Test: `backend/tests/test_scoring.py`

**Interfaces:**
- Consumes: `GoldenExpected` (Task 2); `FinalAnswer`, `VerifiedClaim`, `Claim`, `Methodology` from `app.runtime.state`; `numbers_match(a, b, rel_tol)` from `app.runtime.reconcile`.
- Produces: `TierOneScore(scorable: bool, passed: bool, detail: str)` and `score_tier1(expected: GoldenExpected, final: FinalAnswer | None) -> TierOneScore`. Policy: a question passes if ANY claim of the matching kind matches (runs legitimately emit several claims); statistical match = direction + significance conclusion + acceptable method family.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_scoring.py
from app.evals.golden import GoldenExpected
from app.evals.scoring import score_tier1
from app.runtime.state import Claim, FinalAnswer, Methodology, VerifiedClaim


def vc(claim: Claim) -> VerifiedClaim:
    return VerifiedClaim(claim=claim, status="verified", detail="")


def final_with(*claims: Claim) -> FinalAnswer:
    return FinalAnswer(narrative="x", claims=[vc(c) for c in claims], charts=[], failed=False)


def num_claim(value: float) -> Claim:
    return Claim(id="c1", step_id=1, text="t", kind="numeric", value=value,
                 direction="", significant=None, methodology=None)


def stat_claim(direction: str, significant: bool, method: str) -> Claim:
    return Claim(id="c2", step_id=1, text="t", kind="statistical", value=None,
                 direction=direction, significant=significant,
                 methodology=Methodology(method=method, n=100, p_value=0.001,
                                         effect_size=0.5, effect_size_name="d",
                                         assumptions=[]))


def test_numeric_exact_and_tolerance():
    exp = GoldenExpected(kind="numeric", value=730, tolerance=0.0)
    assert score_tier1(exp, final_with(num_claim(730))).passed
    assert not score_tier1(exp, final_with(num_claim(731))).passed
    band = GoldenExpected(kind="numeric", value=100.0, tolerance=0.02)
    assert score_tier1(band, final_with(num_claim(101.5))).passed
    assert not score_tier1(band, final_with(num_claim(105.0))).passed


def test_numeric_passes_if_any_claim_matches():
    exp = GoldenExpected(kind="numeric", value=42, tolerance=0.0)
    assert score_tier1(exp, final_with(num_claim(7), num_claim(42))).passed


def test_categorical_case_insensitive():
    exp = GoldenExpected(kind="categorical", value="Sat")
    claim = Claim(id="c", step_id=1, text="t", kind="categorical", value="sat",
                  direction="", significant=None, methodology=None)
    assert score_tier1(exp, final_with(claim)).passed


def test_statistical_requires_direction_significance_and_family():
    exp = GoldenExpected(kind="statistical", direction="higher", significant=True,
                         method_family="mean-comparison")
    assert score_tier1(exp, final_with(stat_claim("higher", True, "Welch t-test"))).passed
    assert score_tier1(exp, final_with(stat_claim("higher", True, "Mann-Whitney U"))).passed
    assert not score_tier1(exp, final_with(stat_claim("lower", True, "t-test"))).passed
    assert not score_tier1(exp, final_with(stat_claim("higher", False, "t-test"))).passed
    assert not score_tier1(exp, final_with(stat_claim("higher", True, "chi-square"))).passed


def test_direction_synonyms():
    exp = GoldenExpected(kind="statistical", direction="higher", significant=True,
                         method_family="correlation")
    assert score_tier1(exp, final_with(stat_claim("positive", True, "Pearson correlation"))).passed


def test_failed_or_missing_run_fails():
    exp = GoldenExpected(kind="numeric", value=1, tolerance=0.0)
    assert not score_tier1(exp, None).passed
    failed = FinalAnswer(narrative="", claims=[], charts=[], failed=True)
    assert not score_tier1(exp, failed).passed


def test_narrative_not_scorable():
    s = score_tier1(GoldenExpected(kind="narrative"), final_with(num_claim(1)))
    assert not s.scorable
```

- [ ] **Step 2: Run to verify it fails** → `cd backend && .venv/bin/pytest tests/test_scoring.py -v` → FAIL.

- [ ] **Step 3: Implement scoring**

```python
# backend/app/evals/scoring.py
"""Tier-1 programmatic scoring: compare structured claims against golden expectations."""
from __future__ import annotations

from pydantic import BaseModel

from app.evals.golden import GoldenExpected
from app.runtime.reconcile import numbers_match
from app.runtime.state import Claim, FinalAnswer

METHOD_FAMILIES: dict[str, set[str]] = {
    "mean-comparison": {"t-test", "ttest", "welch", "mann-whitney", "mannwhitney",
                        "wilcoxon", "anova"},
    "correlation": {"pearson", "spearman", "correlation", "kendall"},
    "chi-square": {"chi-square", "chi2", "chisquare", "fisher", "cramer"},
    "regression": {"regression", "ols", "linear", "logistic"},
    "trend": {"regression", "ols", "linear", "trend", "spearman", "pearson",
              "mann-kendall", "correlation"},
}

DIRECTION_SYNONYMS: dict[str, set[str]] = {
    "higher": {"higher", "positive", "increase", "increasing", "up", "more", "greater"},
    "lower": {"lower", "negative", "decrease", "decreasing", "down", "less", "smaller"},
}


class TierOneScore(BaseModel):
    scorable: bool
    passed: bool
    detail: str = ""


def _norm(s: object) -> str:
    return str(s).strip().lower()


def _method_in_family(method: str, family: str) -> bool:
    tokens = _norm(method).replace("-", " ").replace("_", " ")
    return any(key.replace("-", " ") in tokens or tokens in key
               for key in METHOD_FAMILIES.get(family, set()))


def _direction_matches(claimed: str, expected: str) -> bool:
    return _norm(claimed) in DIRECTION_SYNONYMS.get(expected, {expected})


def _numeric_matches(claimed: object, expected: float, tolerance: float) -> bool:
    try:
        value = float(claimed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    rel_tol = tolerance if tolerance > 0 else 1e-9
    return numbers_match(value, expected, rel_tol)


def _claim_matches(claim: Claim, expected: GoldenExpected) -> bool:
    if expected.kind == "numeric":
        return claim.kind in ("numeric", "categorical") and expected.value is not None \
            and _numeric_matches(claim.value, float(expected.value), expected.tolerance)
    if expected.kind == "categorical":
        return _norm(claim.value) == _norm(expected.value)
    if expected.kind == "statistical":
        if claim.kind != "statistical":
            return False
        method = claim.methodology.method if claim.methodology else ""
        return (
            _direction_matches(claim.direction or "", expected.direction)
            and claim.significant is not None
            and bool(claim.significant) == bool(expected.significant)
            and _method_in_family(method, expected.method_family)
        )
    return False


def score_tier1(expected: GoldenExpected, final: FinalAnswer | None) -> TierOneScore:
    if expected.kind == "narrative":
        return TierOneScore(scorable=False, passed=False, detail="narrative (judge only)")
    if final is None or final.failed:
        return TierOneScore(scorable=True, passed=False, detail="run failed or produced no answer")
    claims = [v.claim for v in final.claims]
    if not claims:
        return TierOneScore(scorable=True, passed=False, detail="no claims in answer")
    for claim in claims:
        if _claim_matches(claim, expected):
            return TierOneScore(scorable=True, passed=True,
                                detail=f"matched claim {claim.id}: {claim.value!r}")
    values = [c.value for c in claims]
    return TierOneScore(scorable=True, passed=False,
                        detail=f"no claim matched; expected {expected.value!r}, saw {values!r}")
```

Note for the implementer: check `numbers_match`'s real signature in `app/runtime/reconcile.py` before wiring (`numbers_match(a, b, rel_tol)` per the M2 code; integers compare exactly there — that's why `tolerance 0` questions must derive int values, which `numeric()` in Task 3 guarantees via the `int(v) if v == int(v)` cast).

- [ ] **Step 4: Run tests** → `cd backend && .venv/bin/pytest tests/test_scoring.py -v` → PASS. Also run the full suite (`.venv/bin/pytest`) to confirm nothing broke.

- [ ] **Step 5: Commit**

```bash
git add backend/app/evals/scoring.py backend/tests/test_scoring.py
git commit -m "feat(m4): tier-1 programmatic scoring against golden expectations"
```

---

### Task 5: LLM judge (tier-2) — schema, prompt, scoring fn

**Files:**
- Modify: `backend/app/agents/schemas.py` (add `JudgeTurn`)
- Create: `backend/app/agents/prompts/judge.md`
- Create: `backend/app/evals/judge.py`
- Modify: `backend/app/config.py` (add `judge_model: str = "gpt-4o"`; extend `model_for` with `"judge"`)
- Test: `backend/tests/test_judge.py`

**Interfaces:**
- Consumes: `_structured_fn(role, schema)` pattern from `app/agents/llm.py`; `LLMUsage`; `FinalAnswer`.
- Produces:
  - `JudgeTurn(clarity: int, uncertainty_honesty: int, chart_appropriateness: int, methodological_soundness: int, rationale: str)` — all four scores `Field(ge=1, le=5)`.
  - `JudgeFn = Callable[[list[BaseMessage]], tuple[JudgeTurn, LLMUsage]]`
  - `DIMENSIONS = ("clarity", "uncertainty_honesty", "chart_appropriateness", "methodological_soundness")`
  - `real_judge() -> JudgeFn` (wraps `_structured_fn("judge", JudgeTurn)`)
  - `judge_answer(question: str, final: FinalAnswer, judge: JudgeFn) -> tuple[JudgeTurn, LLMUsage]`
- Design note: the judge model deliberately does NOT follow `cheap_mode`'s collapse if avoidable — but in M4, `model_for` in cheap mode collapses all roles; that is acceptable for dev. M5's study passes explicit models and pins the judge to `gpt-4o` across configs.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_judge.py
import pytest
from pydantic import ValidationError

from app.agents.llm import LLMUsage
from app.agents.schemas import JudgeTurn
from app.evals.judge import DIMENSIONS, judge_answer
from app.runtime.state import FinalAnswer


def test_judge_turn_bounds():
    with pytest.raises(ValidationError):
        JudgeTurn(clarity=6, uncertainty_honesty=1, chart_appropriateness=1,
                  methodological_soundness=1, rationale="x")
    turn = JudgeTurn(clarity=4, uncertainty_honesty=5, chart_appropriateness=3,
                     methodological_soundness=2, rationale="ok")
    assert [getattr(turn, d) for d in DIMENSIONS] == [4, 5, 3, 2]


def test_judge_answer_builds_prompt_and_returns_turn():
    seen: dict = {}

    def stub_judge(messages):
        seen["prompt"] = messages[-1].content
        return (JudgeTurn(clarity=5, uncertainty_honesty=4, chart_appropriateness=3,
                          methodological_soundness=5, rationale="solid"),
                LLMUsage(tokens_in=100, tokens_out=20, model="gpt-4o"))

    final = FinalAnswer(narrative="Weekend fares are higher (p=0.001).",
                        claims=[], charts=[], failed=False)
    turn, usage = judge_answer("Is fare higher on weekends?", final, stub_judge)
    assert turn.clarity == 5
    assert usage.tokens_in == 100
    assert "Is fare higher on weekends?" in seen["prompt"]
    assert "Weekend fares are higher" in seen["prompt"]
```

- [ ] **Step 2: Run to verify it fails** → `cd backend && .venv/bin/pytest tests/test_judge.py -v` → FAIL.

- [ ] **Step 3: Implement**

Add to `backend/app/agents/schemas.py` (mirroring existing turn models):

```python
class JudgeTurn(BaseModel):
    """Tier-2 rubric scores, 1 (poor) to 5 (excellent)."""
    clarity: int = Field(ge=1, le=5)
    uncertainty_honesty: int = Field(ge=1, le=5)
    chart_appropriateness: int = Field(ge=1, le=5)
    methodological_soundness: int = Field(ge=1, le=5)
    rationale: str = ""
```

In `backend/app/config.py`, add `judge_model: str = "gpt-4o"` next to the other model fields and add `"judge": self.judge_model` to the non-cheap branch of `model_for` (cheap mode keeps collapsing every role to `analyst_model`, judge included — document that inline).

```markdown
<!-- backend/app/agents/prompts/judge.md -->
You are grading the answer an automated data analyst gave to a user's question.
Score each dimension 1-5 (1 = poor, 3 = acceptable, 5 = excellent). Be strict;
5 requires no meaningful flaw on that dimension.

- clarity: is the answer direct, well-structured, and does it actually answer the question?
- uncertainty_honesty: are unverified numbers, failures, and caveats surfaced honestly
  (5) or hidden/overclaimed (1)? An answer that admits failure honestly scores HIGH here.
- chart_appropriateness: do included charts fit the question and data (5), are they
  absent when one was clearly needed (2-3), or misleading (1)? If no chart was needed
  and none shown, score 4.
- methodological_soundness: for statistical claims — right test family for the data,
  effect size and p-value reported, assumptions acknowledged, claim strength
  proportionate to evidence. For purely descriptive answers, score on whether the
  aggregation actually answers the question.

Question:
{question}

Analyst's final answer (narrative, claims with verification status, chart count):
{answer}

Return your scores via the structured output schema, with a one-paragraph rationale.
```

```python
# backend/app/evals/judge.py
"""Tier-2 LLM judge: rubric-scores a FinalAnswer. Calibrated against human labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage

from app.agents.llm import LLMUsage, _structured_fn
from app.agents.schemas import JudgeTurn
from app.runtime.state import FinalAnswer

JudgeFn = Callable[[list], tuple[JudgeTurn, LLMUsage]]
DIMENSIONS = ("clarity", "uncertainty_honesty", "chart_appropriateness",
              "methodological_soundness")
_PROMPT = (Path(__file__).resolve().parents[1] / "agents" / "prompts" / "judge.md")


def real_judge() -> JudgeFn:
    return _structured_fn("judge", JudgeTurn)


def _answer_digest(final: FinalAnswer) -> str:
    claims = [
        {"text": v.claim.text, "kind": v.claim.kind, "value": v.claim.value,
         "status": v.status,
         "methodology": v.claim.methodology.model_dump() if v.claim.methodology else None}
        for v in final.claims
    ]
    return json.dumps({"narrative": final.narrative, "claims": claims,
                       "charts": len(final.charts), "failed": final.failed},
                      indent=2, default=str)


def judge_answer(question: str, final: FinalAnswer,
                 judge: JudgeFn) -> tuple[JudgeTurn, LLMUsage]:
    prompt = _PROMPT.read_text().format(question=question, answer=_answer_digest(final))
    return judge([HumanMessage(content=prompt)])
```

Implementer note: `_structured_fn` is private to `llm.py` — if importing it feels wrong, add a public alias `structured_fn = _structured_fn` in `llm.py` and import that; keep whichever ruff accepts without noise.

- [ ] **Step 4: Run tests** → `cd backend && .venv/bin/pytest tests/test_judge.py -v && .venv/bin/pytest` → PASS, suite green.

- [ ] **Step 5: Commit**

```bash
git add backend/app/agents/schemas.py backend/app/agents/prompts/judge.md \
  backend/app/evals/judge.py backend/app/config.py backend/tests/test_judge.py
git commit -m "feat(m4): LLM judge with 4-dimension rubric"
```

---

### Task 6: Store — eval tables

**Files:**
- Modify: `backend/app/tracing/store.py`
- Test: extend `backend/tests/test_store.py`

**Interfaces:**
- Produces (on `Store`):
  - `add_eval_run(id, created_at, label, git_sha, config_hash, config_json, questions_total, tier1_scorable, tier1_passed, judge_avg, cost_usd, duration_ms) -> None`
  - `add_eval_result(eval_run_id, question_id, run_id, dataset, tags_json, tier1_scorable, tier1_passed, tier1_detail, judge_json, judge_rationale, cost_usd, duration_ms) -> None`
  - `list_eval_runs() -> list[dict]` (newest first, each row includes all eval_runs columns)
  - `eval_results(eval_run_id) -> list[dict]`
- DDL added to the existing `_migrate`/schema block (match how `runs`/`spans` are created — `CREATE TABLE IF NOT EXISTS`):

```sql
CREATE TABLE IF NOT EXISTS eval_runs (
  id TEXT PRIMARY KEY,
  created_at REAL NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  git_sha TEXT NOT NULL DEFAULT '',
  config_hash TEXT NOT NULL DEFAULT '',
  config_json TEXT NOT NULL DEFAULT '{}',
  questions_total INTEGER NOT NULL DEFAULT 0,
  tier1_scorable INTEGER NOT NULL DEFAULT 0,
  tier1_passed INTEGER NOT NULL DEFAULT 0,
  judge_avg REAL,
  cost_usd REAL NOT NULL DEFAULT 0,
  duration_ms INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS eval_results (
  eval_run_id TEXT NOT NULL,
  question_id TEXT NOT NULL,
  run_id TEXT NOT NULL DEFAULT '',
  dataset TEXT NOT NULL DEFAULT '',
  tags TEXT NOT NULL DEFAULT '[]',
  tier1_scorable INTEGER NOT NULL,
  tier1_passed INTEGER NOT NULL,
  tier1_detail TEXT NOT NULL DEFAULT '',
  judge TEXT,
  judge_rationale TEXT NOT NULL DEFAULT '',
  cost_usd REAL NOT NULL DEFAULT 0,
  duration_ms INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (eval_run_id, question_id)
);
```

(`judge` column holds `JudgeTurn.model_dump_json()` or NULL when tier 2 didn't run; `tags` holds a JSON list.)

- [ ] **Step 1: Write the failing test** (append to `backend/tests/test_store.py`, matching its existing fixture style for a tmp-path `Store`)

```python
def test_eval_run_roundtrip(tmp_path):
    st = Store(tmp_path / "t.db")
    st.add_eval_run(id="ev1", created_at=1000.0, label="baseline", git_sha="abc1234",
                    config_hash="deadbeef", config_json='{"analyst": "gpt-4o-mini"}',
                    questions_total=33, tier1_scorable=30, tier1_passed=27,
                    judge_avg=4.1, cost_usd=0.42, duration_ms=120_000)
    st.add_eval_result(eval_run_id="ev1", question_id="taxi-001", run_id="r1",
                       dataset="taxi", tags_json='["aggregation"]', tier1_scorable=True,
                       tier1_passed=True, tier1_detail="matched", judge_json=None,
                       judge_rationale="", cost_usd=0.01, duration_ms=3000)
    runs = st.list_eval_runs()
    assert len(runs) == 1 and runs[0]["id"] == "ev1"
    assert runs[0]["tier1_passed"] == 27 and runs[0]["judge_avg"] == 4.1
    rows = st.eval_results("ev1")
    assert len(rows) == 1
    assert rows[0]["question_id"] == "taxi-001" and rows[0]["tier1_passed"] == 1
    assert rows[0]["judge"] is None


def test_eval_runs_ordered_newest_first(tmp_path):
    st = Store(tmp_path / "t.db")
    for i, ts in enumerate([100.0, 300.0, 200.0]):
        st.add_eval_run(id=f"ev{i}", created_at=ts, label="", git_sha="", config_hash="",
                        config_json="{}", questions_total=0, tier1_scorable=0,
                        tier1_passed=0, judge_avg=None, cost_usd=0, duration_ms=0)
    assert [r["id"] for r in st.list_eval_runs()] == ["ev1", "ev2", "ev0"]
```

- [ ] **Step 2: Run to verify it fails** → `cd backend && .venv/bin/pytest tests/test_store.py -v` → new tests FAIL.

- [ ] **Step 3: Implement** — add the DDL above to the schema creation, plus:

```python
def add_eval_run(self, *, id: str, created_at: float, label: str, git_sha: str,
                 config_hash: str, config_json: str, questions_total: int,
                 tier1_scorable: int, tier1_passed: int, judge_avg: float | None,
                 cost_usd: float, duration_ms: int) -> None:
    with self._conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO eval_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, created_at, label, git_sha, config_hash, config_json, questions_total,
             tier1_scorable, tier1_passed, judge_avg, cost_usd, duration_ms))


def add_eval_result(self, *, eval_run_id: str, question_id: str, run_id: str,
                    dataset: str, tags_json: str, tier1_scorable: bool,
                    tier1_passed: bool, tier1_detail: str, judge_json: str | None,
                    judge_rationale: str, cost_usd: float, duration_ms: int) -> None:
    with self._conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO eval_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (eval_run_id, question_id, run_id, dataset, tags_json,
             int(tier1_scorable), int(tier1_passed), tier1_detail, judge_json,
             judge_rationale, cost_usd, duration_ms))


def list_eval_runs(self) -> list[dict]:
    with self._conn() as c:
        rows = c.execute("SELECT * FROM eval_runs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def eval_results(self, eval_run_id: str) -> list[dict]:
    with self._conn() as c:
        rows = c.execute(
            "SELECT * FROM eval_results WHERE eval_run_id = ? ORDER BY question_id",
            (eval_run_id,)).fetchall()
    return [dict(r) for r in rows]
```

Implementer note: match `store.py`'s ACTUAL connection/row-factory idiom (it may use a persistent `self.conn` rather than a `_conn()` contextmanager — mirror whatever `list_runs()` does, including its dict conversion).

- [ ] **Step 4: Run tests** → `cd backend && .venv/bin/pytest tests/test_store.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/tracing/store.py backend/tests/test_store.py
git commit -m "feat(m4): eval_runs + eval_results tables in the trace store"
```

---

### Task 7: Harness + `python -m app.evals run`

**Files:**
- Create: `backend/app/evals/harness.py`
- Modify: `backend/app/evals/__main__.py` (add `run` subcommand)
- Test: `backend/tests/test_harness.py`

**Interfaces:**
- Consumes: `load_golden`, `score_tier1`, `judge_answer`/`JudgeFn`/`DIMENSIONS`, `Store` eval methods, `execute_run`, `RunState`, `profile_dataframe` (find its real import — it lives where the upload route builds profiles, e.g. `app.api.upload` or a `profiling` module; import from there), `bus`, `settings`, `utc_midnight`.
- Produces: `run_eval(st: Store, golden_sets: list[GoldenDataset], deps_factory: Callable[[], GraphDeps], *, judge: JudgeFn | None = None, label: str = "", models: dict[str, str] | None = None, repo_root: Path = REPO_ROOT, enforce_budget: bool = True) -> str` (returns `eval_run_id`). `models` is only recorded into `config_json` in M4; M5 threads it into `GraphDeps.default`.
- Failure isolation: one question crashing (graph exception) records a failed result and continues.
- Cost per question: sum of `cost_usd` over `st.spans_for_run(run_id)` (requires the bus→store sink registered; the harness registers it, guarded by a module flag so double-registration can't happen).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_harness.py
import json

import pandas as pd

from app.agents.llm import GraphDeps, LLMUsage
from app.agents.schemas import AnalystTurn, CriticTurn, JudgeTurn, PlannerTurn
from app.evals.golden import GoldenDataset, GoldenExpected, GoldenQuestion
from app.evals.harness import run_eval
from app.runtime.reconcile import CriticFinding
from app.runtime.state import Claim, PlanStep
from app.tracing.store import Store

U = LLMUsage(tokens_in=10, tokens_out=5)


def _claim(value):
    return Claim(id="c1", step_id=1, text=f"answer is {value}", kind="numeric",
                 value=value, direction="", significant=None, methodology=None)


def _deps(answer_value):
    def planner(_msgs):
        return PlannerTurn(steps=[PlanStep(id=1, description="count rows",
                                           method="descriptive")],
                           rationale="one step"), U

    def analyst(_msgs):
        return AnalystTurn(action="finish", code="", findings=f"value {answer_value}",
                           claims=[_claim(answer_value)]), U

    def critic(_msgs):
        return CriticTurn(action="finish", code="",
                          findings=[CriticFinding(claim_id="c1", could_not_verify=False,
                                                  value=answer_value, direction="",
                                                  significant=None, methodology_ok=True,
                                                  notes="")]), U

    return GraphDeps(planner=planner, analyst_turn=analyst, critic_turn=critic,
                     compose=lambda _m: (f"The answer is {answer_value}.", U),
                     run_code=lambda code, ws: None)


def _golden(tmp_path, expected_value):
    csv = tmp_path / "tiny.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(csv, index=False)
    return [GoldenDataset(name="tiny", csv=csv.name, questions=[
        GoldenQuestion(id="tiny-001", question="How many rows?",
                       expected=GoldenExpected(kind="numeric", value=expected_value,
                                               tolerance=0.0)),
    ])]


def test_harness_scores_pass_and_fail(tmp_path):
    st = Store(tmp_path / "t.db")
    eval_id = run_eval(st, _golden(tmp_path, 3), lambda: _deps(3),
                       label="stub", repo_root=tmp_path, enforce_budget=False)
    run = st.list_eval_runs()[0]
    assert run["id"] == eval_id
    assert run["questions_total"] == 1
    assert run["tier1_scorable"] == 1 and run["tier1_passed"] == 1
    assert run["judge_avg"] is None

    eval_id2 = run_eval(st, _golden(tmp_path, 3), lambda: _deps(99),
                        label="wrong", repo_root=tmp_path, enforce_budget=False)
    wrong = [r for r in st.list_eval_runs() if r["id"] == eval_id2][0]
    assert wrong["tier1_passed"] == 0


def test_harness_records_judge_scores(tmp_path):
    st = Store(tmp_path / "t.db")

    def judge(_msgs):
        return JudgeTurn(clarity=4, uncertainty_honesty=4, chart_appropriateness=3,
                         methodological_soundness=5, rationale="fine"), U

    eval_id = run_eval(st, _golden(tmp_path, 3), lambda: _deps(3), judge=judge,
                       repo_root=tmp_path, enforce_budget=False)
    run = [r for r in st.list_eval_runs() if r["id"] == eval_id][0]
    assert run["judge_avg"] == 4.0
    row = st.eval_results(eval_id)[0]
    assert json.loads(row["judge"])["clarity"] == 4


def test_harness_survives_a_crashing_question(tmp_path):
    st = Store(tmp_path / "t.db")

    def exploding_deps():
        d = _deps(3)
        def bad_planner(_msgs):
            raise RuntimeError("boom")
        return GraphDeps(planner=bad_planner, analyst_turn=d.analyst_turn,
                         critic_turn=d.critic_turn, compose=d.compose,
                         run_code=d.run_code)

    eval_id = run_eval(st, _golden(tmp_path, 3), exploding_deps,
                       repo_root=tmp_path, enforce_budget=False)
    row = st.eval_results(eval_id)[0]
    assert row["tier1_passed"] == 0
    assert "boom" in row["tier1_detail"] or "failed" in row["tier1_detail"].lower()
```

Implementer note: `CriticFinding`'s exact fields live in `app/runtime/reconcile.py` — copy the construction style from `test_graph.py`'s `verifying_critic` stub rather than trusting the field list above. Same for whether the critic stub must echo values from a "Claims to verify:" block; if the plain `critic` stub above yields `unverified` claims, that's fine for these tests (tier-1 scoring doesn't require verified status), but prefer reusing `test_graph.py`'s helper stubs if importable.

- [ ] **Step 2: Run to verify it fails** → `cd backend && .venv/bin/pytest tests/test_harness.py -v` → FAIL.

- [ ] **Step 3: Implement harness + CLI**

```python
# backend/app/evals/harness.py
"""Run the golden set through the graph, score both tiers, persist to the store."""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Callable

import pandas as pd

from app.agents.llm import GraphDeps
from app.config import settings
from app.evals.golden import GOLDEN_DIR, REPO_ROOT, GoldenDataset, load_golden
from app.evals.judge import DIMENSIONS, JudgeFn, judge_answer
from app.evals.scoring import score_tier1
from app.runtime.events import bus
from app.runtime.graph import execute_run
from app.runtime.state import RunState
from app.tracing.store import Store, utc_midnight

_sink_registered = False


def _ensure_sink(st: Store) -> None:
    global _sink_registered
    if not _sink_registered:
        bus.add_sink(st.add_span)
        _sink_registered = True


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def config_snapshot(models: dict[str, str] | None) -> tuple[str, str]:
    cfg = settings()
    snap = {
        "models": models or {r: cfg.model_for(r)
                             for r in ("planner", "analyst", "critic", "composer")},
        "alpha": cfg.alpha,
        "numeric_rel_tolerance": cfg.numeric_rel_tolerance,
        "cheap_mode": cfg.cheap_mode,
    }
    blob = json.dumps(snap, sort_keys=True)
    return blob, sha256(blob.encode()).hexdigest()[:12]


def _profile(df: pd.DataFrame) -> dict:
    from app.api.datasets import profile_dataframe  # noqa: PLC0415 — real location may differ
    return profile_dataframe(df)


def run_eval(st: Store, golden_sets: list[GoldenDataset],
             deps_factory: Callable[[], GraphDeps], *,
             judge: JudgeFn | None = None, label: str = "",
             models: dict[str, str] | None = None,
             repo_root: Path = REPO_ROOT, enforce_budget: bool = True) -> str:
    _ensure_sink(st)
    eval_run_id = uuid.uuid4().hex[:12]
    config_json, config_hash = config_snapshot(models)
    t_eval = time.time()
    scorable = passed = 0
    judge_totals: list[float] = []
    total_cost = 0.0

    for gs in golden_sets:
        csv_path = repo_root / gs.csv
        df = pd.read_csv(csv_path)
        profile = _profile(df)
        existing = [d for d in st.list_datasets() if d["name"] == gs.name]
        dataset_id = existing[0]["id"] if existing else st.add_dataset(
            gs.name, str(csv_path), profile)

        for q in gs.questions:
            if enforce_budget and st.cost_since(utc_midnight()) >= settings().daily_budget_usd:
                raise RuntimeError("daily budget exhausted — aborting eval run")
            run_id = st.create_run(dataset_id, q.question)
            state = RunState(run_id=run_id, question=q.question,
                             dataset_path=str(csv_path), dataset_profile=profile)
            t0 = time.time()
            final = None
            try:
                out = execute_run(state, deps_factory())
                final = out.final if hasattr(out, "final") else out.get("final")
                answer = out.final_answer if hasattr(out, "final_answer") \
                    else out.get("final_answer", "")
                st.finish_run(run_id, answer,
                              result=final.model_dump_json() if final else "")
                detail_prefix = ""
            except Exception as exc:  # noqa: BLE001 — one bad question must not kill the sweep
                st.finish_run(run_id, "", status="error")
                detail_prefix = f"run crashed: {exc} — "
            duration_ms = int((time.time() - t0) * 1000)
            cost = sum(s["cost_usd"] for s in st.spans_for_run(run_id))
            total_cost += cost

            tier1 = score_tier1(q.expected, final)
            if detail_prefix:
                tier1.detail = detail_prefix + tier1.detail
            scorable += int(tier1.scorable)
            passed += int(tier1.scorable and tier1.passed)

            judge_json = rationale = None
            if judge is not None and final is not None:
                turn, _usage = judge_answer(q.question, final, judge)
                judge_json = turn.model_dump_json()
                rationale = turn.rationale
                judge_totals.append(
                    sum(getattr(turn, d) for d in DIMENSIONS) / len(DIMENSIONS))

            st.add_eval_result(
                eval_run_id=eval_run_id, question_id=q.id, run_id=run_id,
                dataset=gs.name, tags_json=json.dumps(q.tags),
                tier1_scorable=tier1.scorable, tier1_passed=tier1.passed,
                tier1_detail=tier1.detail, judge_json=judge_json,
                judge_rationale=rationale or "", cost_usd=cost, duration_ms=duration_ms)

    st.add_eval_run(
        id=eval_run_id, created_at=t_eval, label=label, git_sha=git_sha(),
        config_hash=config_hash, config_json=config_json,
        questions_total=sum(len(g.questions) for g in golden_sets),
        tier1_scorable=scorable, tier1_passed=passed,
        judge_avg=(sum(judge_totals) / len(judge_totals)) if judge_totals else None,
        cost_usd=total_cost, duration_ms=int((time.time() - t_eval) * 1000))
    return eval_run_id
```

Implementer notes:
- `execute_run`'s return type: the report says `_execute` in `app/api/runs.py` does `final.final.model_dump_json()` — read that function and mirror it exactly (drop the `hasattr` dance in favor of whatever the real shape is; also mirror how `_execute` handles `finish_run` args).
- `profile_dataframe`'s real module: grep for it; adjust the import.
- `spans_for_run` row access: mirror the dict/Row shape `store.py` actually returns.

Extend `__main__.py`:

```python
# backend/app/evals/__main__.py — replace the file
"""CLI: python -m app.evals <subcommand>."""
import argparse
import json
import sys

from app.evals.derivations import write_golden
from app.evals.golden import GOLDEN_DIR, load_golden


def _cmd_run(args) -> int:
    from app.agents.llm import GraphDeps
    from app.deps import store
    from app.evals.harness import run_eval
    from app.evals.judge import real_judge

    golden = load_golden(GOLDEN_DIR)
    if args.datasets:
        keep = set(args.datasets.split(","))
        golden = [g for g in golden if g.name in keep]
    st = store()
    eval_id = run_eval(st, golden, GraphDeps.default,
                       judge=real_judge() if args.judge else None,
                       label=args.label)
    run = next(r for r in st.list_eval_runs() if r["id"] == eval_id)
    rate = run["tier1_passed"] / max(run["tier1_scorable"], 1)
    print(f"eval {eval_id}: tier1 {run['tier1_passed']}/{run['tier1_scorable']} "
          f"({rate:.0%}), judge_avg={run['judge_avg']}, cost=${run['cost_usd']:.3f}")
    if args.gate:
        baseline = json.loads((GOLDEN_DIR.parent / "baseline.json").read_text())
        floor = baseline["tier1_pass_rate"] - args.gate_margin
        if rate < floor:
            print(f"GATE FAIL: pass rate {rate:.2%} < floor {floor:.2%}")
            return 1
        print(f"gate ok (floor {floor:.2%})")
    if args.write_baseline:
        (GOLDEN_DIR.parent / "baseline.json").write_text(
            json.dumps({"tier1_pass_rate": round(rate, 4), "eval_run_id": eval_id},
                       indent=2))
        print("baseline.json updated")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.evals")
    sub = parser.add_subparsers(dest="cmd", required=True)

    golden = sub.add_parser("golden")
    golden.add_argument("--write", action="store_true")

    run = sub.add_parser("run", help="run the golden set live (spends API budget)")
    run.add_argument("--label", default="")
    run.add_argument("--judge", action="store_true", help="also run the tier-2 judge")
    run.add_argument("--datasets", default="", help="comma list, e.g. taxi,weather")
    run.add_argument("--gate", action="store_true",
                     help="exit 1 if pass rate drops below baseline - margin")
    run.add_argument("--gate-margin", type=float, default=0.05)
    run.add_argument("--write-baseline", action="store_true")

    args = parser.parse_args()
    if args.cmd == "golden":
        if args.write:
            write_golden()
            print("golden YAMLs updated from derivations")
        return
    if args.cmd == "run":
        sys.exit(_cmd_run(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests** → `cd backend && .venv/bin/pytest tests/test_harness.py -v && .venv/bin/pytest` → PASS, suite green. Also `.venv/bin/python -m ruff check .` → clean.

- [ ] **Step 5: Commit**

```bash
git add backend/app/evals backend/tests/test_harness.py
git commit -m "feat(m4): eval harness — golden sweep, two-tier scoring, regression rows, run CLI"
```

---

### Task 8: Judge calibration — label template + agreement report

**Files:**
- Create: `backend/app/evals/calibration.py`
- Modify: `backend/app/evals/__main__.py` (add `label-template` and `calibration` subcommands)
- Test: `backend/tests/test_calibration.py`

**Interfaces:**
- Consumes: `Store.eval_results`, `DIMENSIONS`, sklearn's `cohen_kappa_score`.
- Produces:
  - `label_template(st: Store, eval_run_id: str) -> str` — YAML text: `eval_run_id` + one entry per judged question with `question_id`, the judge's rationale as a comment aid, and the four dimensions set to `null` for the human to fill.
  - `load_labels(path: Path) -> dict` — parsed labels file.
  - `calibration_report(st: Store, labels: dict) -> dict` with shape:
    `{"available": True, "n": int, "eval_run_id": str, "dimensions": [{"dimension": str, "n": int, "exact_pct": float, "within1_pct": float, "kappa": float, "matrix": [[int]*5]*5}], "overall": {"exact_pct": ..., "within1_pct": ..., "kappa": ...}}`
    (matrix rows = human score 1..5, cols = judge score 1..5). Questions present in labels but missing judge scores are skipped.
  - Default labels location: `backend/app/evals/labels/human_labels.yaml` (module constant `LABELS_PATH`); `calibration_report` callers pass parsed labels, the API/CLI read `LABELS_PATH` if it exists.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_calibration.py
import json

import yaml

from app.evals.calibration import calibration_report, label_template
from app.tracing.store import Store


def _seed(st: Store) -> None:
    st.add_eval_run(id="ev1", created_at=1.0, label="", git_sha="", config_hash="",
                    config_json="{}", questions_total=3, tier1_scorable=2,
                    tier1_passed=2, judge_avg=4.0, cost_usd=0, duration_ms=0)
    judge = {"clarity": 4, "uncertainty_honesty": 5, "chart_appropriateness": 3,
             "methodological_soundness": 4, "rationale": "ok"}
    for qid, j in [("q1", judge), ("q2", {**judge, "clarity": 2}), ("q3", None)]:
        st.add_eval_result(eval_run_id="ev1", question_id=qid, run_id=f"r-{qid}",
                           dataset="taxi", tags_json="[]", tier1_scorable=True,
                           tier1_passed=True, tier1_detail="",
                           judge_json=json.dumps(j) if j else None,
                           judge_rationale="ok" if j else "", cost_usd=0, duration_ms=0)


def test_label_template_lists_judged_questions_with_null_scores(tmp_path):
    st = Store(tmp_path / "t.db")
    _seed(st)
    parsed = yaml.safe_load(label_template(st, "ev1"))
    assert parsed["eval_run_id"] == "ev1"
    ids = [entry["question_id"] for entry in parsed["labels"]]
    assert ids == ["q1", "q2"]  # q3 was never judged
    assert parsed["labels"][0]["clarity"] is None


def test_calibration_report_agreement_and_kappa(tmp_path):
    st = Store(tmp_path / "t.db")
    _seed(st)
    labels = {"eval_run_id": "ev1", "labels": [
        {"question_id": "q1", "clarity": 4, "uncertainty_honesty": 4,
         "chart_appropriateness": 3, "methodological_soundness": 4},
        {"question_id": "q2", "clarity": 2, "uncertainty_honesty": 5,
         "chart_appropriateness": 4, "methodological_soundness": 4},
    ]}
    report = calibration_report(st, labels)
    assert report["available"] and report["n"] == 2
    clarity = next(d for d in report["dimensions"] if d["dimension"] == "clarity")
    assert clarity["exact_pct"] == 100.0  # judge said 4 and 2; human said 4 and 2
    chart = next(d for d in report["dimensions"]
                 if d["dimension"] == "chart_appropriateness")
    assert chart["exact_pct"] == 50.0 and chart["within1_pct"] == 100.0
    assert clarity["matrix"][3][3] == 1  # human 4 / judge 4 bucket
    assert 0.0 <= report["overall"]["within1_pct"] <= 100.0
```

- [ ] **Step 2: Run to verify it fails** → `cd backend && .venv/bin/pytest tests/test_calibration.py -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# backend/app/evals/calibration.py
"""Judge-vs-human agreement: the table that makes the judge trustworthy."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from sklearn.metrics import cohen_kappa_score

from app.evals.judge import DIMENSIONS
from app.tracing.store import Store

LABELS_PATH = Path(__file__).parent / "labels" / "human_labels.yaml"


def label_template(st: Store, eval_run_id: str) -> str:
    entries = []
    for row in st.eval_results(eval_run_id):
        if not row["judge"]:
            continue
        entries.append({
            "question_id": row["question_id"],
            "judge_rationale": row["judge_rationale"],  # context for the labeler; ignored on load
            **{d: None for d in DIMENSIONS},
        })
    return yaml.safe_dump({"eval_run_id": eval_run_id, "labels": entries},
                          sort_keys=False, allow_unicode=True)


def load_labels(path: Path = LABELS_PATH) -> dict | None:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def calibration_report(st: Store, labels: dict) -> dict:
    judged = {r["question_id"]: json.loads(r["judge"])
              for r in st.eval_results(labels["eval_run_id"]) if r["judge"]}
    pairs: dict[str, list[tuple[int, int]]] = {d: [] for d in DIMENSIONS}
    n = 0
    for entry in labels.get("labels", []):
        qid = entry.get("question_id")
        if qid not in judged or any(entry.get(d) is None for d in DIMENSIONS):
            continue
        n += 1
        for d in DIMENSIONS:
            pairs[d].append((int(entry[d]), int(judged[qid][d])))

    if n == 0:
        return {"available": False, "n": 0, "eval_run_id": labels.get("eval_run_id", ""),
                "dimensions": [], "overall": None}

    def stats(vals: list[tuple[int, int]]) -> dict:
        human = [h for h, _ in vals]
        judge = [j for _, j in vals]
        exact = sum(h == j for h, j in vals) / len(vals) * 100
        within1 = sum(abs(h - j) <= 1 for h, j in vals) / len(vals) * 100
        kappa = 0.0 if len(set(human)) < 2 or len(set(judge)) < 2 else float(
            cohen_kappa_score(human, judge, labels=[1, 2, 3, 4, 5]))
        matrix = [[0] * 5 for _ in range(5)]
        for h, j in vals:
            matrix[h - 1][j - 1] += 1
        return {"n": len(vals), "exact_pct": round(exact, 1),
                "within1_pct": round(within1, 1), "kappa": round(kappa, 3),
                "matrix": matrix}

    dimensions = [{"dimension": d, **stats(pairs[d])} for d in DIMENSIONS]
    pooled = [p for d in DIMENSIONS for p in pairs[d]]
    overall = stats(pooled)
    overall.pop("matrix")
    return {"available": True, "n": n, "eval_run_id": labels["eval_run_id"],
            "dimensions": dimensions, "overall": overall}
```

Add to `__main__.py` (new subparsers + dispatch):

```python
    tmpl = sub.add_parser("label-template",
                          help="print a YAML labeling template for an eval run")
    tmpl.add_argument("eval_run_id")

    sub.add_parser("calibration", help="print the calibration report as JSON")
```

```python
    if args.cmd == "label-template":
        from app.deps import store
        from app.evals.calibration import label_template
        print(label_template(store(), args.eval_run_id))
        return
    if args.cmd == "calibration":
        from app.deps import store
        from app.evals.calibration import calibration_report, load_labels
        labels = load_labels()
        if labels is None:
            print("no labels file at backend/app/evals/labels/human_labels.yaml")
            sys.exit(1)
        print(json.dumps(calibration_report(store(), labels), indent=2))
        return
```

Also create `backend/app/evals/labels/.gitkeep` (empty) so the directory exists.

- [ ] **Step 4: Run tests** → `cd backend && .venv/bin/pytest tests/test_calibration.py -v && .venv/bin/pytest` → PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/evals backend/tests/test_calibration.py
git commit -m "feat(m4): judge calibration — label template, agreement %, within-1, Cohen's kappa"
```

---

### Task 9: Evals API routes

**Files:**
- Create: `backend/app/api/evals.py`
- Modify: `backend/app/main.py` (include the router)
- Test: `backend/tests/test_api_evals.py`

**Interfaces:**
- Produces:
  - `GET /api/evals` → `list[dict]`: each eval_runs row plus computed `tier1_pass_rate: float | None` and `config: dict` (parsed from `config_json`; raw `config_json` dropped).
  - `GET /api/evals/calibration` → the `calibration_report` dict, or `{"available": False, "n": 0, "eval_run_id": "", "dimensions": [], "overall": None}` when no labels file exists. (Registered BEFORE the `/{eval_run_id}` route — FastAPI matches in order.)
  - `GET /api/evals/{eval_run_id}` → `{"run": <summary dict>, "results": <eval_results rows with judge parsed to dict|None, tags parsed to list>}`; 404 if unknown id.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_api_evals.py
import json

from fastapi.testclient import TestClient

from app.main import app


def _seed(st):
    st.add_eval_run(id="ev1", created_at=1000.0, label="baseline", git_sha="abc",
                    config_hash="h", config_json='{"models": {"analyst": "gpt-4o-mini"}}',
                    questions_total=2, tier1_scorable=2, tier1_passed=1,
                    judge_avg=3.5, cost_usd=0.1, duration_ms=5000)
    st.add_eval_result(eval_run_id="ev1", question_id="taxi-001", run_id="r1",
                       dataset="taxi", tags_json='["aggregation"]', tier1_scorable=True,
                       tier1_passed=True, tier1_detail="matched",
                       judge_json=json.dumps({"clarity": 4, "uncertainty_honesty": 4,
                                              "chart_appropriateness": 4,
                                              "methodological_soundness": 4,
                                              "rationale": "ok"}),
                       judge_rationale="ok", cost_usd=0.05, duration_ms=2500)


def test_evals_endpoints(tmp_path, monkeypatch):
    # copy the env/cache-clear setup from test_api_budget.py verbatim (DB_PATH etc.)
    from app import deps
    from app.config import settings
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    settings.cache_clear()
    deps.store.cache_clear()
    _seed(deps.store())

    client = TestClient(app)
    runs = client.get("/api/evals").json()
    assert runs[0]["id"] == "ev1"
    assert runs[0]["tier1_pass_rate"] == 0.5
    assert runs[0]["config"]["models"]["analyst"] == "gpt-4o-mini"

    detail = client.get("/api/evals/ev1").json()
    assert detail["run"]["id"] == "ev1"
    assert detail["results"][0]["judge"]["clarity"] == 4
    assert detail["results"][0]["tags"] == ["aggregation"]

    assert client.get("/api/evals/nope").status_code == 404

    cal = client.get("/api/evals/calibration").json()
    assert cal["available"] is False  # no labels file in test env
    settings.cache_clear()
    deps.store.cache_clear()
```

Implementer note: `test_api_budget.py` already solves store/settings isolation for `TestClient` — copy its exact fixture/monkeypatch pattern (including any `store_dep` naming difference) instead of the sketch above.

- [ ] **Step 2: Run to verify it fails** → `cd backend && .venv/bin/pytest tests/test_api_evals.py -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# backend/app/api/evals.py
"""Read-only eval endpoints for the Evals screen."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from app.deps import store
from app.evals.calibration import calibration_report, load_labels

router = APIRouter(prefix="/api/evals", tags=["evals"])

_EMPTY_CALIBRATION = {"available": False, "n": 0, "eval_run_id": "",
                      "dimensions": [], "overall": None}


def _summary(row: dict) -> dict:
    out = dict(row)
    out["config"] = json.loads(out.pop("config_json") or "{}")
    out["tier1_pass_rate"] = (
        row["tier1_passed"] / row["tier1_scorable"] if row["tier1_scorable"] else None)
    return out


@router.get("")
def list_evals() -> list[dict]:
    return [_summary(r) for r in store().list_eval_runs()]


@router.get("/calibration")
def calibration() -> dict:
    labels = load_labels()
    if labels is None:
        return _EMPTY_CALIBRATION
    return calibration_report(store(), labels)


@router.get("/{eval_run_id}")
def eval_detail(eval_run_id: str) -> dict:
    runs = [r for r in store().list_eval_runs() if r["id"] == eval_run_id]
    if not runs:
        raise HTTPException(status_code=404, detail="eval run not found")
    results = []
    for r in store().eval_results(eval_run_id):
        row = dict(r)
        row["judge"] = json.loads(row["judge"]) if row["judge"] else None
        row["tags"] = json.loads(row["tags"] or "[]")
        results.append(row)
    return {"run": _summary(runs[0]), "results": results}
```

In `backend/app/main.py`, import and include next to the existing routers: `app.include_router(evals.router)`.

- [ ] **Step 4: Run tests** → `cd backend && .venv/bin/pytest tests/test_api_evals.py -v && .venv/bin/pytest` → PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/evals.py backend/app/main.py backend/tests/test_api_evals.py
git commit -m "feat(m4): evals API — runs list, detail, calibration report"
```

---

### Task 10: Frontend — Evals screen (score-over-time + runs grid)

**Files:**
- Modify: `frontend/src/lib/types.ts`, `frontend/src/lib/api.ts`
- Create: `frontend/src/pages/EvalsScreen.tsx`
- Modify: `frontend/src/App.tsx` (route `/evals` + nav button)

**Interfaces:**
- Consumes: Task 9's endpoints; existing `json<T>` fetch helper; MUI patterns (`Container maxWidth="lg" sx={{py:4}}` → `Stack spacing={2}`; `Paper variant="outlined" sx={{p:2}}` cards; DataGrid conventions from `RunsDashboard.tsx`).
- Produces TS types used by Task 11's `CalibrationGrid`:

```ts
export interface JudgeScores {
  clarity: number;
  uncertainty_honesty: number;
  chart_appropriateness: number;
  methodological_soundness: number;
  rationale: string;
}
export interface EvalRunSummary {
  id: string;
  created_at: number;
  label: string;
  git_sha: string;
  config_hash: string;
  config: Record<string, unknown>;
  questions_total: number;
  tier1_scorable: number;
  tier1_passed: number;
  tier1_pass_rate: number | null;
  judge_avg: number | null;
  cost_usd: number;
  duration_ms: number;
}
export interface EvalResultRow {
  eval_run_id: string;
  question_id: string;
  run_id: string;
  dataset: string;
  tags: string[];
  tier1_scorable: number;
  tier1_passed: number;
  tier1_detail: string;
  judge: JudgeScores | null;
  judge_rationale: string;
  cost_usd: number;
  duration_ms: number;
}
export interface CalibrationDimension {
  dimension: string;
  n: number;
  exact_pct: number;
  within1_pct: number;
  kappa: number;
  matrix: number[][];
}
export interface CalibrationReport {
  available: boolean;
  n: number;
  eval_run_id: string;
  dimensions: CalibrationDimension[];
  overall: { n: number; exact_pct: number; within1_pct: number; kappa: number } | null;
}
```

- API functions in `api.ts` (same one-liner style):

```ts
export const listEvalRuns = async (): Promise<EvalRunSummary[]> =>
  json(await fetch("/api/evals"));
export const getEvalRun = async (
  id: string,
): Promise<{ run: EvalRunSummary; results: EvalResultRow[] }> =>
  json(await fetch(`/api/evals/${id}`));
export const getCalibration = async (): Promise<CalibrationReport> =>
  json(await fetch("/api/evals/calibration"));
```

- [ ] **Step 1: Add types + api functions** (code above; imports adjusted to the file's actual export style).

- [ ] **Step 2: Build the page**

```tsx
// frontend/src/pages/EvalsScreen.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Alert, Chip, Container, Paper, Stack, Typography } from "@mui/material";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";
import { LineChart } from "@mui/x-charts/LineChart";
import { getCalibration, getEvalRun, listEvalRuns } from "../lib/api";
import type { EvalResultRow, EvalRunSummary } from "../lib/types";
import { CalibrationGrid } from "../components/CalibrationGrid";

const runColumns: GridColDef<EvalRunSummary>[] = [
  {
    field: "created_at", headerName: "When", width: 170,
    valueFormatter: (value: number) => new Date(value * 1000).toLocaleString(),
  },
  { field: "label", headerName: "Label", width: 140 },
  { field: "git_sha", headerName: "SHA", width: 90 },
  { field: "config_hash", headerName: "Config", width: 110 },
  {
    field: "tier1_pass_rate", headerName: "Tier 1", width: 100,
    valueFormatter: (value: number | null) =>
      value == null ? "—" : `${(value * 100).toFixed(0)}%`,
  },
  {
    field: "judge_avg", headerName: "Judge", width: 90,
    valueFormatter: (value: number | null) => (value == null ? "—" : value.toFixed(2)),
  },
  {
    field: "cost_usd", headerName: "Cost", width: 90,
    valueFormatter: (value: number) => `$${value.toFixed(3)}`,
  },
  {
    field: "duration_ms", headerName: "Duration", width: 100,
    valueFormatter: (value: number) => `${(value / 1000).toFixed(0)}s`,
  },
];

const resultColumns: GridColDef<EvalResultRow>[] = [
  { field: "question_id", headerName: "Question", width: 130 },
  { field: "dataset", headerName: "Dataset", width: 100 },
  {
    field: "tier1_passed", headerName: "Tier 1", width: 110,
    renderCell: ({ row }) =>
      row.tier1_scorable ? (
        <Chip size="small" variant="outlined"
          color={row.tier1_passed ? "success" : "error"}
          label={row.tier1_passed ? "pass" : "fail"} />
      ) : (
        <Chip size="small" variant="outlined" label="judge-only" />
      ),
  },
  {
    field: "judge", headerName: "Judge avg", width: 100,
    valueGetter: (_value, row) =>
      row.judge
        ? ((row.judge.clarity + row.judge.uncertainty_honesty +
            row.judge.chart_appropriateness + row.judge.methodological_soundness) / 4
          ).toFixed(2)
        : "—",
  },
  { field: "tier1_detail", headerName: "Detail", flex: 1, minWidth: 220 },
];

export function EvalsScreen() {
  const runs = useQuery({ queryKey: ["evalRuns"], queryFn: listEvalRuns });
  const calibration = useQuery({ queryKey: ["calibration"], queryFn: getCalibration });
  const [selected, setSelected] = useState<string | null>(null);
  const detail = useQuery({
    queryKey: ["evalRun", selected],
    queryFn: () => getEvalRun(selected as string),
    enabled: selected != null,
  });

  const series = [...(runs.data ?? [])].sort((a, b) => a.created_at - b.created_at);

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Stack spacing={2}>
        <Typography variant="h5" fontWeight={700}>Evals</Typography>
        {runs.error != null && <Alert severity="error">{String(runs.error)}</Alert>}

        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>Score over time</Typography>
          {series.length === 0 ? (
            <Typography color="text.secondary">
              No eval runs yet — run `python -m app.evals run` in the backend.
            </Typography>
          ) : (
            <LineChart
              height={280}
              xAxis={[{
                scaleType: "point",
                data: series.map((r) =>
                  `${new Date(r.created_at * 1000).toLocaleDateString()} ${r.git_sha}`),
              }]}
              yAxis={[
                { id: "pct", min: 0, max: 100, label: "tier-1 pass %" },
                { id: "judge", min: 1, max: 5, position: "right", label: "judge avg" },
              ]}
              series={[
                {
                  yAxisId: "pct", label: "tier-1 pass %",
                  data: series.map((r) =>
                    r.tier1_pass_rate == null ? null : r.tier1_pass_rate * 100),
                },
                {
                  yAxisId: "judge", label: "judge avg (1-5)",
                  data: series.map((r) => r.judge_avg),
                },
              ]}
            />
          )}
        </Paper>

        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>Eval runs</Typography>
          <DataGrid
            rows={runs.data ?? []}
            columns={runColumns}
            loading={runs.isLoading}
            density="compact"
            autoHeight
            disableRowSelectionOnClick
            onRowClick={({ row }) => setSelected(row.id)}
            initialState={{ pagination: { paginationModel: { pageSize: 10 } } }}
            pageSizeOptions={[10, 25]}
          />
        </Paper>

        {selected != null && (
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Results — {selected}
            </Typography>
            <DataGrid
              rows={detail.data?.results ?? []}
              getRowId={(row) => row.question_id}
              columns={resultColumns}
              loading={detail.isLoading}
              density="compact"
              autoHeight
              disableRowSelectionOnClick
              initialState={{ pagination: { paginationModel: { pageSize: 25 } } }}
              pageSizeOptions={[25, 50]}
            />
          </Paper>
        )}

        <CalibrationGrid report={calibration.data} loading={calibration.isLoading} />
      </Stack>
    </Container>
  );
}
```

(`CalibrationGrid` is Task 11 — to keep this task compilable, create the file in this task with the real implementation from Task 11, or temporarily inline `const CalibrationGrid = () => null` is NOT allowed; do Task 11's component now if executing sequentially, it's small. If executing as subagents, merge Tasks 10+11 into one dispatch.)

- [ ] **Step 3: Wire route + nav in `App.tsx`**

Add `import { EvalsScreen } from "./pages/EvalsScreen";`, then inside `<Routes>`: `<Route path="/evals" element={<EvalsScreen />} />`, and in `Nav()` next to the Runs button: `<Button size="small" color="inherit" component={RouterLink} to="/evals">Evals</Button>`.

- [ ] **Step 4: Verify** → `cd frontend && npm run typecheck && npm run build` → clean. Device-checklist note (owner): visit `/evals` with backend up; empty states render without errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src
git commit -m "feat(m4): Evals screen — score-over-time chart + eval runs grid"
```

---

### Task 11: CalibrationGrid component

**Files:**
- Create: `frontend/src/components/CalibrationGrid.tsx`

**Interfaces:**
- Consumes: `CalibrationReport`, `CalibrationDimension` types from Task 10.
- Produces: `CalibrationGrid({ report, loading }: { report: CalibrationReport | undefined; loading: boolean })` — a Paper card: per-dimension table (n, exact %, within-1 %, kappa) + a 5×5 confusion mini-grid per dimension (human rows × judge cols, cell opacity ∝ count). MUI X heatmap is Pro-tier, so this is a hand-rolled Box grid (per the build plan).

- [ ] **Step 1: Implement**

```tsx
// frontend/src/components/CalibrationGrid.tsx
import { Box, Paper, Stack, Table, TableBody, TableCell, TableHead, TableRow,
  Tooltip, Typography } from "@mui/material";
import type { CalibrationDimension, CalibrationReport } from "../lib/types";

function Matrix({ dim }: { dim: CalibrationDimension }) {
  const max = Math.max(1, ...dim.matrix.flat());
  return (
    <Box>
      <Typography variant="caption" color="text.secondary">
        {dim.dimension} — human (rows) vs judge (cols), 1→5
      </Typography>
      <Box sx={{ display: "grid", gridTemplateColumns: "repeat(5, 22px)", gap: "2px", mt: 0.5 }}>
        {dim.matrix.flatMap((row, h) =>
          row.map((count, j) => (
            <Tooltip key={`${h}-${j}`} title={`human ${h + 1} / judge ${j + 1}: ${count}`}>
              <Box sx={{
                width: 22, height: 22, borderRadius: 0.5,
                bgcolor: count === 0 ? "action.hover" : "primary.main",
                opacity: count === 0 ? 1 : 0.25 + 0.75 * (count / max),
              }} />
            </Tooltip>
          )),
        )}
      </Box>
    </Box>
  );
}

export function CalibrationGrid({ report, loading }: {
  report: CalibrationReport | undefined;
  loading: boolean;
}) {
  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Typography variant="subtitle2" gutterBottom>Judge calibration</Typography>
      {loading && <Typography color="text.secondary">loading…</Typography>}
      {!loading && (!report || !report.available) && (
        <Typography color="text.secondary">
          No human labels yet. Generate a template with
          `python -m app.evals label-template &lt;eval_run_id&gt;`, hand-fill ~40 answers
          into backend/app/evals/labels/human_labels.yaml, and reload.
        </Typography>
      )}
      {!loading && report?.available && (
        <Stack spacing={2}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Dimension</TableCell>
                <TableCell align="right">n</TableCell>
                <TableCell align="right">Exact %</TableCell>
                <TableCell align="right">Within-1 %</TableCell>
                <TableCell align="right">Cohen&apos;s κ</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {report.dimensions.map((d) => (
                <TableRow key={d.dimension}>
                  <TableCell>{d.dimension.replaceAll("_", " ")}</TableCell>
                  <TableCell align="right">{d.n}</TableCell>
                  <TableCell align="right">{d.exact_pct.toFixed(1)}</TableCell>
                  <TableCell align="right">{d.within1_pct.toFixed(1)}</TableCell>
                  <TableCell align="right">{d.kappa.toFixed(3)}</TableCell>
                </TableRow>
              ))}
              {report.overall && (
                <TableRow>
                  <TableCell sx={{ fontWeight: 700 }}>overall (pooled)</TableCell>
                  <TableCell align="right">{report.overall.n}</TableCell>
                  <TableCell align="right">{report.overall.exact_pct.toFixed(1)}</TableCell>
                  <TableCell align="right">{report.overall.within1_pct.toFixed(1)}</TableCell>
                  <TableCell align="right">{report.overall.kappa.toFixed(3)}</TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
          <Stack direction="row" spacing={3} flexWrap="wrap" useFlexGap>
            {report.dimensions.map((d) => <Matrix key={d.dimension} dim={d} />)}
          </Stack>
        </Stack>
      )}
    </Paper>
  );
}
```

- [ ] **Step 2: Verify + commit**

Run: `cd frontend && npm run typecheck && npm run build` → clean.

```bash
git add frontend/src/components/CalibrationGrid.tsx
git commit -m "feat(m4): calibration grid — agreement table + confusion mini-heatmaps"
```

---

### Task 12: Complexity router — schema, prompt, deps, config, budget

**Files:**
- Modify: `backend/app/agents/schemas.py` (add `RouterTurn`)
- Create: `backend/app/agents/prompts/router.md`
- Modify: `backend/app/agents/llm.py` (add `RouterFn`, `router` field on `GraphDeps`, wire into `GraphDeps.default`)
- Modify: `backend/app/config.py` (add `router_model: str = "gpt-4o-mini"`, extend `model_for`)
- Modify: `backend/app/runtime/budget.py` (add `"router": 1` to `llm_caps`)
- Test: `backend/tests/test_llm.py` (extend)

**Interfaces:**
- Produces:
  - `RouterTurn(route: Literal["simple", "multi_step", "statistical"], reason: str)`
  - `RouterFn = Callable[[list[BaseMessage]], tuple[RouterTurn, LLMUsage]]`
  - `GraphDeps.router: RouterFn | None = None` — **None means "route everything multi_step"** (legacy behavior; keeps every existing test construction valid). `GraphDeps.default()` wires `_structured_fn("router", RouterTurn)`.
- The router is ALWAYS a mini model: even in the non-cheap branch, `model_for("router")` returns `router_model` (`gpt-4o-mini`) — a strong model for a 3-way classification is exactly the waste routing exists to remove; note this inline.

- [ ] **Step 1: Write the failing test** (append to `backend/tests/test_llm.py`)

```python
def test_router_turn_schema():
    from app.agents.schemas import RouterTurn
    turn = RouterTurn(route="simple", reason="single aggregation")
    assert turn.route == "simple"
    with pytest.raises(ValidationError):
        RouterTurn(route="hard", reason="x")


def test_graph_deps_router_defaults_to_none():
    from app.agents.llm import GraphDeps, LLMUsage
    from app.agents.schemas import AnalystTurn, CriticTurn, PlannerTurn  # noqa: F401
    u = LLMUsage()
    deps = GraphDeps(planner=lambda m: None, analyst_turn=lambda m: None,
                     critic_turn=lambda m: None, compose=lambda m: ("", u))
    assert deps.router is None
```

(Match the file's existing import style; it may already import these names at top level.)

- [ ] **Step 2: Run to verify it fails** → `cd backend && .venv/bin/pytest tests/test_llm.py -v` → FAIL.

- [ ] **Step 3: Implement**

`schemas.py`:

```python
class RouterTurn(BaseModel):
    """Complexity routing decision made at graph entry."""
    route: Literal["simple", "multi_step", "statistical"]
    reason: str = ""
```

```markdown
<!-- backend/app/agents/prompts/router.md -->
Classify the complexity of a data-analysis question about a tabular dataset.

Routes:
- simple: one aggregation/lookup a single pandas expression answers
  (count, mean, max, median, groupby-top-1). No hypothesis, no multiple sub-questions.
- multi_step: needs 2+ independent analysis steps, or comparison across several
  derived results, but no statistical testing.
- statistical: asks about significance, correlation, association, trend, prediction,
  distribution comparison, or anomaly detection — anything needing a statistical method
  and methodology review.

When unsure between simple and anything else, prefer the larger route — a wasted
planner call is better than an unplanned statistical answer.

Dataset profile:
{profile}

Question:
{question}

Return route and a one-sentence reason via the structured output schema.
```

`llm.py`: add `RouterFn = Callable[[list[BaseMessage]], tuple[RouterTurn, LLMUsage]]` beside the other aliases, `router: RouterFn | None = None` as the LAST field of `GraphDeps` (after `run_code`, so positional constructions stay valid), and in `GraphDeps.default()` pass `router=_structured_fn("router", RouterTurn)`.

`config.py`: add `router_model: str = "gpt-4o-mini"`; in `model_for`, return `router_model` for `"router"` in BOTH branches (router stays mini even when cheap_mode is off).

`budget.py`: add `"router": 1` to `llm_caps`.

- [ ] **Step 4: Run tests** → `cd backend && .venv/bin/pytest tests/test_llm.py tests/test_budget.py -v && .venv/bin/pytest` → PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/agents backend/app/config.py backend/app/runtime/budget.py backend/tests/test_llm.py
git commit -m "feat(m4): complexity router schema, prompt, GraphDeps field, budget cap"
```

---

### Task 13: Graph — router node, simple-route Send, conditional composer

**Files:**
- Modify: `backend/app/runtime/state.py` (add `route`, `route_reason` to `RunState`)
- Modify: `backend/app/runtime/graph.py` (router node + edges + composer folding)
- Test: `backend/tests/test_graph.py` (extend)

**Interfaces:**
- `RunState` gains: `route: str = ""` and `route_reason: str = ""`.
- New graph shape: `START → router`; `route_from_router(state)` conditional → `"planner"` or `[Send("analyst", task)]`; everything downstream unchanged.
- `router_node(state, deps) -> dict`: calls `deps.router` (or defaults to `multi_step` when `deps.router is None`), emits a `HANDOFF` event `agent="router"`, `payload={"route", "reason", "to"}` (`to` = `"analyst"` for simple else `"planner"`), and for `simple` also returns a synthetic single-step `Plan` so critic-retry and the run view keep working.
- Composer folding: when exactly one analyst result exists, it didn't fail, ALL verdicts are `verified`, and the planner didn't fail → skip `deps.compose`, build `FinalAnswer(narrative=<the result's findings>, ...)` directly, emit `HANDOFF` `agent="composer"`, `payload={"folded": True, "reason": "single verified finding"}`. Otherwise the existing compose path runs untouched.

- [ ] **Step 1: Write the failing tests** (append to `backend/tests/test_graph.py`, reusing its existing helpers `make_state`, stub builders, and `U`)

```python
def _router(route: str):
    def fn(_msgs):
        return RouterTurn(route=route, reason=f"stubbed {route}"), U
    return fn


def test_simple_route_skips_planner(dataset):
    planner_calls = []

    def spying_planner(msgs):
        planner_calls.append(msgs)
        return one_step_planner(msgs)  # reuse the file's existing single-step stub

    deps = GraphDeps(planner=spying_planner,
                     analyst_turn=finishing_analyst({1: ("3 rows.", [count_claim(3)])}),
                     critic_turn=verifying_critic,
                     compose=lambda m: ("unused", U),
                     router=_router("simple"))
    out = execute_run(make_state(dataset, "run-simple"), deps)
    assert not planner_calls  # planner LLM never called
    events = bus.history("run-simple")
    assert not [e for e in events if e.agent == "planner"]
    router_events = [e for e in events if e.agent == "router"]
    assert router_events and router_events[0].payload["route"] == "simple"
    assert out.final is not None and not out.final.failed


def test_simple_route_folds_composer(dataset):
    compose_calls = []

    def spying_compose(msgs):
        compose_calls.append(msgs)
        return ("unused", U)

    deps = GraphDeps(planner=one_step_planner,
                     analyst_turn=finishing_analyst({1: ("3 rows.", [count_claim(3)])}),
                     critic_turn=verifying_critic,
                     compose=spying_compose,
                     router=_router("simple"))
    out = execute_run(make_state(dataset, "run-folded"), deps)
    assert not compose_calls  # composer LLM skipped
    folded = [e for e in bus.history("run-folded")
              if e.agent == "composer" and e.payload.get("folded")]
    assert folded
    assert out.final.narrative  # narrative taken from the analyst's findings


def test_statistical_route_uses_planner(dataset):
    deps = GraphDeps(planner=one_step_planner,
                     analyst_turn=finishing_analyst({1: ("3 rows.", [count_claim(3)])}),
                     critic_turn=verifying_critic,
                     compose=lambda m: ("composed", U),
                     router=_router("statistical"))
    execute_run(make_state(dataset, "run-stat"), deps)
    events = bus.history("run-stat")
    assert [e for e in events if e.agent == "planner"]
    assert [e for e in events if e.agent == "router"][0].payload["route"] == "statistical"


def test_no_router_behaves_like_multi_step(dataset):
    deps = GraphDeps(planner=one_step_planner,
                     analyst_turn=finishing_analyst({1: ("3 rows.", [count_claim(3)])}),
                     critic_turn=verifying_critic,
                     compose=lambda m: ("composed", U))
    out = execute_run(make_state(dataset, "run-norouter"), deps)
    assert out.final is not None
    route_events = [e for e in bus.history("run-norouter") if e.agent == "router"]
    assert route_events and route_events[0].payload["route"] == "multi_step"


def test_multi_finding_run_still_uses_composer(dataset):
    # reuse the file's existing two-step fan-out fixtures; assert compose IS called
    compose_calls = []

    def spying_compose(msgs):
        compose_calls.append(msgs)
        return ("composed", U)

    deps = GraphDeps(planner=two_step_planner,
                     analyst_turn=finishing_analyst({1: ("A.", [count_claim(3)]),
                                                     2: ("B.", [count_claim(3)])}),
                     critic_turn=verifying_critic,
                     compose=spying_compose,
                     router=_router("multi_step"))
    execute_run(make_state(dataset, "run-multi"), deps)
    assert compose_calls
```

Implementer note: the helper names (`one_step_planner`, `two_step_planner`, `finishing_analyst`, `verifying_critic`, `count_claim`, `dataset` fixture) are the pattern from the existing `test_graph.py` — read that file first and reuse ITS actual helper names/signatures; add a tiny `count_claim`-style helper if one doesn't exist. Also verify how a claim gets `verified`: `verifying_critic` echoes claim values, so single-step runs with it produce all-verified verdicts — exactly what the folding tests need.

- [ ] **Step 2: Run to verify they fail** → `cd backend && .venv/bin/pytest tests/test_graph.py -v` → new tests FAIL.

- [ ] **Step 3: Implement**

`state.py`: add to `RunState`:

```python
route: str = ""
route_reason: str = ""
```

`graph.py` — add the router node (mirror `planner_node`'s budget/emit/prompt idioms exactly):

```python
def router_node(state: RunState, deps: GraphDeps) -> dict:
    started = time.time()
    if deps.router is None:
        route, reason, usage = "multi_step", "no router configured", None
    else:
        budget = AgentBudget.for_role("router")
        budget.charge_llm()  # mirror the real budget API used by planner_node
        prompt = _prompt("router", profile=json.dumps(state.dataset_profile),
                         question=state.question)
        turn, usage = deps.router([HumanMessage(content=prompt)])
        route, reason = turn.route, turn.reason
    _emit(state.run_id, "router", EventType.HANDOFF,
          {"route": route, "reason": reason,
           "to": "analyst" if route == "simple" else "planner"},
          started, usage, parent=state.root_span_id)
    out: dict = {"route": route, "route_reason": reason}
    if route == "simple":
        step = PlanStep(id=1, description=state.question, method="descriptive")
        out["plan"] = Plan(steps=[step], rationale=f"router: simple — {reason}")
    return out


def route_from_router(state: RunState):
    if state.route == "simple" and state.plan and state.plan.steps:
        step = state.plan.steps[0]
        return [Send("analyst", AnalystTask(
            run_id=state.run_id, question=state.question,
            dataset_path=state.dataset_path, dataset_profile=state.dataset_profile,
            root_span_id=state.root_span_id, step=step).model_dump())]
    return "planner"
```

Wire in `build_graph`:

```python
g.add_node("router", lambda s: router_node(s, deps))
g.add_edge(START, "router")
g.add_conditional_edges("router", route_from_router, ["planner", "analyst"])
# (delete the old g.add_edge(START, "planner"))
```

Composer folding — at the top of `composer_node`, after computing `results = _latest_results(state.analyst_results)` and the verdict merge (reuse the function's existing locals; insert before the `deps.compose` call):

```python
single = len(results) == 1 and not state.planner_failed
only = next(iter(results.values())) if single else None
all_verified = (
    single and only is not None and not only.failed and state.verdicts
    and all(v.status == "verified" for v in state.verdicts)
)
if all_verified:
    started = time.time()
    _emit(state.run_id, "composer", EventType.HANDOFF,
          {"folded": True, "reason": "single verified finding"},
          started, None, parent=state.root_span_id)
    final = FinalAnswer(narrative=only.findings, claims=verified_claims,
                        charts=charts, failed=False)
    return {"final_answer": only.findings, "final": final}
```

(`verified_claims` and `charts` are whatever locals the existing composer body builds before composing — reuse them, don't rebuild. Keep the honest-failure and multi-finding paths byte-identical.)

Also: `planner_node` guards. If `planner_node` asserts on being the first node or reads anything now produced by the router, adjust; expected to be unaffected.

- [ ] **Step 4: Run the full suite** → `cd backend && .venv/bin/pytest -x` → PASS (existing tests must stay green: `GraphDeps` without `router` now routes `multi_step` through the new router node, and `test_events_flow_for_all_agents`'s `"handoff" in types` assertion still holds). Run `.venv/bin/python -m ruff check .` → clean.

- [ ] **Step 5: Commit**

```bash
git add backend/app/runtime backend/tests/test_graph.py
git commit -m "feat(m4): adaptive routing — router node, planner skip for simple questions, composer folding"
```

---

### Task 14: Run view renders routed graph shapes

**Files:**
- Modify: `frontend/src/lib/graphModel.ts`
- Modify: `frontend/src/components/AgentGraph.tsx`
- Modify: `frontend/src/lib/types.ts` (extend `agent` union with `"router"`)

**Interfaces:**
- Consumes: `AgentEvent`s now including `agent: "router"` HANDOFF events with `payload.route` / `payload.reason` / `payload.to`, and composer HANDOFF events with `payload.folded`.
- Produces: `buildAgentGraph(events)` emits a `router` node when router events exist; the planner node is emitted ONLY when planner events exist (or when there are no router events at all — old recorded runs must render exactly as before); the entry edge is `router → planner` or `router → analyst-1` per `payload.to`; the composer node's `sublabel` shows `"folded"` when the folded HANDOFF is present. `AgentNodeModel.agent` union gains `"router"`.

- [ ] **Step 1: Implement `graphModel.ts` changes**

In `buildAgentGraph`:
- Detect `const routerEvents = events.filter((e) => e.agent === "router")`.
- If `routerEvents.length > 0`: unshift a node `{ id: "router", agent: "router", stepId: null, label: "router", sublabel: routeHandoff?.payload.route as string ?? "", status: rollupStatus(routerEvents, events) }` (reuse the existing per-agent status rollup helper).
- `const hasPlanner = events.some((e) => e.agent === "planner")`; only push the planner node when `hasPlanner || routerEvents.length === 0`.
- Entry edges: with a router, add `router → planner` when `hasPlanner`, else `router → analyst-<firstStepId>` (first analyst node id; falls back to `router → composer` if neither exists — a crashed run).
- Plan steps for analyst nodes: when the planner never ran (simple route), derive the single analyst node from the router HANDOFF's synthetic plan — the analyst events themselves carry `payload.step_id`; reuse the existing fallback that builds analyst nodes from analyst events when no plan event exists (if the current code has no such fallback, add one: group analyst events by `payload.step_id`, one node per step id, label from the run's question).
- Composer folding: `const folded = events.some((e) => e.agent === "composer" && e.type === "handoff" && Boolean(e.payload.folded))`; set the composer node's `sublabel` to `"folded"` when true.

- [ ] **Step 2: Implement `AgentGraph.tsx` changes**

Extend the column map: `const COLUMN_X = { router: 0, planner: 220, analyst: 460, critic: 700, composer: 940 }` and center the router vertically like planner/critic/composer. Old runs (no router node) simply leave column 0 empty — acceptable.

- [ ] **Step 3: Verify** → `cd frontend && npm run typecheck && npm run build` → clean.

Device checklist for the owner (hands-on QA — agents don't grind the browser):
1. Ask a simple question ("How many rows are there?") → run view shows router → analyst → critic → composer, NO planner node; router node sublabel says `simple`; composer shows `folded`.
2. Ask a statistical question → router → planner → analysts → critic → composer, full shape.
3. Open an old (pre-M4) run → graph renders exactly as before.

- [ ] **Step 4: Commit**

```bash
git add frontend/src
git commit -m "feat(m4): run view renders router node and per-route graph shapes"
```

---

### Task 15: CI workflow + baseline + live smoke run

**Files:**
- Create: `.github/workflows/ci.yml`
- Create (generated by CLI): `backend/app/evals/baseline.json`
- Modify: `MILESTONES.md` (check off M4 — do NOT git add)

**Interfaces:**
- Consumes: `python -m app.evals run --gate` / `--write-baseline` from Task 7; Makefile targets.
- Produces: CI with three jobs — `backend` (ruff + pytest), `frontend` (typecheck + build), `eval-gate` (programmatic tier only, CHEAP_MODE, runs only when the `OPENAI_API_KEY` secret exists; judge tier stays manual per the build plan).

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/ci.yml
name: ci
on:
  push:
    branches: [master]
  pull_request:

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install -e "backend[dev]"
      - run: ruff check backend
      - run: pytest backend/tests
        env: {PYTHONPATH: backend}
        working-directory: .

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: {node-version: 22, cache: npm, cache-dependency-path: frontend/package-lock.json}
      - run: npm ci
        working-directory: frontend
      - run: npm run typecheck && npm run build
        working-directory: frontend

  eval-gate:
    runs-on: ubuntu-latest
    needs: backend
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install -e "backend[dev]"
      - name: programmatic eval gate (skipped without API key)
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CHEAP_MODE: "1"
        run: |
          if [ -z "$OPENAI_API_KEY" ]; then
            echo "no OPENAI_API_KEY secret — skipping eval gate"; exit 0
          fi
          cd backend && python -m app.evals run --label "ci pr" --gate
```

Implementer note: verify how `pytest` discovers the app package in CI (the repo may need `working-directory: backend` and plain `pytest` to mirror the Makefile; run whichever variant matches `make test` locally). Fix up before committing by pushing through `act` is NOT required — CI validation happens when the owner eventually pushes; the workflow just has to mirror the exact local commands.

- [ ] **Step 2: Live smoke — the first real regression datapoint (spends ~$0.30-0.60, CHEAP_MODE)**

Preconditions: `backend/.env` has the real key; CHEAP_MODE defaults on.

```bash
cd backend
.venv/bin/python -m app.evals run --label "m4 baseline" --write-baseline
```

Expected: a printed summary like `eval <id>: tier1 27/30 (90%), judge_avg=None, cost=$0.4x`, and `backend/app/evals/baseline.json` written. Investigate obviously broken outcomes (pass rate < ~60% usually means claim extraction/scoring mismatch, not model quality — inspect `tier1_detail` rows via `GET /api/evals/<id>` before blaming the model).

Then one judged run for the calibration labeling set (~$0.50-1.00 more in cheap mode):

```bash
.venv/bin/python -m app.evals run --label "m4 judged" --judge
.venv/bin/python -m app.evals label-template <printed eval id> > app/evals/labels/human_labels.yaml
```

`human_labels.yaml` now holds ~30 judged entries with `null` scores — the OWNER hand-fills them (the agent must never invent human labels). Commit the template as-is.

- [ ] **Step 3: Check off M4 in MILESTONES.md** (all M4 boxes except any the owner must finish — the ~40 hand labels remain unchecked until labeled). Do not `git add MILESTONES.md`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml backend/app/evals/baseline.json backend/app/evals/labels/human_labels.yaml
git commit -m "feat(m4): CI with programmatic eval gate + first live baseline"
```

---

## Owner handoff (end of M4)

Hands-on items only a human can do:
1. **Hand-label the calibration set** — fill every `null` in `backend/app/evals/labels/human_labels.yaml` (1-5 per dimension, rubric in `backend/app/agents/prompts/judge.md`). Then `python -m app.evals calibration` prints the agreement report and the Evals screen's calibration card lights up. ~40 labels ≈ one focused hour.
2. **Device checklist** (run `make dev`): the Task 14 checklist (router shapes) + Evals screen renders the real baseline run + `/evals` empty states.
3. **CI secret** — add `OPENAI_API_KEY` to the GitHub repo secrets whenever this gets pushed; until then the gate self-skips.
4. If judge agreement is weak (κ < ~0.4 on a dimension), iterate `judge.md` wording and re-run `--judge` (replay keeps agent costs at zero; only the judge re-bills).

## Self-review notes (already applied)

- Spec coverage: every M4 checklist line maps to a task — golden sets (1-3), programmatic tier (4), judge+rubric (5), 40-label calibration + stats (8, 15, owner), regression tracking w/ SHA+config hash (6-7), Evals screen (10-11), CI gate (15), router + conditional composer + trace visibility (12-14), routing tests (13), "eval scores unchanged after routing" (re-run Step 2 of Task 15 after Task 13 lands if routing merged after the baseline; compare pass rates).
- Known intentional deviations: golden data is seeded-synthetic rather than downloaded public CSVs (self-verifying derivations beat licensing + download flakiness; ATTRIBUTION.md says so honestly); "~40 labels" depends on how many judged answers exist (~30 questions; run `--judge` twice at different configs if more label volume is wanted).
- Type-consistency: `GoldenExpected` field names appear in Tasks 2/3/4/7; `DIMENSIONS` tuple in 5/8; store method kwargs in 6/7/8/9; TS types in 10/11. `numbers_match`, `profile_dataframe`, `_execute` call shapes are flagged as verify-on-read where the M3 code is the source of truth.

