# Evals

Full methodology, current results, and the honest state of what's still
pending an owner pass (judge calibration, the multi-config study). The
README covers this at pitch depth; this doc covers it at working-with-it
depth — how to reproduce every number, and where the seams are.

## 1. The golden sets

Three bundled CSVs, 11 questions each, 33 total
(`backend/app/evals/golden/{taxi,retail,weather}.yaml`):

```yaml
- id: taxi-007
  question: "Which pickup hour has the highest average fare?"
  expected:
    kind: numeric
    value: 5
    tolerance: 0
  tags: [aggregation, groupby]
```

`GoldenExpected.kind` is one of `numeric | categorical | statistical |
narrative` (`backend/app/evals/golden.py`). Statistical questions carry
`direction` (`higher`/`lower`), `significant` (bool), and `method_family`
(e.g. `mean-comparison`, `correlation`, `chi-square`, `regression`, `trend`)
instead of a single value — the eval is checking whether the agents reached
the right *conclusion* with an *appropriate method*, not whether they typed
a specific p-value.

**Self-verification, not hand-typed expectations.**
`backend/app/evals/derivations.py` computes every `expected.value` (and
statistical direction/significance) directly from the CSVs in
`data/samples/`. `python -m app.evals golden --write` regenerates the YAML
files from those derivations. This matters for trust: the golden set isn't
"a human eyeballed the answer once," it's "a deterministic function over the
actual data produces the same number the eval checks against," which means
the golden set can be regenerated whenever a sample dataset changes instead
of silently drifting from the CSVs it's supposed to describe.

## 2. Two-tier scoring

**Tier 1 — programmatic** (`backend/app/evals/scoring.py`). For every
non-`narrative` question, `score_tier1` pulls the claimed value out of the
run's structured `FinalAnswer.claims` and checks:

- `numeric`/`categorical` claims — direct value comparison, reusing
  `reconcile.numbers_match` so the eval's numeric tolerance is the *same
  function* the critic uses at runtime, not a parallel reimplementation
  that could silently diverge.
- `statistical` claims — the claimed `direction` must match (through a
  small synonym table: "higher/positive/increase/up/more/greater" all count
  as `higher`), `significant` must match exactly, and the claim's
  `methodology.method` string must fall in the expected `method_family` via
  a token-membership check (e.g. `"Welch t-test"` matches `mean-comparison`
  through the `welch`/`ttest` tokens in `METHOD_FAMILIES`).

Tier 1 costs nothing beyond running the graph itself — no extra LLM calls —
and covers 30 of the 33 questions (the other 3 are `narrative`, judge-only).

**Tier 2 — LLM judge** (`backend/app/evals/judge.py`). Every answer,
narrative or not, gets rubric-scored 1–5 on four dimensions
(`DIMENSIONS = ("clarity", "uncertainty_honesty", "chart_appropriateness",
"methodological_soundness")`) by a model built the same way every other
agent role is (`_structured_fn("judge", JudgeTurn, model=...)`), fed a JSON
digest of the narrative + claims + verification status + chart count
(`_answer_digest`) — the judge sees the same verified/unverified/detail
information a user would, not raw run internals.

## 3. Judge calibration — methodology and current state

An LLM judge that's never been checked against a human is, in the harness
writer's own words, "a random number generator with good vibes." The
calibration machinery (`backend/app/evals/calibration.py`) exists to make
that check cheap to run and honest to report:

1. `python -m app.evals label-template <eval_run_id>` walks every judged
   result in that eval run and emits a YAML template — one entry per
   question with the judge's own `judge_rationale` included as context for
   the human labeler, and all four rubric dimensions set to `null` for the
   human to fill in.
2. A human (the project owner, deliberately — see below) fills in their own
   1–5 score per dimension per question, saved to
   `backend/app/evals/labels/human_labels.yaml`.
3. `python -m app.evals calibration` (or `calibration_report(store, labels)`
   directly) pairs human vs. judge scores per dimension and computes, using
   `sklearn.metrics.cohen_kappa_score`:
   - **exact agreement %** — human and judge gave the identical 1–5 score
   - **within-1 agreement %** — scores differ by at most 1
   - **Cohen's κ** — chance-corrected agreement (0 if either rater used
     fewer than 2 distinct values in the sample, since κ is undefined there)

   ...per dimension, plus an **overall (pooled)** row across all four
   dimensions and every labeled question.

4. `python -m app.evals report` prints the resulting table (via
   `calibration_markdown` in `backend/app/evals/report.py`) as GitHub
   markdown, ready to paste into the README/this doc.

**Why the owner labels it, not an agent.** The label template is
purpose-built to be hand-filled by a human who did *not* generate the
judge's rationale and has no incentive to make the judge look calibrated.
An agent filling in its own grading key would make the calibration table
decorative rather than a real check — the entire point of the exercise is
that the numbers could come back bad and reveal a judge prompt that needs
work.

**Current state, stated plainly:** the label template for the 33 judged
answers in eval run `af062bcbea34` already exists at
`backend/app/evals/labels/human_labels.yaml`, but every dimension in it is
still `null`. Running `python -m app.evals calibration` today returns
`{"available": false, "n": 0, ...}` — correctly, since `n=0` labeled
questions is not enough to report agreement on. This is the honest
unfinished state of the project as of this doc, not a bug.

> **Calibration table — placeholder, pending owner hand-labels.**
> ```bash
> cd backend
> .venv/bin/python -m app.evals label-template af062bcbea34 > app/evals/labels/human_labels.yaml
> # hand-score clarity / uncertainty_honesty / chart_appropriateness /
> # methodological_soundness (1-5) for each of the 33 entries, then:
> .venv/bin/python -m app.evals calibration
> .venv/bin/python -m app.evals report      # markdown table, paste below and into README
> ```
> | Dimension | n | Exact % | Within-1 % | Cohen's κ |
> |---|---|---|---|---|
> | clarity | — | — | — | — |
> | uncertainty honesty | — | — | — | — |
> | chart appropriateness | — | — | — | — |
> | methodological soundness | — | — | — | — |
> | **overall (pooled)** | — | — | — | — |
>
> If agreement comes back weak on a dimension, the loop is: read the
> disagreeing cases, tighten `backend/app/agents/prompts/judge.md` for that
> dimension specifically, re-run the judge tier, re-label, re-check κ. Don't
> touch the rubric definition itself without re-labeling — that would be
> moving the target instead of hitting it.

## 4. Real numbers already in hand

Two live runs exist as of this doc, both `gpt-4o-mini` everywhere
(`cheap_mode` config, `config_hash 9d3a79641dca`):

| eval_run_id | label | questions | tier-1 scorable | tier-1 passed | judge_avg | cost |
|---|---|---|---|---|---|---|
| `3b5fec45879f` | m4 baseline | 33 | 30 | 20 (**67%**) | — (no judge) | **$0.11** |
| `af062bcbea34` | m4 judged | 33 | 30 | 20 (67%) | **4.35 / 5** | $0.12 |

Both are `python -m app.evals run` invocations, not `study` runs (no
`study:*` label exists yet — see §5). These are the numbers quoted in the
README as "the first live baseline"; they are the actual harness output,
via `st.list_eval_runs()`, not hand-typed.

A third early run (`550a6fdd8a71`, "m4 baseline") passed only 6/30 — kept in
the store as-is (the regression timeline is meant to show the real
trajectory including a bad run, not a curated best-of).

## 5. The tradeoff study

`backend/app/evals/study.py` + `backend/configs/*.yaml` let the same golden
sweep run once per model configuration, varying which agent roles get the
strong model:

| config | router | planner | analyst | critic | composer | judge |
|---|---|---|---|---|---|---|
| `mini` | mini | mini | mini | mini | mini | gpt-4o |
| `strong-planner` | mini | **gpt-4o** | mini | mini | mini | gpt-4o |
| `strong-critic` | mini | mini | mini | **gpt-4o** | mini | gpt-4o |
| `strong` | mini | gpt-4o | gpt-4o | gpt-4o | gpt-4o | gpt-4o |

The judge is pinned to `gpt-4o` in every config file — never varied with the
config under test. This is the study's validity linchpin: if the judge
model changed alongside the agents, the quality axis would be measuring
"how much does gpt-4o like gpt-4o's own answers" instead of comparing
configs on a fixed yardstick.

`GraphDeps.default(models=cfg.models)` threads each config's per-role model
map through `_structured_fn(role, schema, model=models.get(role))`,
bypassing `cheap_mode`'s collapse entirely — explicit models always win.
`config_snapshot()` (`backend/app/evals/harness.py`) records the *exact*
dict that was passed, not a re-read of `settings()`, and hashes all five
agent roles plus the judge model together, so two configs that differ only
in, say, `router` or `judge_model` get distinct `config_hash`es instead of
colliding.

```bash
cd backend
.venv/bin/python -m app.evals study --configs mini              # ~$1 with judge, ungated
.venv/bin/python -m app.evals study --configs strong-planner,strong-critic,strong  # ~$10-20, OWNER GATE
.venv/bin/python -m app.evals report --png ../docs/assets/tradeoff.png
```

`study_rows()` (`backend/app/evals/report.py`) takes the *latest* eval run
per `study:<config>` label and derives `cost_per_question` and
`latency_s_per_question`; `study_markdown()` renders the comparison table;
`tradeoff_png()` plots cost/question (x, log-scaled if the spread exceeds
10×) against judge_avg (y, falling back to `tier1_pass_rate × 5` when no
judge ran), point size proportional to latency, one point per config,
labeled.

> **Tradeoff study — placeholder.** No `study:*` eval run exists yet as of
> this doc (`study_rows()` returns an empty list against the current
> store). The mini config is cheap enough to run without asking (~$1 with
> the judge); the three strong configs are gated on the owner's explicit
> go-ahead (est. $10–20 combined for 33 questions × 3 configs, gpt-4o roles
> + gpt-4o judge) per the M5 build plan. Regenerate once approved and run:
> ```bash
> cd backend && .venv/bin/python -m app.evals study
> .venv/bin/python -m app.evals report --png ../docs/assets/tradeoff.png
> ```
> | Config | Tier-1 pass | Judge avg (1–5) | $/question | s/question |
> |---|---|---|---|---|
> | mini | — | — | — | — |
> | strong-planner | — | — | — | — |
> | strong-critic | — | — | — | — |
> | strong | — | — | — | — |
>
> ![tradeoff chart](assets/tradeoff.png)
>
> **Conclusions (write after the study runs, from the actual numbers, not
> from priors):**
> 1. _Does `strong-planner` beat `mini` on quality-per-dollar — i.e. does a
>    smarter decomposition step pay for itself, or do cheap analysts squander
>    a good plan?_
> 2. _Does `strong-critic` catch more genuine discrepancies than `mini`, or
>    mostly raise false alarms that cost a retry without changing the
>    verdict?_
> 3. _Does `strong` (every role upgraded) justify its cost over the single-role
>    upgrades, or is the marginal quality gain from strengthening a second or
>    third role smaller than strengthening the first?_

## 6. Regression tracking

Every `run_eval()` call, live or `study`, writes one `eval_runs` row: git
SHA (`git rev-parse --short HEAD`), `config_hash` + the full config JSON,
`questions_total`/`tier1_scorable`/`tier1_passed`, `judge_avg` (`None` when
no judge ran, not 0 — a missing measurement should never silently read as a
bad score), total cost, and wall-clock duration. The Evals screen
(`frontend/src/pages/EvalsScreen.tsx`) plots tier-1 % and judge average
against `created_at`/`git_sha` on a dual-axis `LineChart`, and the runs
`DataGrid` links each row to its per-question `eval_results`.

`.github/workflows/ci.yml`'s `eval-gate` job runs `python -m app.evals run
--label "ci pr" --gate` on every pull request — `--gate` compares the fresh
tier-1 pass rate against `backend/app/evals/baseline.json` (currently
`{"tier1_pass_rate": 0.6667, "eval_run_id": "3b5fec45879f"}`) minus a 5-point
margin and fails the build if the run dropped below the floor. As written,
the job self-skips (prints a message, exits 0) whenever the `OPENAI_API_KEY`
repo secret isn't configured — which is its current state. That's a
deliberate no-op, not a broken pipeline: the gate exists and is correct, it
just needs the owner to add the secret to actually run. The judge (tier 2)
tier intentionally never runs in CI — it stays a manual/local step to
control spend, per the M4 build plan.
