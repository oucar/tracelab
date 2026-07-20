---
DRAFT — Onur to edit voice + fill the calibration numbers after hand-labeling.
Written for blog.ucaronur.com. Angle: "I calibrated my LLM judge." Everything
in the calibration table below is a placeholder until the hand-labeling pass
runs (see docs/evals.md §3). Do not publish with TK cells still in it.
---

# I calibrated my LLM judge

I built an agentic data analyst — call it tracelab. You upload a CSV, ask a
question in plain English, and a small team of LangGraph agents plans the
analysis, writes and runs Python in a sandbox, and a critic independently
re-derives every number before anything renders. I'm confident in that part;
it's covered by an eval harness with a critic that's structurally incapable
of grading its own homework, since it never sees the analyst's code.

Then I got to the fourth layer of the eval story — an LLM judge that scores
every answer 1–5 on clarity, honesty about uncertainty, chart appropriateness,
and methodological soundness — and noticed I was about to do the thing I
give other people grief for.

## Everyone ships LLM-as-judge. Almost nobody checks the judge.

If you've built anything agentic recently, you've reached for an LLM judge.
It's the obvious move: you can't hand-grade every eval run forever, so you
write a rubric, point a model at the output, and get a number back. Fast,
cheap, and it scales with your eval set instead of your calendar.

What almost nobody does is check whether the judge's number means anything.
You write the rubric, run it once, the number looks plausible, and you ship
it. The judge becomes load-bearing — you use it to decide whether a prompt
change helped, whether a cheaper model is "good enough," whether an expensive
config is worth the money — without ever asking whether the judge's 4 and a
human's 4 are the same 4.

That's the gap I didn't want tracelab to have. An unchecked judge is a vibe
wearing a rubric's clothes — numbers, dimensions, a markdown table, while
being exactly as trustworthy as "I read a few and they seemed fine." Worse,
it's a vibe you can no longer see, because it's hiding behind a number.

So before I let the judge referee anything that matters — including a
cost/quality/latency tradeoff study across four model configs — I built the
machinery to calibrate it against a human, and I'm running that process on
tracelab's own eval set. This post is the log: the rubric, the labeling
method, the numbers once I have them, and what I do with a judge that comes
back miscalibrated on some dimension (or doesn't — I'll say either way once
the labels exist).

## The rubric, and why these four dimensions

The judge scores every judged answer on four dimensions
(`backend/app/agents/prompts/judge.md`):

- **Clarity** — is the answer direct, well-structured, and does it actually
  answer the question asked?
- **Uncertainty honesty** — are unverified numbers, failures, and caveats
  surfaced honestly, or hidden and overclaimed? This one's inverted from
  what you'd expect: an answer that admits "the critic couldn't confirm
  this" scores *high*. Confidently wrong beats honestly uncertain in a lot
  of rubrics; not in this one.
- **Chart appropriateness** — does an included chart fit the question and
  data, is one conspicuously missing when clearly needed, or is it there
  but misleading?
- **Methodological soundness** — for statistical claims: right test family,
  effect size and p-value reported, assumptions acknowledged, claim strength
  proportionate to evidence. For descriptive answers: did the aggregation
  actually answer the question?

I picked these four because they're the four ways tracelab's own
verification pipeline can get the numbers right and still ship a bad answer:
a claim reconciles within tolerance but the narrative buries the one number
that didn't verify (honesty failure); the math is right but the chart is
wrong for the question (chart failure); a statistical claim passes the
deterministic tier-1 scorer — right direction, right significance — while
using a test family that doesn't fit the data's shape, which the
tolerance-based critic can't catch on its own (methodology failure).

Tier 1 checks whether the *numbers* are right. The judge checks whether the
*answer* is good — a strictly larger question that can't be reduced to a
value-equals-value comparison. Which is exactly why it needs extra scrutiny
a deterministic scorer doesn't: nothing stops a judge model from being
systematically too generous, or specifically blind to one dimension, and the
only way to find out is to check it against a human on the same rubric.

## Hand-labeling ~33 answers

The mechanism is boring on purpose — the interesting part is the discipline
around who does it, not the tooling. `python -m app.evals label-template
<eval_run_id>` walks a judged eval run and emits a YAML template: one entry
per question, the judge's own rationale included as context, all four rubric
dimensions set to `null`. I fill in my own 1–5 score per dimension per
question by hand, looking at the same digest the judge saw — narrative,
claims, verification status, chart count — not a richer view of the run.

The rule I set myself: don't look at the judge's score before writing mine.
The template includes the judge's *rationale* (useful context for what the
answer claimed) but not its numeric score — otherwise I'd anchor on it, and
the exercise would measure how persuasive the judge's prose is, not how
correct its number is.

What I'm watching for isn't just "do we agree," it's *where* and in *which
direction*. A judge that's uniformly half a point generous is a calibration
constant — annoying, fixable with a prompt tweak. A judge that's
specifically generous on uncertainty_honesty — where "sounds confident" and
"is honest" are easiest to conflate — is sharper: it rewards confident prose
over honest hedging, close to the exact failure mode this pipeline exists to
catch. And disagreements clustered on one *dataset* rather than one
dimension (the 33 questions span taxi trips, retail sales, weather) point at
a domain-familiarity problem, not a rubric problem — a different fix
entirely.

## The numbers

This is the part I can't fill in yet, and I'm not going to fake it.

**Real, already in hand:** the judge's own average score across all four
dimensions, on the first judged eval run (`af062bcbea34`, 33 answers,
`gpt-4o` judge): **4.35 / 5**, straight out of `python -m app.evals report`.
I want to be precise about what that number is and isn't. It's the judge
grading itself against its own rubric — `judge_avg`. It tells you the judge
*thinks* the answers are good, not whether it's *right* to think that.
Calibration is a different measurement — judge-vs-human agreement — and it
requires the human labels I haven't produced yet. A high judge_avg with zero
calibration checking is exactly the failure mode this post is about: a
number that looks like rigor and isn't.

The table below is the honest placeholder. I'm publishing with it still
empty rather than waiting, because hand-labeling all 33 judged answers
against the same rubric — then running `python -m app.evals calibration` —
is the point of this post, not a footnote before it.

```bash
cd backend
.venv/bin/python -m app.evals label-template af062bcbea34 > app/evals/labels/human_labels.yaml
# hand-score clarity / uncertainty_honesty / chart_appropriateness /
# methodological_soundness (1-5), blind to the judge's own score, then:
.venv/bin/python -m app.evals calibration
.venv/bin/python -m app.evals report      # markdown table, pasted below
```

| Dimension | n | Exact % | Within-1 % | Cohen's κ |
|---|---|---|---|---|
| clarity | — | TK | TK | TK |
| uncertainty honesty | — | TK | TK | TK |
| chart appropriateness | — | TK | TK | TK |
| methodological soundness | — | TK | TK | TK |
| **overall (pooled)** | — | TK | TK | TK |

κ is the number I actually care about, more than exact-match %. Raw
agreement is inflated by how often the rubric lands on the "obvious" score
(mostly 4s and 5s agreeing even from an uninformative judge); κ corrects for
chance agreement, so it's the honest read on whether the judge adds signal
beyond "guess 4."

## Iterating the judge prompt against κ, not vibes

If a dimension comes back weak, the loop is specific, not "reword the rubric
and hope": read the disagreeing cases for that dimension only, find the
*pattern* (systematic bias? a kind of answer that fools it? two things the
rubric meant to separate getting conflated?), tighten the wording in
`backend/app/agents/prompts/judge.md`, re-run the judge tier, re-label,
re-check κ.

One rule I'm holding myself to: don't touch what a dimension *means*
without re-labeling against the new definition. It's tempting, when a
dimension scores badly, to quietly redefine it into something the judge
already does well — that's moving the target instead of hitting it.

## Why the tradeoff study needed this first

tracelab's other big M5 artifact is a cost/quality/latency tradeoff study
across four model configurations — cheap everywhere, a stronger planner, a
stronger critic, everything strong — scored on judge_avg as the quality
axis. The judge model is pinned to `gpt-4o` across every config, never
varied with the config under test, so the quality axis measures the
*agents*, not "how much does gpt-4o like gpt-4o's own answers." That's the
first integrity guard; calibration is the second, and it matters more.
Pinning the judge stops it from being a moving target, but says nothing
about whether the fixed target is
any good. If the judge is systematically generous on methodological
soundness, or blind to a kind of uncertainty-hiding, every config in the
study inherits that same blind spot equally — "fair" in the sense of one
consistent yardstick, but a wrong yardstick, and every conclusion I draw
from the tradeoff chart would be quietly downstream of that. "Does the
stronger critic catch more real discrepancies, or just produce prose the
judge happens to reward" is a question I can't answer with an uncalibrated
judge, no matter how carefully I pin its model version. Calibration is what
turns "the judge said X" into "the judge said X, and that's worth listening
to."

## An uncalibrated judge is a random number generator with good vibes

That line isn't really about LLM judges specifically — it's about any metric
you didn't check against ground truth before you started trusting it. A
score with a rationale attached *looks* like a measurement: decimal places,
a rubric, structured output. All of that is theater unless someone has, at
some point, sat down and asked: when I look at the same thing this judge
looked at, do I get the same answer?

I'm not done with this yet — the table above stays a placeholder until I've
hand-labeled all 33 answers, blind to the judge's own scores, and run the
numbers. But the process is built and checked into the repo
(`backend/app/evals/calibration.py`, `label-template`, `calibration`, the
report command), designed so a bad result is exactly as visible as a good
one — nothing here makes a flattering κ easier to get than an honest one.
I didn't build an eval harness to prove the agents are good. I built it to
find out.
