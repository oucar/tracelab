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
