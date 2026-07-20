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
