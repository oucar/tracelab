"""Model price table and cost math.

Prices are USD per 1M tokens, hardcoded deliberately: an observability tool
that silently fetches prices is harder to trust than one you can read.
Versioned API names ("gpt-4o-mini-2024-07-18") prefix-match their base entry;
the longest prefix wins.

$0 means two very different things, and conflating them is how a "hard cap"
quietly stops being one:

  - **Genuinely free** — a test stub (model "") or a replay (model "replay").
    Nothing was spent, so $0 is the truth and the budget should ignore it.
  - **Unpriced** — a real model name with no entry in `PRICES`. Something WAS
    spent and we cannot say how much. Reporting $0 there would make every
    dollar-denominated budget silently unenforceable for that model, so
    `is_unpriced` flags it and callers surface it rather than trusting the 0.

`cost_usd` still returns 0.0 in both cases (an honest $0 beats a fabricated
number), but `is_unpriced` lets the budget and the UI tell them apart.
"""

from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    # model: (usd_per_1m_input_tokens, usd_per_1m_output_tokens)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

#: Models that cost nothing by construction — not "we forgot to price them".
FREE_MODELS = frozenset({"", "replay", "stub"})


def _price_for(model: str) -> tuple[float, float] | None:
    price = PRICES.get(model)
    if price is None and model:
        for name in sorted(PRICES, key=len, reverse=True):
            if model.startswith(name):
                price = PRICES[name]
                break
    return price


def is_unpriced(model: str) -> bool:
    """True for a real model whose cost we cannot compute — the dangerous $0."""
    return model not in FREE_MODELS and _price_for(model) is None


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price = _price_for(model)
    if price is None:
        return 0.0
    price_in, price_out = price
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
