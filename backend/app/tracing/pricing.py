"""Model price table and cost math.

Prices are USD per 1M tokens, hardcoded deliberately: an observability tool
that silently fetches prices is harder to trust than one you can read.
Unknown models (including stubs and "replay") cost 0 — better an honest $0
than a fabricated number. Versioned API names ("gpt-4o-mini-2024-07-18")
prefix-match their base entry; the longest prefix wins.
"""

from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    # model: (usd_per_1m_input_tokens, usd_per_1m_output_tokens)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price = PRICES.get(model)
    if price is None and model:
        for name in sorted(PRICES, key=len, reverse=True):
            if model.startswith(name):
                price = PRICES[name]
                break
    if price is None:
        return 0.0
    price_in, price_out = price
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
