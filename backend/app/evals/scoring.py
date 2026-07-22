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
