import pydantic
import pytest

from app.evals.golden import GoldenExpected
from app.evals.scoring import score_tier1
from app.runtime.state import Claim, FinalAnswer, Methodology, VerifiedClaim


def vc(claim: Claim) -> VerifiedClaim:
    return VerifiedClaim(claim=claim, status="verified", detail="")


def final_with(*claims: Claim) -> FinalAnswer:
    return FinalAnswer(narrative="x", claims=[vc(c) for c in claims], charts=[], failed=False)


def num_claim(value: float) -> Claim:
    return Claim(id="c1", step_id=1, text="t", kind="numeric", value=value,
                 direction="none", significant=None, methodology=None)


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
                  direction="none", significant=None, methodology=None)
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
    # The synonym map lives in scoring.py and normalizes the *expected*-side
    # vocabulary / canonical claim values; a canonical "higher" claim still
    # matches an exact "higher" expectation.
    assert score_tier1(exp, final_with(stat_claim("higher", True, "Pearson correlation"))).passed

    # Claim.direction is a closed Literal — this is the OpenAI structured-output
    # schema enum guarantee, and it's also what keeps reconcile.py's raw string
    # equality (`claim.direction != finding.direction`) valid: a synonym like
    # "positive" must never reach a Claim, since CriticFinding.direction is the
    # same closed Literal and compares directly against it.
    with pytest.raises(pydantic.ValidationError):
        stat_claim("positive", True, "Pearson correlation")


def test_failed_or_missing_run_fails():
    exp = GoldenExpected(kind="numeric", value=1, tolerance=0.0)
    assert not score_tier1(exp, None).passed
    failed = FinalAnswer(narrative="", claims=[], charts=[], failed=True)
    assert not score_tier1(exp, failed).passed


def test_narrative_not_scorable():
    s = score_tier1(GoldenExpected(kind="narrative"), final_with(num_claim(1)))
    assert not s.scorable
