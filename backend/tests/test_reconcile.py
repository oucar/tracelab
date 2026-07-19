"""Tolerance policy: design the rules, test the rules (plan §7 — critic false alarms)."""

from app.runtime.reconcile import CriticFinding, numbers_match, reconcile_claims
from app.runtime.state import Claim


def claim_num(value, id="1-1") -> Claim:
    return Claim(id=id, step_id=1, text=f"x = {value}", kind="numeric", value=value)


def finding(id="1-1", **kw) -> CriticFinding:
    return CriticFinding(claim_id=id, **kw)


def one(claims, findings):
    return reconcile_claims(claims, findings, rel_tol=0.01)[0]


def test_float_within_relative_tolerance_verifies():
    assert numbers_match(100.0, 100.9, 0.01)
    assert not numbers_match(100.0, 102.0, 0.01)
    assert numbers_match(0.0, 0.0, 0.01)


def test_numeric_claim_verified_and_discrepant():
    assert one([claim_num(100.4)], [finding(value=100.9)]).status == "verified"
    v = one([claim_num(100.0)], [finding(value=90.0)])
    assert v.status == "discrepancy" and v.critic_value == 90.0


def test_integral_values_require_exact_match():
    assert one([claim_num(17)], [finding(value=17.0)]).status == "verified"
    assert one([claim_num(17)], [finding(value=18.0)]).status == "discrepancy"


def test_categorical_matching_is_case_insensitive():
    c = Claim(id="1-1", step_id=1, text="busiest day", kind="categorical", value="Saturday")
    assert one([c], [finding(value="  saturday ")]).status == "verified"
    assert one([c], [finding(value="Sunday")]).status == "discrepancy"


def stat_claim(direction="higher", significant=True) -> Claim:
    return Claim(
        id="1-1",
        step_id=1,
        text="weekend fares higher",
        kind="statistical",
        direction=direction,
        significant=significant,
    )


def test_statistical_direction_and_significance_must_match():
    ok = finding(direction="higher", significant=True, methodology_ok=True)
    assert one([stat_claim()], [ok]).status == "verified"
    assert (
        one([stat_claim()], [finding(direction="lower", significant=True)]).status == "discrepancy"
    )
    assert (
        one([stat_claim()], [finding(direction="higher", significant=False)]).status
        == "discrepancy"
    )


def test_bad_methodology_is_a_discrepancy_even_when_numbers_agree():
    v = one(
        [stat_claim()],
        [
            finding(
                direction="higher",
                significant=True,
                methodology_ok=False,
                notes="t-test on heavily skewed data; Mann-Whitney was appropriate",
            )
        ],
    )
    assert v.status == "discrepancy" and "methodology" in v.reason.lower()


def test_unmatched_and_unverifiable_claims():
    assert one([claim_num(5.0)], []).status == "unverifiable"
    assert one([claim_num(5.0)], [finding(could_not_verify=True)]).status == "unverifiable"
