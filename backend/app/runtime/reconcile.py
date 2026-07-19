"""Deterministic reconciliation of analyst claims against critic findings.

The tolerance policy lives HERE, in plain testable Python — the LLM critic only
derives values; it never decides whether a mismatch is within tolerance.

Policy (plan §7, "critic false alarms"):
  - floats: relative epsilon (cfg.numeric_rel_tolerance), near-zero guarded
  - integral values: exact
  - categorical: case-insensitive, whitespace-stripped equality
  - statistical: direction and significance conclusion must match; a method the
    critic judged inappropriate is a discrepancy even when the numbers agree
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.runtime.state import Claim, Verdict


class CriticFinding(BaseModel):
    """The critic's independently derived counterpart of one claim."""

    claim_id: str
    could_not_verify: bool = False
    value: float | str | None = None
    direction: Literal["higher", "lower", "none"] | None = None
    significant: bool | None = None
    methodology_ok: bool | None = None  # statistical claims: was the method appropriate?
    notes: str = ""


def numbers_match(a: float, b: float, rel_tol: float) -> bool:
    """Relative-epsilon comparison; integral values must match exactly."""
    if float(a).is_integer() and float(b).is_integer():
        return a == b
    denom = max(abs(a), abs(b))
    if denom < 1e-9:
        return True
    return abs(a - b) / denom <= rel_tol


def _reconcile_one(claim: Claim, finding: CriticFinding, rel_tol: float) -> Verdict:
    if finding.could_not_verify:
        return Verdict(
            claim_id=claim.id,
            status="unverifiable",
            reason=finding.notes or "critic could not verify this claim",
        )

    if claim.kind == "statistical":
        if finding.methodology_ok is False:
            return Verdict(
                claim_id=claim.id,
                status="discrepancy",
                methodology_ok=False,
                reason=f"methodology: {finding.notes or 'critic judged the method inappropriate'}",
            )
        if claim.direction != finding.direction or claim.significant != finding.significant:
            return Verdict(
                claim_id=claim.id,
                status="discrepancy",
                methodology_ok=finding.methodology_ok,
                reason=(
                    f"critic found direction={finding.direction}, "
                    f"significant={finding.significant}; "
                    f"claimed direction={claim.direction}, significant={claim.significant}"
                ),
            )
        return Verdict(claim_id=claim.id, status="verified", methodology_ok=finding.methodology_ok)

    if isinstance(claim.value, (int, float)) and isinstance(finding.value, (int, float)):
        if numbers_match(float(claim.value), float(finding.value), rel_tol):
            return Verdict(claim_id=claim.id, status="verified", critic_value=finding.value)
        return Verdict(
            claim_id=claim.id,
            status="discrepancy",
            critic_value=finding.value,
            reason=f"claimed {claim.value}, critic derived {finding.value}",
        )

    if isinstance(claim.value, str) and isinstance(finding.value, str):
        if claim.value.strip().casefold() == finding.value.strip().casefold():
            return Verdict(claim_id=claim.id, status="verified", critic_value=finding.value)
        return Verdict(
            claim_id=claim.id,
            status="discrepancy",
            critic_value=finding.value,
            reason=f"claimed {claim.value!r}, critic derived {finding.value!r}",
        )

    return Verdict(
        claim_id=claim.id,
        status="unverifiable",
        reason="claim and critic values are not comparable",
    )


def reconcile_claims(
    claims: list[Claim], findings: list[CriticFinding], rel_tol: float
) -> list[Verdict]:
    by_id = {f.claim_id: f for f in findings}
    verdicts: list[Verdict] = []
    for claim in claims:
        finding = by_id.get(claim.id)
        if finding is None:
            verdicts.append(
                Verdict(
                    claim_id=claim.id,
                    status="unverifiable",
                    reason="critic did not address this claim",
                )
            )
        else:
            verdicts.append(_reconcile_one(claim, finding, rel_tol))
    return verdicts
