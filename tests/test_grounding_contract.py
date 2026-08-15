"""Tests for the Release 2 PR 2 verification contract (postgres_graph_rag/
grounding.py). No provider calls, no live Postgres -- this is pure type/
interface behavior: valid/invalid construction, the grounded compatibility
property's derivation from grounding_status, and that the pre-existing
verifier fixtures structurally satisfy the Verifier protocol without ever
importing from this module.
"""
import pytest

from benchmarks.grounding.verifier_fixtures import (
    MalformedResponseVerifier,
    RaisingVerifier,
    TimingOutVerifier,
)
from postgres_graph_rag.grounding import (
    AnswerClaim,
    ClaimVerification,
    Verifier,
    VerifiedAnswerResult,
    validate_grounding_mode,
)


def test_answer_claim_defaults_to_no_citations():
    claim = AnswerClaim(id="c1", text="checkout-service depends on auth-service.")
    assert claim.citation_ids == []


def test_claim_verification_rejects_invalid_verdict():
    with pytest.raises(ValueError, match="verdict must be one of"):
        ClaimVerification(claim_id="c1", verdict="probably")  # type: ignore[arg-type]


def test_claim_verification_rejects_invalid_reason_code():
    with pytest.raises(ValueError, match="reason_code must be one of"):
        ClaimVerification(claim_id="c1", verdict="supported", reason_code="because")  # type: ignore[arg-type]


def test_claim_verification_rejects_out_of_range_confidence():
    with pytest.raises(ValueError, match="confidence must be in"):
        ClaimVerification(claim_id="c1", verdict="supported", confidence=1.5)


def test_claim_verification_defaults_confidence_to_none():
    v = ClaimVerification(claim_id="c1", verdict="supported")
    assert v.confidence is None


@pytest.mark.parametrize(
    "grounding_status,expected_grounded",
    [
        ("verified", True),
        ("partially_verified", True),
        ("citation_valid_only", True),
        ("contradicted", False),
        ("insufficient", False),
        ("verification_failed", False),
        ("abstained", False),
    ],
)
def test_grounded_property_derives_from_grounding_status(grounding_status, expected_grounded):
    result = VerifiedAnswerResult(
        claims=[], verifications=[], answer="...", grounding_status=grounding_status,
    )
    assert result.grounded is expected_grounded


def test_verified_answer_result_rejects_invalid_grounding_status():
    with pytest.raises(ValueError, match="grounding_status must be one of"):
        VerifiedAnswerResult(claims=[], verifications=[], answer="...", grounding_status="maybe")  # type: ignore[arg-type]


def test_verification_for_looks_up_by_claim_id():
    v1 = ClaimVerification(claim_id="c1", verdict="supported")
    v2 = ClaimVerification(claim_id="c2", verdict="contradicted")
    result = VerifiedAnswerResult(
        claims=[], verifications=[v1, v2], answer="...", grounding_status="partially_verified",
    )
    assert result.verification_for("c2") is v2
    assert result.verification_for("nonexistent") is None


@pytest.mark.parametrize("mode", ["citation_only", "verified", "verified_strict"])
def test_validate_grounding_mode_accepts_known_modes(mode):
    assert validate_grounding_mode(mode) == mode


def test_validate_grounding_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="grounding_mode must be one of"):
        validate_grounding_mode("citation_and_vibes")


@pytest.mark.parametrize(
    "stub_verifier_cls", [RaisingVerifier, TimingOutVerifier, MalformedResponseVerifier],
)
def test_preexisting_verifier_fixtures_structurally_satisfy_the_protocol(stub_verifier_cls):
    """These stubs (benchmarks/grounding/verifier_fixtures.py) were written
    before this contract existed and don't import from it -- confirms the
    Verifier protocol's shape actually matches what they already implement,
    rather than the protocol being defined in a vacuum."""
    assert isinstance(stub_verifier_cls(), Verifier)


def test_a_class_missing_verify_does_not_satisfy_the_protocol():
    class NotAVerifier:
        async def check(self, claims):
            return []

    assert not isinstance(NotAVerifier(), Verifier)
