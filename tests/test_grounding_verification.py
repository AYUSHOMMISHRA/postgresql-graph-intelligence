"""Comprehensive tests for postgres_graph_rag/verification.py (Release 2
PR 3's deterministic layers). No live Postgres, no LLM -- pure functions
and one in-memory verifier over hand-built TenantRetrievedChunk evidence.

Covers, per the PR 3 spec: stale evidence, invalid citations, reversed
relationships, conflicts, partial support, fabricated quotes, and
unsupported claims -- plus policy evaluation and deterministic rendering.
"""
import json
from pathlib import Path

import pytest

from postgres_graph_rag.grounding import AnswerClaim, ClaimVerification
from postgres_graph_rag.tenant_engine import TenantRetrievedChunk
from postgres_graph_rag.verification import (
    DeterministicVerifier,
    evaluate_policy,
    parse_claims,
    render_answer,
)

FABRICATION_FIXTURES_PATH = (
    Path(__file__).parent.parent / "benchmarks" / "grounding" / "fabrication_fixtures.json"
)


def chunk(source_id: str, ordinal: int, content: str) -> TenantRetrievedChunk:
    return TenantRetrievedChunk(
        id=f"{source_id}-{ordinal}", document_id=source_id, source_id=source_id,
        ordinal=ordinal, content=content, rrf_score=1.0, lexical_rank=1, semantic_rank=1,
    )


def marker(source_id: str, ordinal: int) -> str:
    return f"[{source_id}#{ordinal}]"


# ----------------------------------------------------------------------
# 1. Structured claim parsing
# ----------------------------------------------------------------------


def test_parse_claims_valid():
    claims = parse_claims([{"id": "c1", "text": "Acme makes Widgets.", "citation_ids": ["[doc-1#0]"]}])
    assert claims == [AnswerClaim(id="c1", text="Acme makes Widgets.", citation_ids=["[doc-1#0]"])]


def test_parse_claims_assigns_default_id_when_missing():
    claims = parse_claims([{"text": "Acme makes Widgets."}])
    assert claims[0].id == "claim-0"
    assert claims[0].citation_ids == []


def test_parse_claims_rejects_missing_text():
    with pytest.raises(ValueError, match="no non-empty 'text'"):
        parse_claims([{"citation_ids": []}])


def test_parse_claims_rejects_blank_text():
    with pytest.raises(ValueError, match="no non-empty 'text'"):
        parse_claims([{"text": "   "}])


def test_parse_claims_rejects_non_string_citation_ids():
    with pytest.raises(ValueError, match="citation_ids must be a list of strings"):
        parse_claims([{"text": "Acme makes Widgets.", "citation_ids": [123]}])


def test_parse_claims_rejects_non_dict_entry():
    with pytest.raises(ValueError, match="is not an object"):
        parse_claims(["just a string"])


# ----------------------------------------------------------------------
# 2/3. Citation existence + (by construction) active evidence / stale citations
# ----------------------------------------------------------------------


def test_no_citation_is_insufficient():
    verifier = DeterministicVerifier(evidence=[])
    claim = AnswerClaim(id="c1", text="Acme makes Widgets.", citation_ids=[])
    [result] = _run(verifier, [claim])
    assert result.verdict == "insufficient"
    assert result.reason_code == "citation_missing"


def test_citation_outside_retrieved_evidence_is_insufficient():
    """Covers both 'the model hallucinated a citation id' and 'the cited
    document was superseded/deleted before verification ran' -- both look
    identical from here: the marker simply isn't in the evidence pool this
    verifier was constructed with."""
    evidence = [chunk("doc-1", 0, "Acme makes Widgets.")]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(id="c1", text="Acme makes Widgets.", citation_ids=[marker("doc-1-stale", 0)])
    [result] = _run(verifier, [claim])
    assert result.verdict == "insufficient"
    assert result.reason_code == "citation_outside_retrieved_evidence"


def test_one_invalid_citation_among_several_still_fails_the_whole_claim():
    evidence = [chunk("doc-1", 0, "Acme makes Widgets.")]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(
        id="c1", text="Acme makes Widgets.",
        citation_ids=[marker("doc-1", 0), marker("doc-2", 0)],  # doc-2 doesn't exist
    )
    [result] = _run(verifier, [claim])
    assert result.verdict == "insufficient"
    assert result.reason_code == "citation_outside_retrieved_evidence"


# ----------------------------------------------------------------------
# Explicit support (the case the whole pipeline must not regress on)
# ----------------------------------------------------------------------


def test_explicit_supported_relationship():
    evidence = [chunk("incident-1", 0, "checkout-service depends on auth-service for session validation.")]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service.",
        citation_ids=[marker("incident-1", 0)],
    )
    [result] = _run(verifier, [claim])
    assert result.verdict == "supported"
    assert result.reason_code == "explicit_relation_confirmed"
    assert result.supporting_quotes == ["checkout-service depends on auth-service"]
    # The quote actually is a literal substring of the cited evidence.
    assert result.supporting_quotes[0] in evidence[0].content


# ----------------------------------------------------------------------
# 4. Identifier / relevance checks
# ----------------------------------------------------------------------


def test_valid_but_irrelevant_citation():
    evidence = [chunk("incident-1", 0, "The on-call rotation changes every Monday at 09:00 UTC.")]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service.",
        citation_ids=[marker("incident-1", 0)],
    )
    [result] = _run(verifier, [claim])
    assert result.verdict == "insufficient"
    assert result.reason_code == "evidence_irrelevant"


# ----------------------------------------------------------------------
# What the deterministic layer honestly cannot confirm -- must not
# over-claim "supported" from weak signals; PR 4's model layer is where
# these get their recall back.
# ----------------------------------------------------------------------


def test_reversed_relationship_is_not_wrongly_supported():
    evidence = [chunk("incident-1", 0, "auth-service depends on checkout-service for rate-limit telemetry.")]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service.",
        citation_ids=[marker("incident-1", 0)],
    )
    [result] = _run(verifier, [claim])
    assert result.verdict != "supported"


def test_same_entities_different_predicate_is_not_wrongly_supported():
    evidence = [chunk("incident-1", 0, "checkout-service monitors auth-service's health-check endpoint.")]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service.",
        citation_ids=[marker("incident-1", 0)],
    )
    [result] = _run(verifier, [claim])
    assert result.verdict != "supported"


def test_unsupported_causal_statement_is_not_wrongly_supported():
    evidence = [chunk("incident-1", 0, "auth-service latency rose at 03:10 UTC. checkout-service error rate rose at 03:12 UTC.")]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(
        id="c1", text="checkout-service's error rate rose because auth-service latency increased.",
        citation_ids=[marker("incident-1", 0)],
    )
    [result] = _run(verifier, [claim])
    assert result.verdict != "supported"


def test_multi_source_compound_claim_is_a_documented_limitation():
    """Neither citation's text alone contains the full compound claim, so
    the deterministic layer correctly can't confirm it end-to-end -- this
    is the documented recall gap PR 4 (batched model entailment) exists to
    close, not a bug in this layer."""
    evidence = [
        chunk("incident-1", 0, "checkout-service depends on auth-service for session validation."),
        chunk("incident-2", 0, "auth-service is owned by Identity Team."),
    ]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service, which is owned by Identity Team.",
        citation_ids=[marker("incident-1", 0), marker("incident-2", 0)],
    )
    [result] = _run(verifier, [claim])
    assert result.verdict == "insufficient"


# ----------------------------------------------------------------------
# Numeric/date mismatch -- one case the deterministic layer *can* positively
# confirm as contradicted, not just withhold judgment on.
# ----------------------------------------------------------------------


def test_unsupported_number_is_contradicted():
    evidence = [chunk("incident-1", 0, "checkout-service experienced a 12-minute outage starting at 03:14 UTC.")]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(
        id="c1", text="checkout-service experienced a 45-minute outage.",
        citation_ids=[marker("incident-1", 0)],
    )
    [result] = _run(verifier, [claim])
    assert result.verdict == "contradicted"
    assert result.reason_code == "numeric_or_date_mismatch"


# ----------------------------------------------------------------------
# Conflicts: another retrieved chunk (not necessarily cited) negates the
# claim's relationship for the same entities.
# ----------------------------------------------------------------------


def test_conflicting_documents_downgrades_from_supported_to_insufficient():
    evidence = [
        chunk("incident-1", 0, "checkout-service depends on auth-service for session validation."),
        chunk("incident-2", 0, "checkout-service does not depend on auth-service; it validates sessions locally."),
    ]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service.",
        citation_ids=[marker("incident-1", 0)],  # cites only the supporting side
    )
    [result] = _run(verifier, [claim])
    assert result.verdict == "insufficient"
    assert result.reason_code == "evidence_conflicting"


def test_no_conflict_when_no_negation_present():
    """Two sources agreeing (or simply both silent) must not trigger the
    conflict heuristic -- it's specifically a negation-marker check, not a
    'more than one source' check."""
    evidence = [
        chunk("incident-1", 0, "checkout-service depends on auth-service for session validation."),
        chunk("incident-2", 0, "checkout-service depends on auth-service for token refresh too."),
    ]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service.",
        citation_ids=[marker("incident-1", 0)],
    )
    [result] = _run(verifier, [claim])
    assert result.verdict == "supported"


# ----------------------------------------------------------------------
# Incorrect-abstention pressure: formatting differences must not defeat a
# genuine match.
# ----------------------------------------------------------------------


def test_hyphen_vs_space_formatting_does_not_cause_incorrect_abstention():
    evidence = [chunk("incident-1", 0, "checkout service depends on auth service for session validation.")]
    verifier = DeterministicVerifier(evidence=evidence)
    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service.",
        citation_ids=[marker("incident-1", 0)],
    )
    [result] = _run(verifier, [claim])
    assert result.verdict == "supported"
    # The claim's own (hyphenated) phrasing isn't a literal substring of the
    # (space-formatted) evidence, so the reported quote must be the
    # evidence's own text, not a fabricated rephrasing of the claim.
    assert result.supporting_quotes == [evidence[0].content]
    assert result.supporting_quotes[0] in evidence[0].content


# ----------------------------------------------------------------------
# Fabricated quotes: verified against the actual fixtures used for a
# future model-verifier's quote-validation layer, confirming this
# deterministic layer's own quote location never produces one.
# ----------------------------------------------------------------------


def test_located_quotes_are_never_the_fabricated_variant():
    fixtures = json.loads(FABRICATION_FIXTURES_PATH.read_text())["fixtures"]
    for fixture in fixtures:
        evidence = [chunk("src", 0, fixture["chunk_content"])]
        verifier = DeterministicVerifier(evidence=evidence)
        claim = AnswerClaim(id="c1", text=fixture["real_quote"] + ".", citation_ids=[marker("src", 0)])
        [result] = _run(verifier, [claim])
        if result.supporting_quotes:
            assert fixture["fabricated_quote"] not in result.supporting_quotes
            for q in result.supporting_quotes:
                assert q in fixture["chunk_content"], (
                    f"{fixture['id']}: located quote {q!r} is not a literal substring of the chunk"
                )


# ----------------------------------------------------------------------
# 6. Policy evaluation
# ----------------------------------------------------------------------


def _verifications(*pairs):
    return [ClaimVerification(claim_id=cid, verdict=v) for cid, v in pairs]


def test_citation_only_mode_ignores_verdicts():
    claims = [AnswerClaim(id="c1", text="x", citation_ids=["[a#0]"])]
    verifications = _verifications(("c1", "contradicted"))
    assert evaluate_policy(claims, verifications, "citation_only") == "citation_valid_only"


def test_verified_mode_all_supported():
    claims = [AnswerClaim(id="c1", text="x"), AnswerClaim(id="c2", text="y")]
    verifications = _verifications(("c1", "supported"), ("c2", "supported"))
    assert evaluate_policy(claims, verifications, "verified") == "verified"


def test_verified_mode_partial_support():
    claims = [AnswerClaim(id="c1", text="x"), AnswerClaim(id="c2", text="y")]
    verifications = _verifications(("c1", "supported"), ("c2", "insufficient"))
    assert evaluate_policy(claims, verifications, "verified") == "partially_verified"


def test_verified_mode_all_insufficient():
    claims = [AnswerClaim(id="c1", text="x")]
    verifications = _verifications(("c1", "insufficient"))
    assert evaluate_policy(claims, verifications, "verified") == "insufficient"


def test_verified_mode_any_contradicted_flags_contradicted_even_with_supported_claims():
    claims = [AnswerClaim(id="c1", text="x"), AnswerClaim(id="c2", text="y")]
    verifications = _verifications(("c1", "supported"), ("c2", "contradicted"))
    assert evaluate_policy(claims, verifications, "verified") == "contradicted"


def test_verified_strict_mode_abstains_on_any_unsupported_claim():
    claims = [AnswerClaim(id="c1", text="x"), AnswerClaim(id="c2", text="y")]
    verifications = _verifications(("c1", "supported"), ("c2", "insufficient"))
    assert evaluate_policy(claims, verifications, "verified_strict") == "insufficient"


def test_verified_strict_mode_verified_only_when_everything_supported():
    claims = [AnswerClaim(id="c1", text="x"), AnswerClaim(id="c2", text="y")]
    verifications = _verifications(("c1", "supported"), ("c2", "supported"))
    assert evaluate_policy(claims, verifications, "verified_strict") == "verified"


def test_empty_claims_is_abstained_regardless_of_mode():
    for mode in ("citation_only", "verified", "verified_strict"):
        assert evaluate_policy([], [], mode) == "abstained"


# ----------------------------------------------------------------------
# 7. Deterministic rendering
# ----------------------------------------------------------------------


def test_render_answer_drops_unsupported_claims_in_verified_mode():
    claims = [
        AnswerClaim(id="c1", text="checkout-service depends on auth-service.", citation_ids=["[a#0]"]),
        AnswerClaim(id="c2", text="checkout-service handles 1 billion requests per day.", citation_ids=["[b#0]"]),
    ]
    verifications = _verifications(("c1", "supported"), ("c2", "insufficient"))
    answer = render_answer(claims, verifications, "partially_verified")
    assert "checkout-service depends on auth-service" in answer
    assert "1 billion requests" not in answer


def test_render_answer_abstains_when_nothing_survives():
    claims = [AnswerClaim(id="c1", text="x", citation_ids=["[a#0]"])]
    verifications = _verifications(("c1", "insufficient"))
    assert render_answer(claims, verifications, "insufficient") == (
        "Insufficient evidence to answer from the indexed sources."
    )


def test_render_answer_abstains_on_contradicted_status():
    claims = [AnswerClaim(id="c1", text="x", citation_ids=["[a#0]"])]
    verifications = _verifications(("c1", "contradicted"))
    assert render_answer(claims, verifications, "contradicted") == (
        "Insufficient evidence to answer from the indexed sources."
    )


def test_render_answer_citation_only_includes_every_claim_unconditionally():
    claims = [AnswerClaim(id="c1", text="checkout-service depends on auth-service.", citation_ids=["[a#0]"])]
    verifications: list = []  # citation_only mode never even needs verdicts
    answer = render_answer(claims, verifications, "citation_valid_only")
    assert "checkout-service depends on auth-service" in answer
    assert "[a#0]" in answer


def _run(verifier: DeterministicVerifier, claims):
    import asyncio
    return asyncio.run(verifier.verify(claims))
