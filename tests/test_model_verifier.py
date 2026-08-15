"""Tests for postgres_graph_rag/model_verifier.py (Release 2 PR 4).
`LLMExtractor.verify_claims` is mocked throughout -- no live API key or
network call -- exercising: deterministic-first short-circuiting (the
model is never called for claims the deterministic layer already decided),
server-side quote validation rejecting a fabricated model quote, provider
failure -> VerifierUnavailableError, and telemetry capture.
"""
from unittest.mock import AsyncMock

import pytest

from postgres_graph_rag.extractor import ModelVerdict
from postgres_graph_rag.grounding import AnswerClaim, VerifierUnavailableError
from postgres_graph_rag.model_verifier import ModelEntailmentVerifier
from postgres_graph_rag.tenant_engine import TenantRetrievedChunk


def chunk(source_id: str, ordinal: int, content: str) -> TenantRetrievedChunk:
    return TenantRetrievedChunk(
        id=f"{source_id}-{ordinal}", document_id=source_id, source_id=source_id,
        ordinal=ordinal, content=content, rrf_score=1.0, lexical_rank=1, semantic_rank=1,
    )


def marker(source_id: str, ordinal: int) -> str:
    return f"[{source_id}#{ordinal}]"


class FakeExtractor:
    """Stands in for LLMExtractor: verify_claims is the only method
    ModelEntailmentVerifier calls on it."""

    def __init__(self, verdicts=None, error=None, usage=None):
        self._verdicts = verdicts or []
        self._error = error
        self.last_usage = usage
        self.verify_claims = AsyncMock(side_effect=self._verify_claims)

    async def _verify_claims(self, prompt: str):
        if self._error is not None:
            raise self._error
        return self._verdicts


@pytest.mark.asyncio
async def test_deterministically_decidable_claims_never_reach_the_model():
    """A claim with no citation is already 'insufficient' (reason_code
    citation_missing) before this layer even exists -- the model must
    never be called for it."""
    evidence = [chunk("doc-1", 0, "Acme makes Widgets.")]
    extractor = FakeExtractor()
    verifier = ModelEntailmentVerifier(extractor=extractor, evidence=evidence)

    claim = AnswerClaim(id="c1", text="Acme makes Widgets.", citation_ids=[])
    [result] = await verifier.verify([claim])

    assert result.verdict == "insufficient"
    assert result.reason_code == "citation_missing"
    extractor.verify_claims.assert_not_called()
    assert verifier.last_telemetry.model_call_made is False


@pytest.mark.asyncio
async def test_explicit_supported_relationship_never_reaches_the_model_either():
    """A claim the deterministic layer can already confirm via a located
    quote also shouldn't pay for a model call."""
    evidence = [chunk("doc-1", 0, "Acme makes Widgets in Ohio.")]
    extractor = FakeExtractor()
    verifier = ModelEntailmentVerifier(extractor=extractor, evidence=evidence)

    claim = AnswerClaim(id="c1", text="Acme makes Widgets.", citation_ids=[marker("doc-1", 0)])
    [result] = await verifier.verify([claim])

    assert result.verdict == "supported"
    extractor.verify_claims.assert_not_called()


@pytest.mark.asyncio
async def test_genuinely_undetermined_claim_is_forwarded_to_the_model():
    """A compound/multi-hop claim the deterministic layer can't confirm
    (reason_code evidence_insufficient specifically) is exactly the case
    that should reach the model."""
    evidence = [
        chunk("doc-1", 0, "checkout-service depends on auth-service."),
        chunk("doc-2", 0, "auth-service is owned by Identity Team."),
    ]
    model_verdicts = [
        ModelVerdict(
            claim_id="c1", verdict="supported",
            supporting_quote="checkout-service depends on auth-service", confidence=0.9,
        ),
    ]
    extractor = FakeExtractor(verdicts=model_verdicts, usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    verifier = ModelEntailmentVerifier(extractor=extractor, evidence=evidence)

    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service, which is owned by Identity Team.",
        citation_ids=[marker("doc-1", 0), marker("doc-2", 0)],
    )
    [result] = await verifier.verify([claim])

    extractor.verify_claims.assert_awaited_once()
    assert result.verdict == "supported"
    assert result.reason_code == "model_entailment_supported"
    assert result.confidence == 0.9
    assert verifier.last_telemetry.model_call_made is True
    assert verifier.last_telemetry.usage == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    assert verifier.last_telemetry.latency_ms is not None


@pytest.mark.asyncio
async def test_fabricated_model_quote_is_downgraded_not_trusted():
    """The model claims 'supported' with a quote that was never actually
    in the cited evidence -- this must be rejected, not passed through."""
    evidence = [
        chunk("doc-1", 0, "checkout-service depends on auth-service."),
        chunk("doc-2", 0, "auth-service is owned by Identity Team."),
    ]
    model_verdicts = [
        ModelVerdict(
            claim_id="c1", verdict="supported",
            supporting_quote="checkout-service is 99.9% reliable thanks to auth-service",  # fabricated
            confidence=0.95,
        ),
    ]
    extractor = FakeExtractor(verdicts=model_verdicts)
    verifier = ModelEntailmentVerifier(extractor=extractor, evidence=evidence)

    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service, which is owned by Identity Team.",
        citation_ids=[marker("doc-1", 0), marker("doc-2", 0)],
    )
    [result] = await verifier.verify([claim])

    assert result.verdict == "insufficient"
    assert result.reason_code == "quote_not_found_verbatim"


@pytest.mark.asyncio
async def test_model_supported_verdict_with_no_quote_is_downgraded():
    evidence = [
        chunk("doc-1", 0, "checkout-service depends on auth-service."),
        chunk("doc-2", 0, "auth-service is owned by Identity Team."),
    ]
    model_verdicts = [ModelVerdict(claim_id="c1", verdict="supported", supporting_quote=None, confidence=0.8)]
    extractor = FakeExtractor(verdicts=model_verdicts)
    verifier = ModelEntailmentVerifier(extractor=extractor, evidence=evidence)

    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service, which is owned by Identity Team.",
        citation_ids=[marker("doc-1", 0), marker("doc-2", 0)],
    )
    [result] = await verifier.verify([claim])
    assert result.verdict == "insufficient"
    assert result.reason_code == "quote_not_found_verbatim"


@pytest.mark.asyncio
async def test_model_contradicted_verdict_with_valid_quote_is_trusted():
    evidence = [
        chunk("doc-1", 0, "auth-service depends on checkout-service, not the other way around."),
        chunk("doc-2", 0, "auth-service is owned by Identity Team."),
    ]
    model_verdicts = [
        ModelVerdict(
            claim_id="c1", verdict="contradicted",
            supporting_quote="auth-service depends on checkout-service", confidence=0.85,
        ),
    ]
    extractor = FakeExtractor(verdicts=model_verdicts)
    verifier = ModelEntailmentVerifier(extractor=extractor, evidence=evidence)

    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service, which is owned by Identity Team.",
        citation_ids=[marker("doc-1", 0), marker("doc-2", 0)],
    )
    [result] = await verifier.verify([claim])
    assert result.verdict == "contradicted"
    assert result.reason_code == "model_entailment_contradicted"


@pytest.mark.asyncio
async def test_provider_failure_raises_verifier_unavailable_error():
    evidence = [
        chunk("doc-1", 0, "checkout-service depends on auth-service."),
        chunk("doc-2", 0, "auth-service is owned by Identity Team."),
    ]
    extractor = FakeExtractor(error=RuntimeError("simulated provider outage"))
    verifier = ModelEntailmentVerifier(extractor=extractor, evidence=evidence)

    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service, which is owned by Identity Team.",
        citation_ids=[marker("doc-1", 0), marker("doc-2", 0)],
    )
    with pytest.raises(VerifierUnavailableError, match="simulated provider outage"):
        await verifier.verify([claim])


@pytest.mark.asyncio
async def test_missing_model_verdict_for_a_claim_defaults_to_insufficient():
    """The model's batch response simply omits one claim_id -- must not
    crash, must not silently default to 'supported'."""
    evidence = [
        chunk("doc-1", 0, "checkout-service depends on auth-service."),
        chunk("doc-2", 0, "auth-service is owned by Identity Team."),
    ]
    extractor = FakeExtractor(verdicts=[])  # model returned zero verdicts
    verifier = ModelEntailmentVerifier(extractor=extractor, evidence=evidence)

    claim = AnswerClaim(
        id="c1", text="checkout-service depends on auth-service, which is owned by Identity Team.",
        citation_ids=[marker("doc-1", 0), marker("doc-2", 0)],
    )
    [result] = await verifier.verify([claim])
    assert result.verdict == "insufficient"


@pytest.mark.asyncio
async def test_mixed_batch_only_forwards_undetermined_claims_to_the_model():
    evidence = [
        chunk("doc-1", 0, "checkout-service depends on auth-service."),
        chunk("doc-2", 0, "auth-service is owned by Identity Team."),
    ]
    model_verdicts = [
        ModelVerdict(
            claim_id="c2", verdict="supported",
            supporting_quote="checkout-service depends on auth-service", confidence=0.9,
        ),
    ]
    extractor = FakeExtractor(verdicts=model_verdicts)
    verifier = ModelEntailmentVerifier(extractor=extractor, evidence=evidence)

    claims = [
        AnswerClaim(id="c1", text="checkout-service depends on auth-service.", citation_ids=[marker("doc-1", 0)]),
        AnswerClaim(
            id="c2", text="checkout-service depends on auth-service, which is owned by Identity Team.",
            citation_ids=[marker("doc-1", 0), marker("doc-2", 0)],
        ),
    ]
    results = await verifier.verify(claims)
    by_id = {r.claim_id: r for r in results}

    assert by_id["c1"].verdict == "supported"
    assert by_id["c1"].reason_code == "explicit_relation_confirmed"  # deterministic, not model
    assert by_id["c2"].verdict == "supported"
    assert by_id["c2"].reason_code == "model_entailment_supported"  # model

    # Only c2 (the undetermined one) was in the prompt sent to the model.
    prompt = extractor.verify_claims.await_args.args[0]
    assert "c2" in prompt
    assert "Claim c1" not in prompt
