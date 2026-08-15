from unittest.mock import AsyncMock
import uuid

import pytest

from postgres_graph_rag.models import DEFAULT_INGESTION_CONFIG, DEFAULT_RETRIEVAL_CONFIG
from postgres_graph_rag.tenant_engine import TenantGraphRAG


class FakeExtractor:
    config = {"extraction_model": "offline-test", "embedding_model": "offline", "dimension": 4}
    last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    async def get_embedding(self, value):
        return [1.0, 0.0, 0.0, 0.0]

    async def generate_text(self, prompt, max_tokens=500):
        return "Identity Team owns the dependency [ownership-auth#0]"


def engine(store, extractor=None, **retrieval_overrides):
    return TenantGraphRAG(
        tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        store=store,
        extractor=extractor or FakeExtractor(),
        chunker=lambda text: [text],
        ingestion_config=dict(DEFAULT_INGESTION_CONFIG),
        retrieval_config={**DEFAULT_RETRIEVAL_CONFIG, **retrieval_overrides},
    )


def hit(chunk_id, source_id, content, score):
    return {
        "id": chunk_id,
        "document_id": str(uuid.uuid4()),
        "source_id": source_id,
        "ordinal": 0,
        "content": content,
        "rrf_score": score,
        "lex_rank": 1,
        "sem_rank": 1,
    }


@pytest.mark.asyncio
async def test_retrieval_propagates_chunk_scores_and_ranks_edges():
    store = AsyncMock()
    chunk_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    store.hybrid_search.return_value = [
        hit(chunk_id, "ownership-auth", "auth-service is owned by Identity Team", 0.032)
    ]
    store.mentioned_node_scores_for_chunks.return_value = {"node-a": 1.0, "node-b": 0.4}
    store.traverse_graph.return_value = {
        "nodes": [
            {"id": "node-a", "content": "auth-service", "metadata": {}, "hop_distance": 0, "score": 1.0},
            {"id": "node-b", "content": "Identity Team", "metadata": {}, "hop_distance": 1, "score": 0.7},
            {"id": "node-c", "content": "runbook", "metadata": {}, "hop_distance": 2, "score": 0.2},
        ],
        "edges": [
            {"source_node_id": "node-b", "target_node_id": "node-c", "source_content": "Identity Team", "target_content": "runbook", "relation": "has_runbook", "weight": 0.2},
            {"source_node_id": "node-a", "target_node_id": "node-b", "source_content": "auth-service", "target_content": "Identity Team", "relation": "owned_by", "weight": 3.0},
        ],
    }

    result = await engine(store).retrieve("Who owns auth?", "incidents")

    assert result.trace.seed_scores == {"node-a": 1.0, "node-b": 0.4}
    assert [edge.relation for edge in result.edges] == ["owned_by", "has_runbook"]
    assert result.edges[0].score > result.edges[1].score
    assert store.traverse_graph.call_args.kwargs["seed_scores"] == result.trace.seed_scores


@pytest.mark.asyncio
async def test_context_budget_is_enforced_before_graph_expansion():
    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "large", "x" * 200, 0.04),
        hit("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "small", "short", 0.03),
    ]
    store.mentioned_node_scores_for_chunks.return_value = {}

    result = await engine(store, max_context_tokens=10).retrieve("q", "ns")

    assert [chunk.source_id for chunk in result.chunks] == ["small"]
    assert result.trace.context_truncated is True
    assert result.trace.context_tokens <= 10


@pytest.mark.asyncio
async def test_graph_expansion_seeds_only_the_best_evidence_chunk():
    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "best", "best chunk", 0.04),
        hit("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "decoy", "decoy chunk", 0.03),
    ]
    store.mentioned_node_scores_for_chunks.return_value = {}

    await engine(store).retrieve("q", "ns", top_k=2)

    assert store.mentioned_node_scores_for_chunks.call_args.args[1] == {
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa": 0.04,
    }


@pytest.mark.asyncio
async def test_exact_query_identifier_selects_matching_graph_seed_chunk():
    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "fluent-decoy", "ownership prose", 0.04),
        hit(
            "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "exact-match",
            "checkout-service-001 depends_on auth-service-001",
            0.03,
        ),
    ]
    store.mentioned_node_scores_for_chunks.return_value = {}

    await engine(store).retrieve("Who owns checkout-service-001?", "ns", top_k=2)

    assert store.mentioned_node_scores_for_chunks.call_args.args[1] == {
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb": 0.03,
    }


@pytest.mark.asyncio
async def test_graph_edge_evidence_is_added_as_citable_context():
    class GraphAnswerExtractor(FakeExtractor):
        async def generate_text(self, prompt, max_tokens=500):
            assert "[ownership-001#0] auth-service-001 owned_by Identity Team 001" in prompt
            return "Identity Team 001 owns it [ownership-001#0]"

    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "deployment",
            "checkout-service-001 depends_on auth-service-001",
            0.04,
        )
    ]
    store.mentioned_node_scores_for_chunks.return_value = {"node-a": 1.0}
    store.traverse_graph.return_value = {
        "nodes": [
            {"id": "node-a", "content": "auth-service-001", "metadata": {}, "hop_distance": 0, "score": 1.0},
            {"id": "node-b", "content": "Identity Team 001", "metadata": {}, "hop_distance": 1, "score": 0.7},
        ],
        "edges": [{
            "id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
            "source_node_id": "node-a", "target_node_id": "node-b",
            "source_content": "auth-service-001", "target_content": "Identity Team 001",
            "relation": "owned_by", "weight": 1.0,
        }],
    }
    store.get_edges_evidence.return_value = [{
        "chunk_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "document_id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
        "source_id": "ownership-001", "ordinal": 0,
        "content": "auth-service-001 owned_by Identity Team 001",
    }]

    answer = await engine(store, extractor=GraphAnswerExtractor()).answer(
        "Who owns checkout-service-001?", "ns", top_k=1, hops=2
    )

    assert answer.grounded is True
    assert [citation.source_id for citation in answer.citations] == ["ownership-001"]
    assert [chunk.source_id for chunk in answer.retrieval.chunks] == ["deployment", "ownership-001"]
    store.get_edges_evidence.assert_awaited_once()


@pytest.mark.asyncio
async def test_answer_returns_only_validated_citations():
    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "ownership-auth",
            "auth-service is owned by Identity Team",
            0.032,
        )
    ]

    result = await engine(store).answer(
        "Who owns auth-service?", "incidents", mode="hybrid"
    )

    assert result.grounded is True
    assert result.citations[0].source_id == "ownership-auth"
    assert result.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_answer_abstains_on_unknown_model_citation():
    class BadExtractor(FakeExtractor):
        async def generate_text(self, prompt, max_tokens=500):
            return "Made up claim [unknown-document#9]"

    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "ownership-auth",
            "auth-service is owned by Identity Team",
            0.032,
        )
    ]

    result = await engine(store, extractor=BadExtractor()).answer(
        "Who owns auth-service?", "incidents", mode="hybrid"
    )

    assert result.grounded is False
    assert result.citations == []
    assert result.answer.startswith("Insufficient evidence")


@pytest.mark.asyncio
async def test_answer_abstains_when_question_identifier_is_not_in_evidence():
    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "checkout",
            "checkout-service-001 depends_on auth-service-001",
            0.032,
        )
    ]

    result = await engine(store).answer(
        "Which team owns database-001 used by checkout-service-001?",
        "incidents",
        mode="hybrid",
    )

    assert result.grounded is False
    assert result.usage == {}
    assert result.abstain_reason == "missing_query_anchor"


@pytest.mark.asyncio
async def test_answer_does_not_abstain_on_hyphen_vs_space_identifier_mismatch():
    """Regression test for a real bug found on a live end-to-end run: a
    question naming "checkout-service" (hyphenated) abstained with
    "Insufficient evidence" even though the retrieved evidence plainly
    described "checkout service" (spaced) — the anchor check's exact
    substring match was stricter than the anti-hallucination guarantee it's
    meant to enforce. Fixed by normalizing separators before comparing."""
    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "checkout-doc",
            "The checkout service depends on the authentication service.",
            0.032,
        )
    ]

    class Extractor(FakeExtractor):
        async def generate_text(self, prompt, max_tokens=500):
            return "The checkout service depends on the auth service. [checkout-doc#0]"

    result = await engine(store, extractor=Extractor()).answer(
        "What does checkout-service depend on?", "incidents", mode="hybrid"
    )

    assert result.grounded is True
    assert result.abstain_reason is None
    assert result.citations[0].source_id == "checkout-doc"


@pytest.mark.asyncio
async def test_answer_result_reports_no_evidence_reason():
    store = AsyncMock()
    store.hybrid_search.return_value = []
    store.mentioned_node_scores_for_chunks.return_value = {}

    result = await engine(store).answer("Who owns auth-service?", "incidents", mode="hybrid")

    assert result.grounded is False
    assert result.abstain_reason == "no_evidence_retrieved"


@pytest.mark.asyncio
async def test_answer_result_reports_invalid_citation_reason():
    class BadExtractor(FakeExtractor):
        async def generate_text(self, prompt, max_tokens=500):
            return "Made up claim [unknown-document#9]"

    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "ownership-auth",
            "auth-service is owned by Identity Team",
            0.032,
        )
    ]

    result = await engine(store, extractor=BadExtractor()).answer(
        "Who owns auth-service?", "incidents", mode="hybrid"
    )

    assert result.grounded is False
    assert result.abstain_reason == "invalid_or_missing_citation"


@pytest.mark.asyncio
async def test_answer_result_reports_model_abstained_reason():
    class AbstainingExtractor(FakeExtractor):
        async def generate_text(self, prompt, max_tokens=500):
            return "Insufficient evidence to answer from the indexed sources."

    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "ownership-auth",
            "auth-service is owned by Identity Team",
            0.032,
        )
    ]

    result = await engine(store, extractor=AbstainingExtractor()).answer(
        "Who owns auth-service?", "incidents", mode="hybrid"
    )

    assert result.grounded is False
    assert result.abstain_reason == "model_abstained"


@pytest.mark.asyncio
async def test_answer_result_reports_no_abstain_reason_when_grounded():
    result = await engine(AsyncMock(hybrid_search=AsyncMock(return_value=[
        hit("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "ownership-auth", "auth-service is owned by Identity Team", 0.032)
    ]))).answer("Who owns auth-service?", "incidents", mode="hybrid")

    assert result.grounded is True
    assert result.abstain_reason is None


@pytest.mark.asyncio
async def test_answer_result_reports_empty_model_response_reason():
    """Regression test for a real bug found against the live OpenAI API: a
    reasoning-tier model (gpt-5.6-luna) can consume its entire completion-
    token budget on invisible reasoning tokens and return an empty visible
    completion, with finish_reason still "stop" — indistinguishable from a
    badly-cited response unless empty output is checked for specifically.
    The real fix is a larger max_answer_tokens default; this test locks in
    that empty output gets its own distinct abstain_reason rather than
    being folded into "invalid_or_missing_citation"."""
    class EmptyExtractor(FakeExtractor):
        async def generate_text(self, prompt, max_tokens=500):
            return ""

    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "ownership-auth",
            "auth-service is owned by Identity Team",
            0.032,
        )
    ]

    result = await engine(store, extractor=EmptyExtractor()).answer(
        "Who owns auth-service?", "incidents", mode="hybrid"
    )

    assert result.grounded is False
    assert result.abstain_reason == "empty_model_response"
    assert result.answer.startswith("Insufficient evidence")


@pytest.mark.asyncio
async def test_answer_max_answer_tokens_is_configurable_and_forwarded():
    captured = {}

    class RecordingExtractor(FakeExtractor):
        async def generate_text(self, prompt, max_tokens=500):
            captured["max_tokens"] = max_tokens
            return "Identity Team owns the dependency [ownership-auth#0]"

    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "ownership-auth", "auth-service is owned by Identity Team", 0.032)
    ]

    await engine(store, extractor=RecordingExtractor()).answer(
        "Who owns auth-service?", "incidents", mode="hybrid", max_answer_tokens=3000
    )

    assert captured["max_tokens"] == 3000


@pytest.mark.asyncio
async def test_answer_retries_once_with_larger_budget_on_empty_response():
    """Regression test for a real bug found against the live OpenAI API:
    reasoning-token consumption for a fixed prompt is not deterministic —
    the *same* question and evidence sometimes returned a full cited
    answer and sometimes an entirely empty completion at the same token
    budget. A single bounded retry at double the budget recovered most of
    these in practice; this locks in that the retry actually happens and
    is reported via the `retried_with_larger_budget` observability flag."""
    calls = []

    class FlakyExtractor(FakeExtractor):
        async def generate_text(self, prompt, max_tokens=500):
            calls.append(max_tokens)
            if len(calls) == 1:
                return ""
            return "Identity Team owns the dependency [ownership-auth#0]"

    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "ownership-auth", "auth-service is owned by Identity Team", 0.032)
    ]

    result = await engine(store, extractor=FlakyExtractor()).answer(
        "Who owns auth-service?", "incidents", mode="hybrid", max_answer_tokens=1500
    )

    assert calls == [1500, 3000]
    assert result.grounded is True
    assert result.abstain_reason is None


@pytest.mark.asyncio
async def test_answer_reports_empty_model_response_if_retry_also_empty():
    class AlwaysEmptyExtractor(FakeExtractor):
        async def generate_text(self, prompt, max_tokens=500):
            return ""

    store = AsyncMock()
    store.hybrid_search.return_value = [
        hit("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "ownership-auth", "auth-service is owned by Identity Team", 0.032)
    ]

    result = await engine(store, extractor=AlwaysEmptyExtractor()).answer(
        "Who owns auth-service?", "incidents", mode="hybrid"
    )

    assert result.grounded is False
    assert result.abstain_reason == "empty_model_response"
