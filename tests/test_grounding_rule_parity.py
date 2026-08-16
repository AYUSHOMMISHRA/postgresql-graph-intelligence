"""LEGACY_DELETION_PLAN.md L4: characterization test binding the two
independent implementations of the citation-only grounding rule.

The rule ("grounded iff the answer carries at least one citation marker and
every marker names retrieved evidence") is implemented independently in two
places:

1. `tenant_engine.answer()`'s `citation_only` branch -- production.
2. `benchmarks/grounding/baseline_runner.citation_only_grounded()` -- a
   frozen "before" measurement, deliberately NOT coupled to production (see
   that module's own docstring): if it imported a shared predicate, the
   baseline would move whenever production changes, hiding exactly the
   drift this test exists to catch.

(`verification.evaluate_policy()`'s `citation_only` branch is a third
mention of this rule in the codebase, but it does not implement the
predicate -- it unconditionally returns "citation_valid_only", trusting
that citation existence was already checked upstream. It is unreachable
from `answer()`, which returns from its own self-contained `citation_only`
branch before `evaluate_policy` is ever called in that mode. See the
comment on that branch in verification.py.)

This test runs a fixed case matrix through both real implementations (not
a third reimplementation) and asserts they agree, so a future change to
either one that silently breaks parity fails a test instead of drifting.

Scope: only the citation-validity semantic that both implementations
actually share. `answer()`'s special-cased abstain_text equality check
(the model returning the literal "Insufficient evidence..." string) has no
analogue in `baseline_runner`, which operates on abstract case data with no
concept of that string -- cases here deliberately avoid triggering it.
"""
import re
import uuid

import pytest

from benchmarks.grounding.baseline_runner import citation_only_grounded
from postgres_graph_rag.models import DEFAULT_INGESTION_CONFIG, DEFAULT_RETRIEVAL_CONFIG
from postgres_graph_rag.tenant_engine import TenantGraphRAG
from unittest.mock import AsyncMock


class _FixedTextExtractor:
    config = {"extraction_model": "offline-test", "embedding_model": "offline", "dimension": 4}
    last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def __init__(self, generated_text: str):
        self._generated_text = generated_text

    async def get_embedding(self, value):
        return [1.0, 0.0, 0.0, 0.0]

    async def generate_text(self, prompt, max_tokens=500):
        return self._generated_text


def _engine(store, generated_text: str) -> TenantGraphRAG:
    return TenantGraphRAG(
        tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        store=store,
        extractor=_FixedTextExtractor(generated_text),
        chunker=lambda text: [text],
        ingestion_config=dict(DEFAULT_INGESTION_CONFIG),
        retrieval_config=dict(DEFAULT_RETRIEVAL_CONFIG),
    )


def _hit(chunk_id: str, source_id: str, ordinal: int, content: str):
    return {
        "id": chunk_id, "document_id": str(uuid.uuid4()), "source_id": source_id,
        "ordinal": ordinal, "content": content, "rrf_score": 1.0, "lex_rank": 1, "sem_rank": 1,
    }


# Test-only glue to build baseline_runner's case-dict shape from the same
# generated text tenant_engine.answer() consumes. This mirrors the literal
# regex tenant_engine.py uses internally, but it is NOT a production
# reimplementation: it exists solely to translate one shared fixture (the
# generated text) into the two systems' different input shapes, the same
# role a test fixture/factory always plays.
_MARKER_RE = re.compile(r"\[([^\[\]\n]+)#(\d+)\]")


def _asserted_citation_ids(generated_text: str) -> list:
    return [f"{source_id}#{ordinal}" for source_id, ordinal in _MARKER_RE.findall(generated_text)]


CASES = [
    pytest.param(
        "auth-service is owned by Identity Team [ownership-auth#0]",
        [("ownership-auth", 0)],
        True,
        id="single_valid_citation",
    ),
    pytest.param(
        "auth-service is owned by Identity Team",
        [("ownership-auth", 0)],
        False,
        id="no_citation_at_all",
    ),
    pytest.param(
        "auth-service is owned by Identity Team [unknown-doc#9]",
        [("ownership-auth", 0)],
        False,
        id="citation_outside_retrieved_evidence",
    ),
    pytest.param(
        "auth-service is owned by Identity Team [ownership-auth#0] and also [unknown-doc#9]",
        [("ownership-auth", 0)],
        False,
        id="one_valid_one_invalid_citation",
    ),
    pytest.param(
        "checkout-service depends on auth-service [ownership-auth#0] and [ownership-auth#1]",
        [("ownership-auth", 0), ("ownership-auth", 1)],
        True,
        id="multiple_valid_citations",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("generated_text,evidence,expected_grounded", CASES)
async def test_production_and_baseline_agree_on_citation_only_grounding(
    generated_text, evidence, expected_grounded,
):
    store = AsyncMock()
    store.hybrid_search.return_value = [
        # Content must contain the question's identifier anchor
        # ("auth-service") -- answer() abstains with "missing_query_anchor"
        # before ever reaching the citation-marker check otherwise, which
        # would test the wrong code path.
        _hit(f"chunk-{i}", source_id, ordinal, "auth-service is owned by Identity Team")
        for i, (source_id, ordinal) in enumerate(evidence)
    ]

    # 1. Production: tenant_engine.answer(), default grounding_mode="citation_only".
    result = await _engine(store, generated_text).answer(
        "Who owns auth-service?", "incidents", mode="hybrid",
    )

    # 2. The frozen baseline: benchmarks/grounding/baseline_runner.citation_only_grounded().
    case = {
        "asserted_citation_ids": _asserted_citation_ids(generated_text),
        "retrieved_evidence": [
            {"source_id": source_id, "ordinal": ordinal} for source_id, ordinal in evidence
        ],
    }
    baseline_result = citation_only_grounded(case)

    assert result.grounded is expected_grounded, (
        f"production disagreed with the expected outcome for: {generated_text!r}"
    )
    assert baseline_result is expected_grounded, (
        f"baseline_runner disagreed with the expected outcome for: {generated_text!r}"
    )
    assert result.grounded == baseline_result, (
        "production and the frozen benchmark baseline must agree on citation-only "
        f"grounding for the same evidence/citations -- production={result.grounded}, "
        f"baseline={baseline_result}, generated_text={generated_text!r}"
    )
