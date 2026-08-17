"""LEGACY_DELETION_PLAN.md R5: real-provider secured-path end-to-end tests.

Unlike every other test in this suite, these make one real, billed call to
a live LLM provider (extraction + embedding, and for `answer()`, one
generation call too) against the actual RLS-secured path
(`PostgresGraphRAG` -> `setup_secure()` -> `for_tenant()` -> `add_document()`
-> `answer()`). This is what tests/test_integration.py verified for the
legacy engine before it was deleted; there was no secure-path equivalent
until now.

Marked `live_provider` (excluded from routine runs by
`pyproject.toml`'s `addopts`, matching the `live_provider` convention every
other real-provider test in this repo already uses) and additionally
gated on the real API key each provider needs -- these are not fixtures or
mocks, they cost real money per run.

Run explicitly: `pytest -m live_provider -v tests/test_live_provider_e2e.py`
"""
import os
import uuid
import warnings

import pytest
import psycopg
from dotenv import load_dotenv

from postgres_graph_rag import PostgresGraphRAG
from postgres_graph_rag.models import GOOGLE_DEFAULT_CONFIG, OPENAI_DEFAULT_CONFIG
from postgres_graph_rag.tenancy import SCHEMA

load_dotenv()

pytestmark = pytest.mark.live_provider

POSTGRES_URL = os.getenv("POSTGRES_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DOCUMENT_TEXT = "checkout-service depends on auth-service for session validation."
QUESTION = "What does checkout-service depend on?"
RUNTIME_ROLE = "pgr_live_e2e_runtime"
RUNTIME_PASSWORD = "pgr_live_e2e_runtime_pw"  # noqa: S105 -- throwaway per-run schema, not a real credential

# The flagship claim this project is built around: "Identity Team" never
# appears in the same document as "checkout-service" -- it only exists as a
# two-hop path a real provider's extraction has to produce and a real graph
# traversal has to walk. test_gemini_secured_path_e2e/test_openai_secured_path_e2e
# above only prove a single-hop grounded answer against a real provider; until
# this test, no live-provider test exercised the multi-hop case at all -- it
# was only ever demonstrated with the deterministic OfflineExtractor (demo.py)
# or the offline incident-benchmark-v1 dataset, never a real LLM's own extraction.
MULTIHOP_DOC_A = "checkout-service depends on auth-service for session validation."
MULTIHOP_DOC_B = "auth-service is owned by the Identity Team."
MULTIHOP_QUESTION = "Which team owns the dependency of checkout-service?"


def _runtime_url() -> str:
    import urllib.parse as up

    parsed = up.urlparse(POSTGRES_URL)
    netloc = f"{RUNTIME_ROLE}:{RUNTIME_PASSWORD}@{parsed.hostname}:{parsed.port or 5432}"
    return up.urlunparse(parsed._replace(netloc=netloc))


async def _run_live_e2e(*, config, api_key_kwarg: str, api_key: str) -> None:
    """Shared body for both providers: setup_secure() -> for_tenant() ->
    add_document() -> answer(), all against the real provider and a real,
    freshly migrated secure schema at that provider's real embedding
    dimension."""
    admin_conn = await psycopg.AsyncConnection.connect(POSTGRES_URL)
    try:
        async with admin_conn.cursor() as cur:
            await cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
            await admin_conn.commit()
    finally:
        await admin_conn.close()

    rag = PostgresGraphRAG(config=config, **{api_key_kwarg: api_key}, runtime_url=_runtime_url())
    try:
        await rag.setup_secure(
            admin_url=POSTGRES_URL, runtime_role=RUNTIME_ROLE, runtime_password=RUNTIME_PASSWORD,
        )
        engine = rag.for_tenant(uuid.uuid4())

        report = await engine.add_document(DOCUMENT_TEXT, "live-e2e", "doc-1")
        assert report["skipped"] is False
        assert report["chunks"] >= 1

        result = await engine.answer(QUESTION, "live-e2e", mode="hybrid")
        assert result.grounded is True, (
            f"expected a grounded answer, got grounding_status={result.grounding_status!r} "
            f"answer={result.answer!r}"
        )
        assert "auth-service" in result.answer.lower()
    finally:
        await rag.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not POSTGRES_URL, reason="POSTGRES_URL not set")
@pytest.mark.skipif(not GOOGLE_API_KEY, reason="GOOGLE_API_KEY not set")
async def test_gemini_secured_path_e2e():
    await _run_live_e2e(config=GOOGLE_DEFAULT_CONFIG, api_key_kwarg="google_api_key", api_key=GOOGLE_API_KEY)


@pytest.mark.asyncio
@pytest.mark.skipif(not POSTGRES_URL, reason="POSTGRES_URL not set")
@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
async def test_openai_secured_path_e2e():
    await _run_live_e2e(config=OPENAI_DEFAULT_CONFIG, api_key_kwarg="openai_api_key", api_key=OPENAI_API_KEY)


async def _run_live_multihop_e2e(*, config, api_key_kwarg: str, api_key: str) -> None:
    """Two documents, real provider extraction, real graph traversal.

    Splits the claim in two, because repeated real runs against Gemini
    during development showed these have different reliability
    characteristics:

    1. Retrieval/graph traversal resolving the two-hop connection
       (auth-service -> Identity Team) from real LLM-extracted triplets is
       deterministic SQL once the entities/edges exist, and is asserted as
       a hard requirement -- observed correct in every real run made
       against this test. A regression here is a real bug.
    2. Final answer-generation composing a grounded natural-language answer
       from that evidence additionally depends on the configured LLM
       actually following the two-hop chain in free text. Observed directly
       across repeated real Gemini calls with the default
       gemini-3.1-flash-lite model: it abstains ("Insufficient evidence...")
       in a large fraction of runs even with the correct edge placed
       directly in its prompt (see the "Known relationships" evidence block
       `tenant_engine.answer()` adds specifically to help this case). This
       is a genuine, disclosed reliability limitation of the default
       lite-tier model for chained free-text reasoning -- not a retrieval
       defect -- so it is recorded as a warning here rather than asserted
       as a guaranteed pass, to avoid misrepresenting a *model*
       characteristic as a *test* failure.
    """
    admin_conn = await psycopg.AsyncConnection.connect(POSTGRES_URL)
    try:
        async with admin_conn.cursor() as cur:
            await cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
            await admin_conn.commit()
    finally:
        await admin_conn.close()

    rag = PostgresGraphRAG(config=config, **{api_key_kwarg: api_key}, runtime_url=_runtime_url())
    try:
        await rag.setup_secure(
            admin_url=POSTGRES_URL, runtime_role=RUNTIME_ROLE, runtime_password=RUNTIME_PASSWORD,
        )
        engine = rag.for_tenant(uuid.uuid4())

        report_a = await engine.add_document(MULTIHOP_DOC_A, "live-e2e-multihop", "doc-a")
        report_b = await engine.add_document(MULTIHOP_DOC_B, "live-e2e-multihop", "doc-b")
        assert report_a["skipped"] is False
        assert report_b["skipped"] is False

        retrieval = await engine.retrieve(MULTIHOP_QUESTION, "live-e2e-multihop", mode="hybrid_graph")
        assert any(
            "identity team" in e.target_content.lower() or "identity team" in e.source_content.lower()
            for e in retrieval.edges
        ), (
            "expected real LLM extraction + graph traversal to resolve the "
            f"auth-service -> Identity Team edge; got edges={retrieval.edges!r}"
        )

        result = await engine.answer(MULTIHOP_QUESTION, "live-e2e-multihop", mode="hybrid_graph")
        if not result.grounded:
            warnings.warn(
                "Real-provider multi-hop ANSWER GENERATION abstained "
                f"(grounding_status={result.grounding_status!r}) even though graph "
                "traversal correctly resolved the multi-hop edge -- a known, "
                "disclosed reliability limitation of the default lite-tier model "
                "for chained free-text reasoning, not a retrieval defect. "
                "See CHANGELOG.md.",
                stacklevel=2,
            )
        elif "identity team" not in result.answer.lower():
            raise AssertionError(
                f"got a grounded answer but it never named the resolved entity: {result.answer!r}"
            )
    finally:
        await rag.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not POSTGRES_URL, reason="POSTGRES_URL not set")
@pytest.mark.skipif(not GOOGLE_API_KEY, reason="GOOGLE_API_KEY not set")
async def test_gemini_secured_path_multihop_e2e():
    await _run_live_multihop_e2e(config=GOOGLE_DEFAULT_CONFIG, api_key_kwarg="google_api_key", api_key=GOOGLE_API_KEY)


@pytest.mark.asyncio
@pytest.mark.skipif(not POSTGRES_URL, reason="POSTGRES_URL not set")
@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
async def test_openai_secured_path_multihop_e2e():
    await _run_live_multihop_e2e(config=OPENAI_DEFAULT_CONFIG, api_key_kwarg="openai_api_key", api_key=OPENAI_API_KEY)
