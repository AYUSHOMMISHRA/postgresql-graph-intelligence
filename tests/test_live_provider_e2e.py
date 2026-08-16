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
