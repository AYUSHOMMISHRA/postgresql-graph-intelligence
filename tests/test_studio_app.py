"""Tests for postgres_graph_rag/studio_app.py -- FastAPI TestClient only,
no real Postgres connection and no real provider call. Mocks the same
I/O-performing TenantGraphRAG methods as tests/test_playground.py, for the
same reason (SecureGraphStore's constructor does no network I/O, so a real
PostgresGraphRAG/TenantGraphRAG can be exercised safely with only those
methods patched).
"""
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed -- install the 'studio' extra (uv sync --extra studio)")
from fastapi.testclient import TestClient  # noqa: E402

from postgres_graph_rag import playground_service  # noqa: E402
from postgres_graph_rag.studio_app import create_app  # noqa: E402
from postgres_graph_rag.tenant_engine import (  # noqa: E402
    AnswerResult,
    Citation,
    TenantRetrievalResult,
    TenantRetrievedChunk,
)

FAKE_RUNTIME_URL = "postgresql://pgr_runtime:pw@localhost:5432/graph_rag_test"

_READY_1536 = {"ready": True, "embedding_dimension": 1536, "vector_type": "vector",
               "schema_version": 7, "updated_at": "2026-01-01T00:00:00+00:00"}


@pytest.fixture(autouse=True)
def _mock_schema_status():
    """B1's preflight calls a real SecureGraphStore.schema_status() before
    any ingest -- mocked here as an offline/OpenAI-compatible (1536-dim)
    schema, applied to every test in this file by default."""
    with patch(
        "postgres_graph_rag.tenancy.SecureGraphStore.schema_status",
        new=AsyncMock(return_value=dict(_READY_1536)),
    ):
        yield


def _client(provider: str = "offline", **kwargs) -> TestClient:
    return TestClient(create_app(runtime_url=FAKE_RUNTIME_URL, provider=provider, **kwargs))


def _fake_answer(answer="Identity Team [studio-doc-1#0]", mode="hybrid_graph"):
    return AnswerResult(
        answer=answer,
        citations=[Citation(source_id="studio-doc-1", chunk_id="c1", ordinal=0, excerpt="...")],
        retrieval=TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None),
        usage={"total_tokens": 10}, latency_ms=5.0,
        grounding_mode="citation_only", grounding_status="citation_valid_only",
    )


def test_index_renders_with_no_credentials_in_body():
    client = _client()
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert FAKE_RUNTIME_URL not in body
    assert "pw@localhost" not in body
    assert "PGR_RUNTIME_URL" not in body or "not configured" not in body  # only shown if actually missing


def test_index_warns_when_runtime_url_missing():
    client = TestClient(create_app(runtime_url=None))
    resp = client.get("/")
    assert "not configured" in resp.text


def test_trusted_host_middleware_rejects_unrecognized_host_header():
    """B8: a second, independent layer against DNS-rebinding -- even though
    the Studio should only ever be bound to loopback (enforced in
    studio_cli.py), a request whose Host header isn't one of the allowed
    values must be rejected before it reaches any route."""
    client = _client()
    resp = client.get("/", headers={"Host": "evil.example.com"})
    assert resp.status_code == 400


def test_ask_without_a_document_shows_error_not_500():
    client = _client()
    resp = client.post("/ask", data={"question": "anything?", "mode": "hybrid_graph"})
    assert resp.status_code == 200
    assert "Add at least one document" in resp.text


def test_ingest_then_ask_happy_path(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = _client(provider="openai", openai_api_key="sk-test-not-real")
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ):
        ingest_resp = client.post("/ingest", data={"text": "Checkout depends_on Auth."})
    assert ingest_resp.status_code == 200
    assert "documents added: 1" in ingest_resp.text

    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.answer",
        new=AsyncMock(return_value=_fake_answer()),
    ) as mocked_answer:
        ask_resp = client.post("/ask", data={"question": "Who owns it?", "mode": "hybrid_graph"})
    mocked_answer.assert_awaited_once()
    assert "Identity Team" in ask_resp.text


def test_ingest_offline_plain_prose_shows_warning():
    client = _client()
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 0}),
    ):
        resp = client.post("/ingest", data={"text": "ordinary prose with no fixed predicate"})
    assert "0 relationships" in resp.text


def test_compare_ingests_once_and_retrieves_three_modes_for_free(monkeypatch):
    """/compare is the free comparison: retrieval only, no answer-generation
    call, for any provider -- see retrieve_playground_evidence(). Only
    /compare-answers (a separate, explicit action) calls answer()."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = _client(provider="openai", openai_api_key="sk-test-not-real")
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ) as mocked_ingest:
        client.post("/ingest", data={"text": "Checkout depends_on Auth."})

    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve",
        new=AsyncMock(return_value=TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None)),
    ) as mocked_retrieve, patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.answer", new=AsyncMock()
    ) as mocked_answer:
        resp = client.post("/compare", data={"question": "Who owns it?"})

    mocked_ingest.assert_awaited_once()  # /compare must not re-ingest
    assert mocked_retrieve.await_count == 3  # vector, hybrid, hybrid_graph
    mocked_answer.assert_not_awaited()  # the free comparison must never generate an answer
    assert resp.status_code == 200
    assert "vector" in resp.text and "hybrid_graph" in resp.text


def test_compare_renders_distinct_retrieval_evidence_per_mode(monkeypatch):
    """B5: /compare must visibly render each mode's own retrieved evidence
    (passages/entities/relationships, via retrieval_context), not just the
    no-answer note -- distinct mocked evidence per mode must each appear in
    the rendered response."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = _client(provider="openai", openai_api_key="sk-test-not-real")
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ):
        client.post("/ingest", data={"text": "Checkout depends_on Auth."})

    async def fake_retrieve(self, question, namespace, **overrides):
        mode = overrides["mode"]
        chunk = TenantRetrievedChunk(
            id=f"c-{mode}", document_id="doc-1", source_id="doc-1", ordinal=0,
            content=f"Distinctive-Evidence-For-{mode}", rrf_score=1.0,
            lexical_rank=1, semantic_rank=1,
        )
        return TenantRetrievalResult(chunks=[chunk], nodes=[], edges=[], trace=None)

    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve", new=fake_retrieve,
    ):
        resp = client.post("/compare", data={"question": "Who owns it?"})

    assert resp.status_code == 200
    for mode in ("vector", "hybrid", "hybrid_graph"):
        assert f"Distinctive-Evidence-For-{mode}" in resp.text
    assert resp.text.count("No answer generated for this mode") == 3


def test_compare_answers_is_a_separate_explicit_paid_action(monkeypatch):
    """/compare-answers is the opt-in, 3x-generation-call comparison --
    distinct from /compare, which never calls answer()."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = _client(provider="openai", openai_api_key="sk-test-not-real")
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ):
        client.post("/ingest", data={"text": "Checkout depends_on Auth."})

    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.answer",
        new=AsyncMock(return_value=_fake_answer()),
    ) as mocked_answer:
        resp = client.post("/compare-answers", data={"question": "Who owns it?"})

    assert mocked_answer.await_count == 3
    assert resp.status_code == 200
    assert "Identity Team" in resp.text


def test_ingest_ignores_a_posted_provider_field_since_provider_is_fixed_per_process(monkeypatch):
    """B3: provider is fixed for the whole Studio process (set once via
    `postgres-graph-rag-studio --provider`), not chosen per request -- the
    UI no longer offers a provider dropdown. If a client posts a `provider`
    form field anyway (no such form control exists in the current template,
    but nothing stops a raw HTTP client from trying), it must be silently
    ignored rather than switching providers for that request."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = _client(provider="offline")
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ) as mocked_ingest:
        resp = client.post("/ingest", data={"text": "Checkout depends_on Auth.", "provider": "openai"})

    assert resp.status_code == 200
    assert "documents added: 1" in resp.text
    mocked_ingest.assert_awaited_once()


def test_second_document_accumulates_in_the_same_session_scope():
    """The multi-hop scenario requires two documents to land in the same
    tenant/namespace -- confirms /ingest called twice reuses the session's
    scope rather than creating a second isolated one."""
    client = _client()
    captured_namespaces = []

    async def fake_ingest(self, text, namespace, source_id):
        captured_namespaces.append(namespace)
        return {"skipped": False, "chunks": 1, "triplets": 1}

    with patch("postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed", new=fake_ingest):
        client.post("/ingest", data={"text": "checkout-service depends_on auth-service.", "provider": "offline"})
        client.post("/ingest", data={"text": "auth-service is owned_by Identity Team.", "provider": "offline"})

    assert len(captured_namespaces) == 2
    assert captured_namespaces[0] == captured_namespaces[1]


def test_clear_deletes_only_this_sessions_documents():
    client = _client()
    captured_namespaces = []

    async def fake_ingest(self, text, namespace, source_id):
        captured_namespaces.append(namespace)
        return {"skipped": False, "chunks": 1, "triplets": 1}

    with patch("postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed", new=fake_ingest):
        client.post("/ingest", data={"text": "Checkout depends_on Auth.", "provider": "offline"})

    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.delete_document", new=AsyncMock()
    ) as mocked_delete:
        resp = client.post("/clear")

    # patch.object with `new=AsyncMock()` replaces the class attribute with a
    # plain object, so no implicit `self` is bound -- the call args are
    # exactly the (namespace, source_id) the engine method itself passes.
    mocked_delete.assert_awaited_once_with(captured_namespaces[0], "studio-doc-1")
    assert "documents added: 0" in resp.text


def test_clear_reports_failure_honestly_and_preserves_session_state():
    """B12: if removing this session's documents fails, /clear must report
    that honestly and leave the session's visible state (its document
    count) unchanged -- not silently reset it as if the removal succeeded."""
    client = _client()
    with patch("postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed", new=AsyncMock(
        return_value={"skipped": False, "chunks": 1, "triplets": 1},
    )):
        client.post("/ingest", data={"text": "Checkout depends_on Auth."})

    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.delete_document",
        new=AsyncMock(side_effect=RuntimeError("connection reset")),
    ):
        resp = client.post("/clear")

    assert resp.status_code == 200
    assert "Failed to remove this session" in resp.text
    assert "connection reset" not in resp.text  # no raw exception text/traceback to the page
    assert "documents added: 1" in resp.text  # state was NOT reset after the failed removal


def test_unexpected_error_returns_generic_message_not_a_traceback():
    """B12: a genuinely unexpected failure (not PlaygroundInputError) must
    never leak a raw traceback, DSN, key, or SQL to the client -- only a
    generic message with a request id for server-side log correlation.

    `raise_server_exceptions=False`: TestClient's default re-raises any
    exception in the test process regardless of a registered exception
    handler (so app bugs aren't silently swallowed by other tests) --
    exactly the response-generation behavior this test verifies, so it must
    be disabled here specifically.
    """
    client = TestClient(
        create_app(runtime_url=FAKE_RUNTIME_URL, provider="offline"), raise_server_exceptions=False,
    )
    with patch("postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed", new=AsyncMock(
        return_value={"skipped": False, "chunks": 1, "triplets": 1},
    )):
        client.post("/ingest", data={"text": "Checkout depends_on Auth."})

    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve",
        new=AsyncMock(side_effect=RuntimeError(f"secret leak test {FAKE_RUNTIME_URL}")),
    ):
        resp = client.post("/ask", data={"question": "Who owns it?"})

    assert resp.status_code == 500
    assert "request id" in resp.text
    assert FAKE_RUNTIME_URL not in resp.text
    assert "Traceback" not in resp.text


@pytest.mark.asyncio
async def test_concurrent_asks_on_the_same_session_do_not_interleave(monkeypatch):
    """B6: a per-session async lock must serialize ingest/ask/compare/
    compare-answers/clear -- two concurrent requests against the *same*
    session must run one after another, never overlapping. Without this, a
    double-clicked paid comparison could fire its generation calls twice
    concurrently, or an ask could race a document removal."""
    import asyncio

    import httpx

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    app = create_app(runtime_url=FAKE_RUNTIME_URL, provider="openai", openai_api_key="sk-test-not-real")
    concurrency = {"in_flight": 0, "max_in_flight": 0}

    async def fake_ingest(self, text, namespace, source_id):
        return {"skipped": False, "chunks": 1, "triplets": 1}

    async def fake_answer(self, question, namespace, **kwargs):
        concurrency["in_flight"] += 1
        concurrency["max_in_flight"] = max(concurrency["max_in_flight"], concurrency["in_flight"])
        await asyncio.sleep(0.05)
        concurrency["in_flight"] -= 1
        return _fake_answer()

    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed", new=fake_ingest,
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.answer", new=fake_answer,
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            await client.post("/ingest", data={"text": "Checkout depends_on Auth."})
            await asyncio.gather(
                client.post("/ask", data={"question": "q1"}),
                client.post("/ask", data={"question": "q2"}),
            )

    assert concurrency["max_in_flight"] == 1


def test_ttl_eviction_cleans_up_the_sessions_own_documents(monkeypatch):
    """B9: TTL/LRU eviction must attempt to remove the evicted session's own
    documents from PostgreSQL before dropping its in-memory state -- not
    just make them unreachable in memory while leaving them in the
    database indefinitely."""
    from postgres_graph_rag import studio_app as studio_app_module

    monkeypatch.setattr(studio_app_module, "_SESSION_TTL_SECONDS", 0)
    client = _client()
    captured_namespaces = []

    async def fake_ingest(self, text, namespace, source_id):
        captured_namespaces.append(namespace)
        return {"skipped": False, "chunks": 1, "triplets": 1}

    with patch("postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed", new=fake_ingest):
        client.post("/ingest", data={"text": "Checkout depends_on Auth."})

    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.delete_document", new=AsyncMock()
    ) as mocked_delete:
        # Any request triggers a sweep; TTL=0 means the session created
        # above is already expired by the time this one runs.
        resp = client.get("/")

    mocked_delete.assert_awaited_once_with(captured_namespaces[0], "studio-doc-1")
    # A fresh session was created for this request -- the evicted one's
    # document count must not leak into it.
    assert "documents added: 0" in resp.text


def test_shutdown_cleans_up_remaining_sessions_own_documents():
    """B9: graceful shutdown must attempt best-effort document cleanup for
    every still-open session, not only for sessions that were TTL/LRU-
    evicted or explicitly cleared."""
    captured_namespaces = []

    async def fake_ingest(self, text, namespace, source_id):
        captured_namespaces.append(namespace)
        return {"skipped": False, "chunks": 1, "triplets": 1}

    with patch("postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed", new=fake_ingest), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.delete_document", new=AsyncMock()
    ) as mocked_delete:
        with TestClient(create_app(runtime_url=FAKE_RUNTIME_URL, provider="offline")) as client:
            client.post("/ingest", data={"text": "Checkout depends_on Auth."})
            mocked_delete.assert_not_awaited()
        # Lifespan shutdown ran on __exit__ above.
        mocked_delete.assert_awaited_once_with(captured_namespaces[0], "studio-doc-1")


def test_ingest_rejects_simultaneous_text_and_file(monkeypatch):
    """B7: pasted text and an uploaded file must never be silently
    resolved in favor of one -- both present is a clean, explicit error."""
    client = _client()
    resp = client.post(
        "/ingest",
        data={"text": "Checkout depends_on Auth."},
        files={"file": ("doc.txt", b"Auth depends_on Identity.", "text/plain")},
    )
    assert resp.status_code == 200
    assert "Pass either pasted text or an uploaded file, not both" in resp.text


def test_ingest_rejects_disallowed_upload_extension():
    client = _client()
    resp = client.post(
        "/ingest",
        data={"text": ""},
        files={"file": ("doc.exe", b"whatever", "application/octet-stream")},
    )
    assert resp.status_code == 200
    assert "Only .txt/.md files are accepted" in resp.text


def test_ingest_rejects_oversized_upload_without_buffering_it_all():
    """B7: an upload past the playground's character cap must be rejected
    cleanly -- the in-handler read is bounded to MAX_CHARS + 1, so this
    must reject deterministically regardless of how large the file is."""
    client = _client()
    oversized = b"x" * (playground_service.PLAYGROUND_MAX_CHARS + 1000)
    resp = client.post(
        "/ingest",
        data={"text": ""},
        files={"file": ("doc.txt", oversized, "text/plain")},
    )
    assert resp.status_code == 200
    assert "exceeds the" in resp.text and "character playground limit" in resp.text


def test_ingest_rejects_oversized_request_body_via_content_length(monkeypatch):
    """B7: a declared oversized /ingest request body must be rejected by a
    Content-Length check before Starlette's multipart parser buffers it --
    a bounded `file.read(n)` inside the route alone can't prevent that
    buffering, since form parsing happens before the route function runs."""
    import asyncio

    import httpx

    app = create_app(runtime_url=FAKE_RUNTIME_URL, provider="offline")

    async def _run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            headers = {"content-length": str(10 * 1024 * 1024)}
            return await client.post("/ingest", data={"text": "x"}, headers=headers)

    resp = asyncio.run(_run())
    assert resp.status_code == 413


def test_invalid_triplets_json_shows_error_not_500():
    client = _client()
    resp = client.post("/ingest", data={
        "text": "some text", "provider": "offline", "triplets_json": "{not valid",
    })
    assert resp.status_code == 200
    assert "Invalid triplets JSON" in resp.text


@pytest.mark.parametrize("provider,env_var", [("openai", "OPENAI_API_KEY"), ("gemini", "GOOGLE_API_KEY")])
def test_missing_provider_key_fails_at_startup_not_first_request(monkeypatch, provider, env_var):
    """B3: a live provider's missing credentials must fail when the Studio
    process starts (create_app()/build_extractor()), not silently wait
    until the first /ingest request to surface as a form error."""
    monkeypatch.delenv(env_var, raising=False)
    from postgres_graph_rag.playground_service import PlaygroundInputError
    with pytest.raises(PlaygroundInputError, match=env_var):
        create_app(runtime_url=FAKE_RUNTIME_URL, provider=provider)


def test_missing_litellm_configuration_fails_at_startup():
    with pytest.raises(playground_service.PlaygroundInputError, match="LITELLM_API_KEY"):
        create_app(runtime_url=FAKE_RUNTIME_URL, provider="litellm")


def test_extractor_is_built_once_per_process_not_per_request(monkeypatch):
    """B3/B4: the provider extractor must be constructed exactly once, at
    create_app() time -- never rebuilt on /ingest, /ask, /compare,
    /compare-answers, or /clear, which would otherwise construct a fresh
    provider HTTP client per request with nothing ever closing the old
    ones."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    import postgres_graph_rag.playground_service as ps
    original_build_extractor = ps.build_extractor
    calls = []

    def counting_build_extractor(*args, **kwargs):
        calls.append((args, kwargs))
        return original_build_extractor(*args, **kwargs)

    with patch.object(ps, "build_extractor", side_effect=counting_build_extractor) as mocked:
        client = TestClient(create_app(runtime_url=FAKE_RUNTIME_URL, provider="openai", openai_api_key="sk-test-not-real"))
        assert mocked.call_count == 1  # built once, at app-creation time

        with patch(
            "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
            new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
        ), patch(
            "postgres_graph_rag.tenant_engine.TenantGraphRAG.answer",
            new=AsyncMock(return_value=_fake_answer()),
        ):
            client.post("/ingest", data={"text": "Checkout depends_on Auth."})
            client.post("/ask", data={"question": "Who owns it?"})
            client.post("/compare-answers", data={"question": "Who owns it?"})

    assert mocked.call_count == 1  # still exactly once after several requests
    assert calls[0][1].get("openai_api_key") == "sk-test-not-real"


def test_shared_resources_closed_on_app_shutdown():
    """B4: the shared PostgresGraphRAG (and its connection pool) must be
    closed once, on FastAPI lifespan shutdown -- not left open, and not
    closed per-request (which would break every subsequent request)."""
    with patch(
        "postgres_graph_rag.core.PostgresGraphRAG.close", new=AsyncMock()
    ) as mocked_close:
        with TestClient(create_app(runtime_url=FAKE_RUNTIME_URL, provider="offline")) as client:
            mocked_close.assert_not_awaited()
            client.get("/")
            mocked_close.assert_not_awaited()
        # TestClient used as a context manager runs the lifespan's shutdown
        # phase on __exit__.
        mocked_close.assert_awaited_once()


def test_studio_never_calls_setup_secure_behaviorally():
    """Behavioral guard: patches setup_secure() to fail loudly if the
    Studio's /ingest route ever actually reaches it."""
    client = _client()
    with patch(
        "postgres_graph_rag.core.PostgresGraphRAG.setup_secure",
        new=AsyncMock(side_effect=AssertionError("setup_secure() must never be called by the studio")),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ):
        resp = client.post("/ingest", data={"text": "Checkout depends_on Auth.", "provider": "offline"})
    assert resp.status_code == 200
    assert "documents added: 1" in resp.text
