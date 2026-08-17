"""Tests for postgres_graph_rag/playground_service.py and demo.py's
`playground` subcommand -- all offline/mocked, no real Postgres connection
and no real provider call anywhere in this file.

Ingestion/answer/retrieval are exercised through the real `TenantGraphRAG`
class (not re-implemented), with only its I/O-performing methods
(`add_document_detailed`/`answer`/`retrieve`/`delete_document`) patched --
`SecureGraphStore`'s own constructor does no network I/O (its connection
pool is lazily opened on first real query), so constructing a real
`PostgresGraphRAG`/`TenantGraphRAG` here is safe with those methods patched.
"""
import argparse
import inspect
import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from postgres_graph_rag import demo, playground_service
from postgres_graph_rag.playground_service import (
    IngestedScope,
    PlaygroundInputError,
    answer_playground_question,
    build_extractor,
    clear_playground_scope,
    ingest_playground_document,
    new_playground_scope,
    run_playground_session,
)
from postgres_graph_rag.models import DEFAULT_RETRIEVAL_CONFIG
from postgres_graph_rag.tenant_engine import AnswerResult, Citation, TenantRetrievalResult


FAKE_RUNTIME_URL = "postgresql://pgr_runtime:pw@localhost:5432/graph_rag_test"

_READY_1536 = {"ready": True, "embedding_dimension": 1536, "vector_type": "vector",
               "schema_version": 7, "updated_at": "2026-01-01T00:00:00+00:00"}


@pytest.fixture(autouse=True)
def _mock_schema_status():
    """B1's preflight (check_schema_compatibility) calls a real
    SecureGraphStore.schema_status() before any ingest -- mocked here,
    applied to every test in this file by default as a schema already
    provisioned for the offline/OpenAI (1536) dimension. Tests exercising a
    mismatch or an unready schema override this within their own body."""
    with patch(
        "postgres_graph_rag.tenancy.SecureGraphStore.schema_status",
        new=AsyncMock(return_value=dict(_READY_1536)),
    ):
        yield


# --- pure validation (no mocking needed -- these raise before any I/O) -----

@pytest.mark.asyncio
async def test_ingest_rejects_empty_text():
    with pytest.raises(PlaygroundInputError, match="empty"):
        await ingest_playground_document(text="   ", provider="offline", runtime_url=FAKE_RUNTIME_URL)


@pytest.mark.asyncio
async def test_ingest_rejects_oversized_text():
    text = "x" * (playground_service.PLAYGROUND_MAX_CHARS + 1)
    with pytest.raises(PlaygroundInputError, match="characters"):
        await ingest_playground_document(text=text, provider="offline", runtime_url=FAKE_RUNTIME_URL)


@pytest.mark.asyncio
async def test_ingest_rejects_missing_runtime_url():
    with pytest.raises(PlaygroundInputError, match="runtime_url"):
        await ingest_playground_document(text="hello", provider="offline", runtime_url="")


@pytest.mark.asyncio
async def test_answer_rejects_empty_question():
    scope = IngestedScope(
        tenant_id=uuid.uuid4(), namespace="ns",
        provider="offline", source_id="doc-1",
        ingest_report={}, edge_count=0,
    )
    with pytest.raises(PlaygroundInputError, match="empty"):
        await answer_playground_question(scope=scope, question="")


@pytest.mark.asyncio
async def test_answer_rejects_invalid_mode():
    scope = IngestedScope(
        tenant_id=uuid.uuid4(), namespace="ns",
        provider="offline", source_id="doc-1",
        ingest_report={}, edge_count=0,
    )
    with pytest.raises(PlaygroundInputError, match="mode"):
        await answer_playground_question(scope=scope, question="q", mode="bogus")


def test_build_extractor_rejects_unknown_provider():
    with pytest.raises(PlaygroundInputError, match="provider"):
        build_extractor("anthropic")


def test_build_extractor_requires_openai_key():
    with pytest.raises(PlaygroundInputError, match="OPENAI_API_KEY"):
        build_extractor("openai", openai_api_key=None)


def test_build_extractor_requires_google_key():
    with pytest.raises(PlaygroundInputError, match="GOOGLE_API_KEY"):
        build_extractor("gemini", google_api_key=None)


def test_build_extractor_requires_complete_litellm_configuration():
    with pytest.raises(PlaygroundInputError, match="LITELLM_BASE_URL"):
        build_extractor("litellm", litellm_api_key="sk-test")


def test_build_extractor_configures_litellm_as_openai_compatible():
    extractor = build_extractor(
        "litellm",
        litellm_api_key="sk-test",
        litellm_base_url="http://localhost:4000/v1",
        litellm_chat_model="graph-chat",
        litellm_embedding_model="graph-embed",
        litellm_embedding_dimension=768,
    )
    assert extractor.config == {
        "extraction_model": "graph-chat",
        "embedding_model": "graph-embed",
        "dimension": 768,
        "api_family": "openai",
    }
    assert str(extractor.openai_client.base_url) == "http://localhost:4000/v1/"


def test_litellm_env_dimension_is_validated():
    with pytest.raises(PlaygroundInputError, match="positive integer"):
        playground_service.litellm_kwargs_from_env({"LITELLM_EMBEDDING_DIMENSION": "many"})


@pytest.mark.asyncio
async def test_litellm_schema_dimension_mismatch_fails_before_provider_call():
    with pytest.raises(PlaygroundInputError, match="needs 768-dimensional"):
        await playground_service.check_schema_compatibility(
            FAKE_RUNTIME_URL,
            "litellm",
            litellm_embedding_dimension=768,
        )


def test_build_extractor_offline_needs_no_key():
    extractor = build_extractor("offline")
    assert extractor.config["extraction_model"] == "offline-fixture"


def test_new_playground_scope_is_isolated_and_random():
    tenant_a, ns_a = new_playground_scope()
    tenant_b, ns_b = new_playground_scope()
    assert tenant_a != tenant_b
    assert ns_a != ns_b
    assert ns_a.startswith("playground-")
    assert str(tenant_a) != "11111111-1111-1111-1111-111111111111"  # DEMO_TENANT


# --- ingestion / answer orchestration (real TenantGraphRAG, mocked I/O) ---

@pytest.mark.asyncio
async def test_ingest_offline_with_explicit_triplets_reports_edges():
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 2, "status": "active"}),
    ) as mocked:
        scope = await ingest_playground_document(
            text="Checkout depends_on Auth.",
            provider="offline",
            runtime_url=FAKE_RUNTIME_URL,
            triplets=[{"subject": "Checkout", "predicate": "depends_on", "object": "Auth"}],
        )
    mocked.assert_awaited_once()
    assert scope.edge_count == 2
    assert scope.warnings == []
    assert scope.namespace.startswith("playground-")


@pytest.mark.asyncio
async def test_ingest_offline_plain_prose_warns_about_zero_edges():
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 0, "status": "active"}),
    ):
        scope = await ingest_playground_document(
            text="our new payments module talks to the ledger sometimes",
            provider="offline",
            runtime_url=FAKE_RUNTIME_URL,
        )
    assert scope.edge_count == 0
    assert scope.warnings, "expected a zero-edges warning"
    assert "0 relationships" in scope.warnings[0]


@pytest.mark.asyncio
async def test_offline_mode_never_calls_answer_shows_retrieval_only():
    """Regression test: OfflineExtractor.generate_text() is a truncation
    stub of its own prompt, not a real synthesizer -- verified directly by
    calling answer() with OfflineExtractor and observing it echo the whole
    prompt back as the "answer". answer_playground_question() must use
    retrieve(), never answer(), for provider="offline"."""
    scope = IngestedScope(
        tenant_id=uuid.uuid4(), namespace="playground-abcd",
        provider="offline", source_id="playground-doc", ingest_report={}, edge_count=1,
    )
    fake_retrieval = TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None)

    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.answer", new=AsyncMock()
    ) as mocked_answer, patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve",
        new=AsyncMock(return_value=fake_retrieval),
    ) as mocked_retrieve:
        result = await answer_playground_question(
            scope=scope, question="What does Checkout depend on?",
            runtime_url=FAKE_RUNTIME_URL, extractor=playground_service.OfflineExtractor(),
        )

    mocked_answer.assert_not_awaited()
    mocked_retrieve.assert_awaited_once()
    assert mocked_retrieve.await_args.kwargs["graph_seed_chunks"] == 3
    assert result.grounded is None
    assert result.grounding_status is None
    assert result.citations == []
    assert "does not synthesize" in result.answer


@pytest.mark.asyncio
async def test_live_provider_mode_uses_answer_call():
    scope = IngestedScope(
        tenant_id=uuid.uuid4(), namespace="playground-xyz",
        provider="openai", source_id="playground-doc",
        ingest_report={}, edge_count=1,
    )
    fake_answer = AnswerResult(
        answer="Identity Team [doc-1#0]",
        citations=[Citation(source_id="doc-1", chunk_id="c1", ordinal=0, excerpt="...")],
        retrieval=TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None),
        usage={"total_tokens": 42}, latency_ms=12.5,
        grounding_mode="citation_only", grounding_status="citation_valid_only",
    )
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.answer",
        new=AsyncMock(return_value=fake_answer),
    ) as mocked_answer:
        result = await answer_playground_question(
            scope=scope, question="Who owns it?",
            top_k=2,
            runtime_url=FAKE_RUNTIME_URL, extractor=playground_service.OfflineExtractor(),
        )

    mocked_answer.assert_awaited_once()
    assert mocked_answer.await_args.kwargs["graph_seed_chunks"] == 2
    assert result.answer == "Identity Team [doc-1#0]"
    assert result.grounded is True
    assert result.usage == {"total_tokens": 42}


@pytest.mark.asyncio
async def test_playground_seed_policy_is_broader_without_changing_core_default():
    """Playground/Studio use up to three seed chunks for multi-document
    questions, while library/MCP callers retain the core default of one."""
    assert DEFAULT_RETRIEVAL_CONFIG["graph_seed_chunks"] == 1

    scope = IngestedScope(
        tenant_id=uuid.uuid4(), namespace="playground-seeds",
        provider="offline", source_id="playground-doc", ingest_report={}, edge_count=1,
    )
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve",
        new=AsyncMock(return_value=TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None)),
    ) as mocked_retrieve:
        await playground_service.retrieve_playground_evidence(
            scope=scope,
            question="Which team maintains the login service?",
            top_k=5,
            runtime_url=FAKE_RUNTIME_URL,
            extractor=playground_service.OfflineExtractor(),
        )

    assert mocked_retrieve.await_args.kwargs["graph_seed_chunks"] == 3


@pytest.mark.asyncio
async def test_clear_playground_scope_deletes_only_its_own_document():
    scope = IngestedScope(
        tenant_id=uuid.uuid4(), namespace="playground-own",
        provider="offline", source_id="playground-doc", ingest_report={}, edge_count=0,
    )
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.delete_document", new=AsyncMock()
    ) as mocked_delete:
        await clear_playground_scope(
            scope, runtime_url=FAKE_RUNTIME_URL, extractor=playground_service.OfflineExtractor(),
        )
    mocked_delete.assert_awaited_once_with("playground-own", "playground-doc")


@pytest.mark.asyncio
async def test_run_playground_session_ingests_then_answers():
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve",
        new=AsyncMock(return_value=TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None)),
    ):
        result = await run_playground_session(
            text="Checkout depends_on Auth.", question="What does Checkout depend on?",
            provider="offline", mode="hybrid_graph", runtime_url=FAKE_RUNTIME_URL,
        )
    assert result.scope.edge_count == 1


@pytest.mark.asyncio
async def test_ingest_rejects_mixing_providers_within_one_session():
    """Offline and OpenAI are both 1536-dim but occupy unrelated embedding
    spaces -- a second document in the same session must not silently
    switch providers, even though no dimension error would occur."""
    with pytest.raises(PlaygroundInputError, match="already ingested a document with provider"):
        await ingest_playground_document(
            text="Auth is owned_by Identity Team.",
            provider="openai",
            runtime_url=FAKE_RUNTIME_URL,
            expected_provider="offline",
        )


@pytest.mark.asyncio
async def test_ingest_preflight_rejects_dimension_mismatch_before_any_call():
    """B1: the schema/provider dimension mismatch must be caught by the
    preflight check (check_schema_compatibility), before add_document
    ever runs -- not discovered reactively from a psycopg exception after
    extraction/embedding has already been attempted."""
    with patch(
        "postgres_graph_rag.tenancy.SecureGraphStore.schema_status",
        new=AsyncMock(return_value={"ready": True, "embedding_dimension": 1536, "vector_type": "vector"}),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed", new=AsyncMock()
    ) as mocked_add_document:
        with pytest.raises(PlaygroundInputError, match="needs 3072-dimensional embeddings"):
            await ingest_playground_document(
                text="Checkout depends_on Auth.", provider="gemini", runtime_url=FAKE_RUNTIME_URL,
                google_api_key="unused-for-this-test",
            )
    mocked_add_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_preflight_rejects_when_schema_not_ready():
    with patch(
        "postgres_graph_rag.tenancy.SecureGraphStore.schema_status",
        new=AsyncMock(return_value={"ready": False, "reason": "schema_settings_missing"}),
    ):
        with pytest.raises(PlaygroundInputError, match="not ready"):
            await ingest_playground_document(
                text="Checkout depends_on Auth.", provider="offline", runtime_url=FAKE_RUNTIME_URL,
            )


@pytest.mark.asyncio
async def test_ingest_converts_dimension_mismatch_to_clear_error_as_a_backstop():
    """Even if a mismatch somehow slipped past the preflight check (schema
    metadata could theoretically be stale), a raw psycopg.errors.DataException
    from add_document_detailed() itself must still never reach the caller
    directly -- this is defense in depth, not the primary mechanism (see the
    preflight test above)."""
    import psycopg

    with patch(
        # Preflight reports a *matching* dimension here (3072, correct for
        # gemini) so it passes -- isolating this test to the reactive
        # backstop inside add_document_detailed() itself.
        "postgres_graph_rag.tenancy.SecureGraphStore.schema_status",
        new=AsyncMock(return_value={"ready": True, "embedding_dimension": 3072, "vector_type": "halfvec"}),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(side_effect=psycopg.errors.DataException("expected 1536 dimensions, not 3072")),
    ):
        with pytest.raises(PlaygroundInputError, match="incompatible with the already-provisioned schema"):
            await ingest_playground_document(
                text="Checkout depends_on Auth.", provider="gemini", runtime_url=FAKE_RUNTIME_URL,
                google_api_key="unused-for-this-test",
            )


# --- CLI wrapper (demo.py's `playground` subcommand) -----------------------

def _base_playground_args(**overrides):
    defaults = dict(
        admin_url="postgresql://admin:pw@localhost:5432/graph_rag_test",
        runtime_url=FAKE_RUNTIME_URL,
        runtime_role="pgr_demo_runtime", runtime_password="pw",
        text="Checkout depends_on Auth.", text_file=None,
        triplets_json=None, triplets_file=None,
        question="What does Checkout depend on?",
        provider="offline", mode="hybrid_graph",
        tenant_id=None, namespace=None,
        top_k=5, hops=3, json=False, cleanup=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.asyncio
async def test_cli_playground_prints_json_output(capsys):
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve",
        new=AsyncMock(return_value=TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None)),
    ):
        await demo._playground(_base_playground_args(json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "offline"
    assert payload["namespace"].startswith("playground-")
    assert payload["namespace"] != "incident-demo"


@pytest.mark.asyncio
async def test_cli_playground_requires_text_or_text_file():
    with pytest.raises(SystemExit, match="text"):
        await demo._playground(_base_playground_args(text=None, text_file=None))


@pytest.mark.asyncio
async def test_cli_playground_reports_invalid_triplets_json():
    with pytest.raises(SystemExit, match="Invalid JSON"):
        await demo._playground(_base_playground_args(triplets_json="{not valid json"))


@pytest.mark.asyncio
async def test_cli_playground_reports_missing_provider_key():
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        await demo._playground(_base_playground_args(provider="openai"))


@pytest.mark.asyncio
async def test_cli_playground_rejects_both_text_and_text_file(tmp_path):
    text_file = tmp_path / "doc.txt"
    text_file.write_text("file contents")
    with pytest.raises(SystemExit, match="only one of --text or --text-file"):
        await demo._playground(_base_playground_args(text="inline text", text_file=str(text_file)))


@pytest.mark.asyncio
async def test_cli_playground_rejects_both_triplets_json_and_file(tmp_path):
    triplets_file = tmp_path / "triplets.json"
    triplets_file.write_text('{"triplets": []}')
    with pytest.raises(SystemExit, match="only one of --triplets-json or --triplets-file"):
        await demo._playground(_base_playground_args(
            triplets_json='{"triplets": []}', triplets_file=str(triplets_file),
        ))


@pytest.mark.asyncio
async def test_cli_playground_reports_malformed_triplet_schema_cleanly():
    """A structurally-valid-JSON-but-schema-invalid triplet (missing
    predicate/object) previously raised a raw pydantic.ValidationError
    traceback -- reproduced directly, now caught and converted."""
    with pytest.raises(SystemExit, match="must look like"):
        await demo._playground(_base_playground_args(triplets_json='{"triplets": [{"subject": "A"}]}'))


@pytest.mark.asyncio
async def test_cli_playground_runs_without_admin_url_or_postgres_url(monkeypatch):
    """Reproduced bug: the playground previously required POSTGRES_URL/
    --admin-url even though it only ever needs the restricted runtime role.
    `_runtime_only_url` must not demand an admin DSN."""
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve",
        new=AsyncMock(return_value=TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None)),
    ):
        await demo._playground(_base_playground_args(admin_url=None))  # no SystemExit expected


def test_cli_playground_runtime_url_resolver_requires_no_admin_dsn(monkeypatch):
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("PGR_RUNTIME_URL", raising=False)
    with pytest.raises(SystemExit, match="PGR_RUNTIME_URL"):
        demo._runtime_only_url(_base_playground_args(admin_url=None, runtime_url=None))


@pytest.mark.asyncio
async def test_cli_ingest_query_evaluate_run_without_admin_url_or_postgres_url(monkeypatch):
    """Reproduced bug: `ingest`/`query`/`evaluate` called `_urls()` and
    immediately discarded its admin_url (`_, runtime_url = _urls(args)`),
    yet `_urls()` unconditionally required POSTGRES_URL/--admin-url even
    when --runtime-url was already supplied directly and no admin access
    was ever going to be used. Fixed by switching these three subcommands
    to the same `_runtime_only_url()` helper `_playground` already used."""
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    base_args = argparse.Namespace(
        admin_url=None, runtime_url=FAKE_RUNTIME_URL,
        runtime_role="pgr_demo_runtime", runtime_password="pw",
        tenant_id=str(demo.DEMO_TENANT), namespace="incident-demo",
        mode="hybrid_graph", top_k=5, hops=3, output=None,
    )
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ):
        await demo._ingest(base_args)  # no SystemExit expected

    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve",
        new=AsyncMock(return_value=TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None)),
    ):
        await demo._query(argparse.Namespace(question="Which service?", **vars(base_args)))
        await demo._evaluate(base_args)


@pytest.mark.asyncio
async def test_cli_playground_cleanup_flag_deletes_its_own_document():
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve",
        new=AsyncMock(return_value=TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None)),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.delete_document", new=AsyncMock()
    ) as mocked_delete:
        await demo._playground(_base_playground_args(cleanup=True))
    mocked_delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_cli_playground_without_cleanup_flag_keeps_its_document():
    with patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve",
        new=AsyncMock(return_value=TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None)),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.delete_document", new=AsyncMock()
    ) as mocked_delete:
        await demo._playground(_base_playground_args(cleanup=False))
    mocked_delete.assert_not_awaited()


def test_playground_subcommand_defaults_to_isolated_scope_not_demo_scope(monkeypatch):
    """Argparse-level check: the `playground` subparser's own --tenant-id/
    --namespace default to None (auto-generated), overriding the top-level
    parser's PGR_DEMO_TENANT_ID/PGR_DEMO_NAMESPACE-derived defaults used by
    every other subcommand."""
    import sys

    captured = {}

    async def fake_playground(args):
        captured["args"] = args

    monkeypatch.setattr(demo, "_playground", fake_playground)
    monkeypatch.setattr(sys, "argv", ["postgres-graph-rag-demo", "playground", "--text", "hi", "question?"])
    demo.main()

    args = captured["args"]
    assert args.tenant_id is None
    assert args.namespace is None


def test_playground_never_touches_setup_or_reset():
    """Source-level invariant: the playground path must never call
    setup_secure() or any schema-drop/reset statement. Checks for actual
    call-site patterns (a leading `.`), not just the word appearing in a
    docstring explaining the precondition. See
    test_playground_never_calls_setup_secure_behaviorally below for a
    stronger, runtime-behavior version of the same guarantee."""
    forbidden = (".setup_secure(", "DROP SCHEMA", "DROP TABLE")
    for source in (
        inspect.getsource(demo._playground),
        inspect.getsource(playground_service),
    ):
        for term in forbidden:
            assert term not in source, f"found forbidden {term!r}"


@pytest.mark.asyncio
async def test_playground_never_calls_setup_secure_behaviorally():
    """Behavioral guard, stronger than the source-inspection check above:
    patches setup_secure() to fail loudly if anything in a real playground
    run actually reaches it, rather than only checking today's source
    text."""
    with patch(
        "postgres_graph_rag.core.PostgresGraphRAG.setup_secure",
        new=AsyncMock(side_effect=AssertionError("setup_secure() must never be called by the playground")),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.add_document_detailed",
        new=AsyncMock(return_value={"skipped": False, "chunks": 1, "triplets": 1}),
    ), patch(
        "postgres_graph_rag.tenant_engine.TenantGraphRAG.retrieve",
        new=AsyncMock(return_value=TenantRetrievalResult(chunks=[], nodes=[], edges=[], trace=None)),
    ):
        await run_playground_session(
            text="Checkout depends_on Auth.", question="What does Checkout depend on?",
            provider="offline", mode="hybrid_graph", runtime_url=FAKE_RUNTIME_URL,
        )


# --- B2: provider-aware `setup` (dimension only, no paid call) --------------

def _setup_args(**overrides):
    defaults = dict(
        admin_url=FAKE_RUNTIME_URL, runtime_url=None,
        runtime_role="pgr_demo_runtime", runtime_password="pw",
        tenant_id=None, namespace=None,
        reset=False, migrate_legacy=False, provider="offline",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,expected_dim", [("offline", 1536), ("openai", 1536), ("gemini", 3072)])
async def test_setup_selects_dimension_for_provider_with_no_paid_call(provider, expected_dim, capsys):
    """setup --provider must provision the schema at the right dimension
    for each provider, and must never construct a real API client (no key
    is ever passed) -- confirmed by not supplying one and not patching any
    network call, only setup_secure() itself."""
    with patch(
        "postgres_graph_rag.core.PostgresGraphRAG.setup_secure", new=AsyncMock()
    ) as mocked_setup_secure:
        await demo._setup(_setup_args(provider=provider))

    mocked_setup_secure.assert_awaited_once()
    out = capsys.readouterr().out
    assert f"provider={provider!r}" in out
    assert f"dimension={expected_dim}" in out


@pytest.mark.asyncio
async def test_setup_selects_configured_litellm_dimension(monkeypatch, capsys):
    monkeypatch.setenv("LITELLM_EMBEDDING_DIMENSION", "768")
    with patch(
        "postgres_graph_rag.core.PostgresGraphRAG.setup_secure", new=AsyncMock()
    ) as mocked_setup_secure:
        await demo._setup(_setup_args(provider="litellm"))

    mocked_setup_secure.assert_awaited_once()
    out = capsys.readouterr().out
    assert "provider='litellm'" in out
    assert "dimension=768" in out
