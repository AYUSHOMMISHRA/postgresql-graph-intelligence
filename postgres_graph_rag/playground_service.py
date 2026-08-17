"""Shared logic for the interactive playground: a CLI subcommand
(`postgres-graph-rag-demo playground`) and, later, a browser demo studio,
both letting a user try their own text and question against the real,
RLS-secured engine instead of only the bundled fixed demo scenario.

Neither interface re-implements ingestion/retrieval/answer/citation logic --
they only call the functions in this module, which call `PostgresGraphRAG`/
`TenantGraphRAG` exactly as the bundled demo does.
"""
from __future__ import annotations

import secrets
import uuid
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import psycopg

from .core import PostgresGraphRAG
from .extractor import LLMExtractor, Triplet
from .models import (
    GOOGLE_DEFAULT_CONFIG,
    OPENAI_DEFAULT_CONFIG,
    ProviderConfig,
    build_litellm_config,
)
from .offline import OFFLINE_CONFIG, OfflineExtractor
from .tenancy import SecureGraphStore
from .tenant_engine import Citation

# Well under add_document()'s own 2,000,000-character engine limit -- this
# is a demo-tool-appropriate cap, not the engine's real ceiling, so that a
# pasted-text playground can't be used to push an unreasonably large
# document through embedding/extraction by accident.
PLAYGROUND_MAX_CHARS = 20_000
# The core engine intentionally seeds traversal from one chunk by default.
# Playground/Studio questions commonly span several pasted documents, so this
# surface opts into a bounded broader seed set without changing library/MCP
# retrieval semantics.
PLAYGROUND_GRAPH_SEED_CHUNKS = 3

_PROVIDERS = ("offline", "openai", "gemini", "litellm")
_MODES = ("vector", "hybrid", "hybrid_graph")

# Every provider's real embedding dimension -- matching dimensions does not
# mean compatible embeddings (offline's hash-based vectors and OpenAI's real
# embedding model are both 1536-dim but occupy unrelated vector spaces), but
# a *mismatched* dimension is always definitely incompatible and is cheap to
# check before ever calling a provider.
_PROVIDER_DIMENSIONS = {
    "offline": OFFLINE_CONFIG["dimension"],
    "openai": OPENAI_DEFAULT_CONFIG["dimension"],
    "gemini": GOOGLE_DEFAULT_CONFIG["dimension"],
}


class PlaygroundInputError(ValueError):
    """A validation failure meant to be shown to the user as a clean
    message -- never a raw traceback -- by either interface."""


def litellm_kwargs_from_env(
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Read LiteLLM gateway settings without logging or storing secrets.

    Returned keys match ``build_extractor``/playground function keyword
    arguments so every console surface uses one configuration contract.
    """
    source = os.environ if environ is None else environ
    dimension_text = source.get("LITELLM_EMBEDDING_DIMENSION")
    dimension: Optional[int] = None
    if dimension_text:
        try:
            dimension = int(dimension_text)
        except ValueError as exc:
            raise PlaygroundInputError(
                "LITELLM_EMBEDDING_DIMENSION must be a positive integer"
            ) from exc
        if dimension <= 0:
            raise PlaygroundInputError(
                "LITELLM_EMBEDDING_DIMENSION must be a positive integer"
            )
    return {
        "litellm_api_key": source.get("LITELLM_API_KEY") or None,
        "litellm_base_url": source.get("LITELLM_BASE_URL") or None,
        "litellm_chat_model": source.get("LITELLM_CHAT_MODEL") or None,
        "litellm_embedding_model": source.get("LITELLM_EMBEDDING_MODEL") or None,
        "litellm_embedding_dimension": dimension,
    }


def new_playground_scope(prefix: str = "playground") -> tuple[uuid.UUID, str]:
    """A fresh, random tenant id and namespace, isolated by construction
    from the bundled demo's fixed `DEMO_TENANT`/`"incident-demo"` scope and
    from any other playground session. Never derived from user input."""
    return uuid.uuid4(), f"{prefix}-{secrets.token_hex(4)}"


def build_extractor(
    provider: str,
    *,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    litellm_api_key: Optional[str] = None,
    litellm_base_url: Optional[str] = None,
    litellm_chat_model: Optional[str] = None,
    litellm_embedding_model: Optional[str] = None,
    litellm_embedding_dimension: Optional[int] = None,
) -> Any:
    """Builds a plain extractor for the chosen provider -- no document text
    involved. `offline` never makes a network call and, on its own, relies
    on `OfflineExtractor`'s fixed-predicate regex fallback (see its
    docstring), which produces zero edges for ordinary prose -- surfaced as
    a warning by `ingest_playground_document`, never hidden. When explicit
    triplets are supplied for offline mode, `ingest_playground_document`
    builds its own `OfflineExtractor(triplets_by_text={text: triplets})`
    directly instead of calling this function, since that mapping needs the
    exact document text, which this function deliberately doesn't take --
    it only ever picks a provider/model.
    """
    if provider not in _PROVIDERS:
        raise PlaygroundInputError(f"provider must be one of {_PROVIDERS}, got {provider!r}")

    if provider == "offline":
        return OfflineExtractor()

    if provider == "openai":
        if not openai_api_key:
            raise PlaygroundInputError(
                "provider=openai requires OPENAI_API_KEY to be set (env var or --openai-api-key)"
            )
        openai_config: ProviderConfig = {**OPENAI_DEFAULT_CONFIG}
        return LLMExtractor(config=openai_config, openai_api_key=openai_api_key)

    if provider == "gemini":
        if not google_api_key:
            raise PlaygroundInputError(
                "provider=gemini requires GOOGLE_API_KEY to be set (env var or --google-api-key)"
            )
        google_config: ProviderConfig = {**GOOGLE_DEFAULT_CONFIG}
        return LLMExtractor(config=google_config, google_api_key=google_api_key)

    missing = [
        name
        for name, value in (
            ("LITELLM_API_KEY", litellm_api_key),
            ("LITELLM_BASE_URL", litellm_base_url),
            ("LITELLM_CHAT_MODEL", litellm_chat_model),
            ("LITELLM_EMBEDDING_MODEL", litellm_embedding_model),
            ("LITELLM_EMBEDDING_DIMENSION", litellm_embedding_dimension),
        )
        if value is None or value == ""
    ]
    if missing:
        raise PlaygroundInputError(
            f"provider=litellm requires {', '.join(missing)} to be set"
        )
    assert litellm_api_key is not None
    assert litellm_base_url is not None
    assert litellm_chat_model is not None
    assert litellm_embedding_model is not None
    assert litellm_embedding_dimension is not None
    if not litellm_base_url.startswith(("http://", "https://")):
        raise PlaygroundInputError("LITELLM_BASE_URL must start with http:// or https://")
    try:
        litellm_config = build_litellm_config(
            extraction_model=litellm_chat_model,
            embedding_model=litellm_embedding_model,
            dimension=litellm_embedding_dimension,
        )
    except ValueError as exc:
        raise PlaygroundInputError(str(exc)) from exc
    return LLMExtractor(
        config=litellm_config,
        openai_api_key=litellm_api_key,
        openai_base_url=litellm_base_url,
    )


@dataclass
class IngestedScope:
    """One playground document, ingested under one isolated tenant/namespace.

    Identifier-only: no DSN, no API key, no client/pool object. Neither
    interface may store a runtime URL, an extractor, or a `PostgresGraphRAG`
    on this -- callers that need to act on a scope again (answer/retrieve/
    delete) supply their own resources explicitly (either a caller-owned
    `rag`, or `runtime_url` + `extractor` for a one-shot call) rather than
    reading them back off the scope. This is what makes it safe to put this
    object in session state, a log line, or a repr.
    """

    tenant_id: uuid.UUID
    namespace: str
    provider: str
    source_id: str
    ingest_report: Dict[str, Any]
    edge_count: int
    warnings: List[str] = field(default_factory=list)


def _validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise PlaygroundInputError("document text must not be empty")
    if len(text) > PLAYGROUND_MAX_CHARS:
        raise PlaygroundInputError(
            f"document text must be at most {PLAYGROUND_MAX_CHARS} characters "
            f"(got {len(text)}); trim it and try again"
        )
    return text


def _validate_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise PlaygroundInputError("question must not be empty")
    return question


async def check_schema_compatibility(
    runtime_url: str,
    provider: str,
    *,
    litellm_embedding_dimension: Optional[int] = None,
) -> None:
    """Preflight: confirms the already-provisioned schema's embedding
    dimension matches `provider`'s, before any extraction/embedding call is
    attempted -- so an incompatible request fails before it can incur any
    provider cost, rather than after `add_document_detailed()` has already
    started (a raw `psycopg.errors.DataException` from mid-ingestion, which
    is error *translation*, not prevention).

    Uses a throwaway `SecureGraphStore` purely to read
    `schema_settings` metadata -- no tenant context, no vector operations,
    so the store's own `vector_type` default is irrelevant here.
    """
    if provider not in _PROVIDERS:
        raise PlaygroundInputError(f"provider must be one of {_PROVIDERS}, got {provider!r}")

    store = SecureGraphStore(runtime_url)
    try:
        status = await store.schema_status()
    finally:
        await store.close()

    if not status.get("ready"):
        raise PlaygroundInputError(
            f"Secure schema is not ready ({status.get('reason', 'unknown')}). Run "
            "`postgres-graph-rag-demo setup` (with --provider matching what you intend to use "
            "here) before ingesting."
        )

    if provider == "litellm":
        if litellm_embedding_dimension is None:
            raise PlaygroundInputError(
                "provider=litellm requires LITELLM_EMBEDDING_DIMENSION to be set"
            )
        expected_dim = litellm_embedding_dimension
    else:
        expected_dim = _PROVIDER_DIMENSIONS[provider]
    actual_dim = status["embedding_dimension"]
    if actual_dim != expected_dim:
        raise PlaygroundInputError(
            f"provider={provider!r} needs {expected_dim}-dimensional embeddings, but this schema "
            f"is provisioned for {actual_dim} dimensions. Provision a separate schema/database for "
            f"this provider (`postgres-graph-rag-demo setup --provider {provider}` against a fresh "
            "database), or use a provider matching the existing schema."
        )


async def ingest_playground_document(
    *,
    text: str,
    provider: str,
    runtime_url: str,
    triplets: Optional[List[Triplet]] = None,
    tenant_id: Optional[uuid.UUID] = None,
    namespace: Optional[str] = None,
    source_id: str = "playground-doc",
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    litellm_api_key: Optional[str] = None,
    litellm_base_url: Optional[str] = None,
    litellm_chat_model: Optional[str] = None,
    litellm_embedding_model: Optional[str] = None,
    litellm_embedding_dimension: Optional[int] = None,
    expected_provider: Optional[str] = None,
    rag: Optional[PostgresGraphRAG] = None,
) -> IngestedScope:
    """Validates input, resolves an isolated tenant/namespace if not given,
    builds the chosen provider's extractor, and ingests exactly one
    document. Never calls `setup_secure()` or any schema-reset path --
    `setup_secure()` must already have been run once (by an operator, via
    `postgres-graph-rag-demo setup`) against the admin DSN.

    `expected_provider`: pass the provider a *prior* document in this same
    session/scope already used (e.g. the Studio's session state), to reject
    mixing providers within one tenant/namespace. This matters even when
    two providers share an embedding dimension: offline's hash-based
    embeddings and OpenAI's real embedding model both produce 1536-dim
    vectors, but occupy unrelated vector spaces -- mixing them produces
    similarity scores that look valid (no dimension error) but are
    meaningless. A caller that never reuses a scope across calls (the CLI's
    single-shot `run_playground_session`) can omit this.

    `rag`: an already-constructed, caller-owned `PostgresGraphRAG` (e.g. the
    Studio's one shared instance, built once at process startup with its
    provider's real extractor/client). When given, it is reused as-is and
    is never closed here -- the caller owns its lifecycle. When omitted (the
    CLI's one-shot call), a fresh `PostgresGraphRAG` is built from
    `runtime_url`/the resolved provider extractor and closed before this
    function returns, exactly as before. Explicit offline triplets always
    need a text-bound extractor (see below), so that case builds and closes
    its own one-off `PostgresGraphRAG` even when a shared `rag` was passed
    in -- offline mode has no real client/connection to leak by doing so.
    """
    text = _validate_text(text)
    if not runtime_url:
        raise PlaygroundInputError("runtime_url is required")
    if expected_provider is not None and provider != expected_provider:
        raise PlaygroundInputError(
            f"this session already ingested a document with provider={expected_provider!r}; "
            f"mixing providers within one session isn't safe (even same-dimension providers, "
            f"e.g. offline and openai, use unrelated embedding spaces, which would make "
            f"retrieval scores meaningless). Clear the session before switching providers."
        )

    # Preflight before any extractor is built (so a mismatched provider
    # never even constructs a real API-backed client) and before any
    # extraction/embedding call could be attempted.
    await check_schema_compatibility(
        runtime_url,
        provider,
        litellm_embedding_dimension=litellm_embedding_dimension,
    )

    if tenant_id is None or namespace is None:
        generated_tenant_id, generated_namespace = new_playground_scope()
    resolved_tenant_id = tenant_id if tenant_id is not None else generated_tenant_id
    resolved_namespace = namespace if namespace is not None else generated_namespace

    extractor_triplets = list(triplets) if triplets else None
    owns_rag = True
    if provider == "offline" and extractor_triplets:
        # OfflineExtractor only maps *exact* text, so register this call's
        # text verbatim against the supplied triplets -- build_extractor()
        # deliberately doesn't take document text, so this mapping is built
        # here rather than inside it. Always a fresh, one-off instance, even
        # when a shared `rag` was supplied, since it must be bound to this
        # exact text.
        active_rag = PostgresGraphRAG(
            runtime_url=runtime_url, extractor=OfflineExtractor(triplets_by_text={text: extractor_triplets})
        )
    elif rag is not None:
        active_rag, owns_rag = rag, False
    else:
        extractor = build_extractor(
            provider,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            litellm_api_key=litellm_api_key,
            litellm_base_url=litellm_base_url,
            litellm_chat_model=litellm_chat_model,
            litellm_embedding_model=litellm_embedding_model,
            litellm_embedding_dimension=litellm_embedding_dimension,
        )
        active_rag = PostgresGraphRAG(runtime_url=runtime_url, extractor=extractor)

    try:
        engine = active_rag.for_tenant(resolved_tenant_id)
        report = await engine.add_document_detailed(text, resolved_namespace, source_id)
    except psycopg.errors.DataException as exc:
        # Typically an embedding-dimension mismatch: the schema's embedding
        # column is fixed at `setup_secure()` time (see README's "pgvector
        # dimension limits"). Converted to a clean, playground-appropriate
        # message instead of a raw DB exception surfacing to the CLI/UI.
        raise PlaygroundInputError(
            f"provider={provider!r} is incompatible with the already-provisioned schema: {exc}. "
            "The schema's embedding dimension/type is fixed at setup time; use a provider "
            "matching whatever `postgres-graph-rag-demo setup` was run with, or provision a "
            "separate schema/database for this provider."
        ) from exc
    finally:
        if owns_rag:
            await active_rag.close()

    edge_count = int(report.get("triplets", 0))
    warnings: List[str] = []
    if provider == "offline" and edge_count == 0:
        warnings.append(
            "Offline mode extracted 0 relationships from this text. It only recognizes "
            "'Capitalized Phrase <predicate> Capitalized Phrase' with a fixed predicate "
            "list (depends_on/owned_by/caused/deployed/uses/runs_on/has_runbook) -- note "
            "this REQUIRES capitalized entity names, so realistic lowercase service names "
            "like 'checkout-service depends_on auth-service' will NOT match (try "
            "'Checkout-Service depends_on Auth-Service' instead, or pass explicit "
            "--triplets-json). Vector/hybrid retrieval still works on the raw text either "
            "way; hybrid_graph will have no extra graph evidence to add without a match. "
            "Use --provider openai/gemini/litellm to extract relationships from ordinary prose "
            "regardless of casing."
        )

    return IngestedScope(
        tenant_id=resolved_tenant_id,
        namespace=resolved_namespace,
        provider=provider,
        source_id=source_id,
        ingest_report=report,
        edge_count=edge_count,
        warnings=warnings,
    )


@dataclass
class PlaygroundResult:
    scope: IngestedScope
    mode: str
    question: str
    answer: Optional[str]
    citations: List[Citation]
    grounded: Optional[bool]
    grounding_status: Optional[str]
    retrieval_context: str
    graph_path: Optional[str]
    usage: Dict[str, int]
    latency_ms: float
    warnings: List[str]


def _format_graph_path(edges) -> Optional[str]:
    if not edges:
        return None
    return "; ".join(f"{e.source_content} --[{e.relation}]--> {e.target_content}" for e in edges)


async def _retrieve_only_result(*, engine, scope: IngestedScope, question: str, mode: str,
                                 top_k: int, hops: int, note: str) -> PlaygroundResult:
    """Shared by offline mode (which never calls `answer()`, see below) and
    `retrieve_playground_evidence()` (the free, no-generation-call
    comparison view): builds a `PlaygroundResult` from `retrieve()` alone,
    with `note` explaining why there's no generated answer."""
    retrieval = await engine.retrieve(
        question,
        scope.namespace,
        mode=mode,
        top_k=top_k,
        hops=hops,
        graph_seed_chunks=min(PLAYGROUND_GRAPH_SEED_CHUNKS, top_k),
    )
    return PlaygroundResult(
        scope=scope,
        mode=mode,
        question=question,
        answer=note,
        citations=[],
        grounded=None,
        grounding_status=None,
        retrieval_context=retrieval.to_context_string(),
        graph_path=_format_graph_path(retrieval.edges) if mode == "hybrid_graph" else None,
        usage={},
        latency_ms=(retrieval.trace.embedding_ms + retrieval.trace.search_ms + retrieval.trace.traversal_ms)
        if retrieval.trace else 0.0,
        warnings=list(scope.warnings),
    )


async def retrieve_playground_evidence(
    *,
    scope: IngestedScope,
    question: str,
    mode: str = "hybrid_graph",
    top_k: int = 5,
    hops: int = 3,
    runtime_url: Optional[str] = None,
    extractor: Optional[Any] = None,
    rag: Optional[PostgresGraphRAG] = None,
) -> PlaygroundResult:
    """Free retrieval-only view of a question against an already-ingested
    scope -- never calls a provider's answer-generation, regardless of
    provider. Intended for comparing all three retrieval modes side by side
    without paying for three generation calls; `answer_playground_question`
    is what actually generates a single, real answer for one selected mode.

    Resources: pass either a caller-owned `rag` (reused, never closed here)
    or both `runtime_url` and `extractor` (a fresh `PostgresGraphRAG` is
    built and closed for this one call) -- see `ingest_playground_document`.
    """
    question = _validate_question(question)
    if mode not in _MODES:
        raise PlaygroundInputError(f"mode must be one of {_MODES}, got {mode!r}")
    owns_rag = rag is None
    if rag is None:
        if runtime_url is None or extractor is None:
            raise PlaygroundInputError("either `rag`, or both `runtime_url` and `extractor`, must be provided")
        rag = PostgresGraphRAG(runtime_url=runtime_url, extractor=extractor)
    try:
        engine = rag.for_tenant(scope.tenant_id)
        return await _retrieve_only_result(
            engine=engine, scope=scope, question=question, mode=mode, top_k=top_k, hops=hops,
            note=(
                "(Retrieval-only comparison -- no answer generated for this mode. Use "
                "\"Ask\" for a single generated answer, or explicitly opt into generating "
                "all three.)"
            ),
        )
    finally:
        if owns_rag:
            await rag.close()


async def answer_playground_question(
    *,
    scope: IngestedScope,
    question: str,
    mode: str = "hybrid_graph",
    top_k: int = 5,
    hops: int = 3,
    max_answer_tokens: int = 1500,
    grounding_mode: str = "citation_only",
    runtime_url: Optional[str] = None,
    extractor: Optional[Any] = None,
    rag: Optional[PostgresGraphRAG] = None,
) -> PlaygroundResult:
    """Answers a question against an already-ingested playground scope.

    Offline mode deliberately never calls `answer()`: `OfflineExtractor`'s
    `generate_text()` is a fixture stub (a truncated echo of its prompt),
    not a real synthesizer -- verified directly (calling `answer()` with
    `OfflineExtractor` returns the *entire prompt text*, including its own
    instructions, as the "answer"). The bundled demo's own `_query` command
    avoids this the same way, by only ever calling `retrieve()`. Offline
    mode here shows retrieved evidence and the graph path instead, which is
    honest and still useful; real providers get a composed `answer()`.

    Resources: pass either a caller-owned `rag` (reused, never closed here)
    or both `runtime_url` and `extractor` (a fresh `PostgresGraphRAG` is
    built and closed for this one call) -- see `ingest_playground_document`.
    """
    question = _validate_question(question)
    if mode not in _MODES:
        raise PlaygroundInputError(f"mode must be one of {_MODES}, got {mode!r}")

    owns_rag = rag is None
    if rag is None:
        if runtime_url is None or extractor is None:
            raise PlaygroundInputError("either `rag`, or both `runtime_url` and `extractor`, must be provided")
        rag = PostgresGraphRAG(runtime_url=runtime_url, extractor=extractor)
    try:
        engine = rag.for_tenant(scope.tenant_id)

        if scope.provider == "offline":
            return await _retrieve_only_result(
                engine=engine, scope=scope, question=question, mode=mode, top_k=top_k, hops=hops,
                note=(
                    "(Offline mode shows retrieved evidence only -- it does not "
                    "synthesize a natural-language answer. Use --provider openai, "
                    "gemini, or litellm for a composed, cited answer.)"
                ),
            )

        result = await engine.answer(
            question, scope.namespace, max_answer_tokens=max_answer_tokens,
            grounding_mode=grounding_mode,
            mode=mode,
            top_k=top_k,
            hops=hops,
            graph_seed_chunks=min(PLAYGROUND_GRAPH_SEED_CHUNKS, top_k),
        )
        return PlaygroundResult(
            scope=scope,
            mode=mode,
            question=question,
            answer=result.answer,
            citations=result.citations,
            grounded=result.grounded,
            grounding_status=result.grounding_status,
            retrieval_context=result.retrieval.to_context_string(),
            graph_path=_format_graph_path(result.retrieval.edges) if mode == "hybrid_graph" else None,
            usage=result.usage,
            latency_ms=result.latency_ms,
            warnings=list(scope.warnings),
        )
    finally:
        if owns_rag:
            await rag.close()


async def run_playground_session(
    *,
    text: str,
    question: str,
    provider: str,
    mode: str,
    runtime_url: str,
    triplets: Optional[List[Triplet]] = None,
    source_id: str = "playground-doc",
    tenant_id: Optional[uuid.UUID] = None,
    namespace: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    litellm_api_key: Optional[str] = None,
    litellm_base_url: Optional[str] = None,
    litellm_chat_model: Optional[str] = None,
    litellm_embedding_model: Optional[str] = None,
    litellm_embedding_dimension: Optional[int] = None,
    top_k: int = 5,
    hops: int = 3,
    max_answer_tokens: int = 1500,
    grounding_mode: str = "citation_only",
) -> PlaygroundResult:
    """One-shot ingest + answer, used by the CLI's single invocation. Builds
    its own extractor for the answer step: `retrieve()`/`answer()` only ever
    call `get_embedding()`/`generate_text()`, never `extract_triplets()`, so
    a plain provider extractor (not bound to the ingested text's explicit
    triplets mapping, if any) is correct and sufficient here."""
    scope = await ingest_playground_document(
        text=text, provider=provider, runtime_url=runtime_url, triplets=triplets,
        tenant_id=tenant_id, namespace=namespace, source_id=source_id,
        openai_api_key=openai_api_key, google_api_key=google_api_key,
        litellm_api_key=litellm_api_key, litellm_base_url=litellm_base_url,
        litellm_chat_model=litellm_chat_model,
        litellm_embedding_model=litellm_embedding_model,
        litellm_embedding_dimension=litellm_embedding_dimension,
    )
    extractor = build_extractor(
        provider,
        openai_api_key=openai_api_key,
        google_api_key=google_api_key,
        litellm_api_key=litellm_api_key,
        litellm_base_url=litellm_base_url,
        litellm_chat_model=litellm_chat_model,
        litellm_embedding_model=litellm_embedding_model,
        litellm_embedding_dimension=litellm_embedding_dimension,
    )
    return await answer_playground_question(
        scope=scope, question=question, mode=mode, top_k=top_k, hops=hops,
        max_answer_tokens=max_answer_tokens, grounding_mode=grounding_mode,
        runtime_url=runtime_url, extractor=extractor,
    )


async def clear_playground_scope(
    scope: IngestedScope,
    *,
    runtime_url: Optional[str] = None,
    extractor: Optional[Any] = None,
    rag: Optional[PostgresGraphRAG] = None,
) -> None:
    """Deletes only this session's own document -- never a schema reset,
    never another tenant/namespace's data. Reuses the existing, already
    RLS-scoped `delete_document`; no new core-engine method needed since
    the caller already knows exactly which source_id(s) it ingested.

    Resources: pass either a caller-owned `rag` (reused, never closed here)
    or both `runtime_url` and `extractor` (a fresh `PostgresGraphRAG` is
    built and closed for this one call) -- see `ingest_playground_document`.
    """
    await delete_playground_documents(
        tenant_id=scope.tenant_id, namespace=scope.namespace,
        source_ids=[scope.source_id],
        runtime_url=runtime_url, extractor=extractor, rag=rag,
    )


async def delete_playground_documents(
    *,
    tenant_id: uuid.UUID,
    namespace: str,
    source_ids: List[str],
    runtime_url: Optional[str] = None,
    extractor: Optional[Any] = None,
    rag: Optional[PostgresGraphRAG] = None,
) -> None:
    """Like `clear_playground_scope`, but for a scope that has accumulated
    more than one ingested document (e.g. the browser studio, where a user
    can paste several documents into one session to set up a multi-hop
    question). Deletes exactly the given source_ids under this
    tenant/namespace -- never a schema reset, never another session's data.

    Resources: pass either a caller-owned `rag` (reused, never closed here)
    or both `runtime_url` and `extractor` (a fresh `PostgresGraphRAG` is
    built and closed for this one call) -- see `ingest_playground_document`.
    """
    if not source_ids:
        return
    owns_rag = rag is None
    if rag is None:
        if runtime_url is None or extractor is None:
            raise PlaygroundInputError("either `rag`, or both `runtime_url` and `extractor`, must be provided")
        rag = PostgresGraphRAG(runtime_url=runtime_url, extractor=extractor)
    try:
        engine = rag.for_tenant(tenant_id)
        for source_id in source_ids:
            await engine.delete_document(namespace, source_id)
    finally:
        if owns_rag:
            await rag.close()
