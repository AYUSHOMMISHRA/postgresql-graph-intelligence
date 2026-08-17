"""Local browser Demo Studio: a thin FastAPI presentation layer over
`playground_service.py` -- no ingestion/retrieval/answer/citation logic
lives here, only form handling, session bookkeeping, and HTML rendering.

A session (one browser tab, tracked by an unguessable cookie) can accumulate
more than one ingested document under one isolated tenant/namespace before
asking a question -- this is what lets a user paste two related documents
(mirroring the bundled demo's checkout/auth/Identity-Team scenario) and ask
a genuinely multi-hop question, rather than being limited to one document
per question.

No DSN, runtime/admin URL, or provider API key is ever placed in a template
context, a cookie, or a log line here -- only tenant_id/namespace (not
secret) and whatever `playground_service` returns.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import playground_service
from .core import PostgresGraphRAG
from .playground_service import IngestedScope, PlaygroundInputError

logger = logging.getLogger(__name__)

# This is a local, single-operator demo tool with no authentication and no
# CORS support -- it must never be exposed beyond localhost (see
# studio_cli.py's own loopback-bind enforcement). TrustedHostMiddleware is a
# second, independent layer against DNS-rebinding: even if a browser is
# tricked into resolving some other domain to 127.0.0.1, a request whose
# Host header isn't one of these is rejected before it reaches any route.
# "testserver" is FastAPI/Starlette's own TestClient default Host header --
# harmless to allow permanently, since an attacker exploiting DNS rebinding
# controls what domain their page navigates to (their own), never literally
# this fixed test-only string.
DEFAULT_ALLOWED_HOSTS = ["127.0.0.1", "localhost", "[::1]", "testserver"]

SESSION_COOKIE = "pgr_studio_session"
_TEMPLATES_DIR = Path(__file__).parent / "templates"

# One session per *browser* (the cookie is shared across every tab of the
# same browser) -- not one session per tab. Documented in the page copy
# rather than solved with client-side sessionStorage plumbing, which would
# reintroduce the small-JS-footprint tradeoff this design deliberately
# avoids for a local, single-operator demo tool.
_SESSION_TTL_SECONDS = 2 * 60 * 60
_MAX_SESSIONS = 200

_PROVIDER_LABELS = {
    "offline": "Offline — no API calls, $0",
    "openai": "OpenAI — uses OPENAI_API_KEY, may incur cost",
    "gemini": "Gemini — uses GOOGLE_API_KEY, may incur cost",
    "litellm": "LiteLLM Gateway — uses LITELLM_API_KEY, may incur cost",
}

# A UI convenience only -- the uploaded content is still treated as
# untrusted text regardless of extension, exactly like pasted text.
_ALLOWED_UPLOAD_EXTENSIONS = (".txt", ".md")

# Starlette's multipart form parser buffers the whole request body *before*
# a route function ever runs (so a bounded `file.read(n)` inside the route
# alone doesn't stop it from buffering an oversized upload first) -- this
# generous ceiling (well above the plain-text character cap, to allow for
# multipart boundary/header/encoding overhead) lets a declared oversized
# request get rejected by Content-Length before that buffering happens.
_MAX_INGEST_REQUEST_BYTES = playground_service.PLAYGROUND_MAX_CHARS * 4


class _IngestBodySizeLimitMiddleware:
    """Plain ASGI middleware (not `@app.middleware("http")`/BaseHTTPMiddleware
    -- see create_app()'s comment on why) that rejects a declared oversized
    `POST /ingest` body by Content-Length, before Starlette's multipart
    parser ever buffers it."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/ingest":
            headers = dict(scope.get("headers") or [])
            content_length = headers.get(b"content-length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > _MAX_INGEST_REQUEST_BYTES:
                    response = PlainTextResponse(
                        f"Request body exceeds the {_MAX_INGEST_REQUEST_BYTES}-byte limit for /ingest.",
                        status_code=413,
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


def _new_session_state() -> Dict[str, Any]:
    return {
        "tenant_id": None,
        "namespace": None,
        "provider": None,
        "source_ids": [],
        "warnings": [],
        "last_result": None,
        "last_compare": None,
        "error": None,
        "last_used": time.monotonic(),
        # Guards ingest/ask/compare/compare-answers/clear for this one
        # session so concurrent requests (e.g. a double-clicked paid
        # comparison, or a removal racing an ask) can't interleave and
        # corrupt this session's state or produce duplicate source ids.
        "lock": asyncio.Lock(),
    }


def create_app(
    runtime_url: Optional[str] = None,
    provider: str = "offline",
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    litellm_api_key: Optional[str] = None,
    litellm_base_url: Optional[str] = None,
    litellm_chat_model: Optional[str] = None,
    litellm_embedding_model: Optional[str] = None,
    litellm_embedding_dimension: Optional[int] = None,
    allowed_hosts: Optional[list[str]] = None,
) -> FastAPI:
    """`runtime_url` defaults to `PGR_RUNTIME_URL` from the environment if
    not given -- read once at app-creation time, never re-read per request
    and never rendered anywhere.

    `provider` is fixed for the lifetime of this process (one provider per
    Studio process, never a per-request/per-session choice): every document
    ingested by any session in this process uses the same provider and the
    same shared extractor/client, built once here. Missing credentials for
    a live provider raise `PlaygroundInputError` immediately, out of
    `build_extractor()` -- i.e. at app-creation time, before the server
    starts accepting requests, not on the first `/ingest` call.
    """
    runtime_url = runtime_url or os.getenv("PGR_RUNTIME_URL")
    extractor = playground_service.build_extractor(
        provider,
        openai_api_key=openai_api_key,
        google_api_key=google_api_key,
        litellm_api_key=litellm_api_key,
        litellm_base_url=litellm_base_url,
        litellm_chat_model=litellm_chat_model,
        litellm_embedding_model=litellm_embedding_model,
        litellm_embedding_dimension=litellm_embedding_dimension,
    )
    # One shared PostgresGraphRAG (extractor + connection pool) for the whole
    # process -- built once here, reused by every request, closed once at
    # shutdown. `SecureGraphStore.__init__`/pool construction do no eager
    # network I/O, so this is safe to build even before the pool is used.
    shared_rag = PostgresGraphRAG(runtime_url=runtime_url, extractor=extractor) if runtime_url else None

    # In-memory session store: fine for a local, single-operator demo tool.
    # Keyed only by an unguessable server-generated id, never a client value.
    sessions: Dict[str, Dict[str, Any]] = {}

    async def _cleanup_session_documents(state: Dict[str, Any]) -> None:
        """Best-effort: removes this session's own ingested documents from
        PostgreSQL before its state is dropped (TTL/LRU eviction or
        graceful shutdown). Never raises -- a cleanup failure must not
        block eviction or shutdown. Only non-secret identifiers
        (tenant_id/namespace -- random UUIDs/strings, never a DSN or API
        key) are logged. Per `delete_document`'s own documented behavior,
        this does not sweep every orphaned graph entity/edge."""
        if shared_rag is None or not state["source_ids"] or not state["tenant_id"]:
            return
        try:
            await playground_service.delete_playground_documents(
                rag=shared_rag, tenant_id=state["tenant_id"],
                namespace=state["namespace"], source_ids=list(state["source_ids"]),
            )
        except Exception:  # noqa: BLE001 -- best-effort cleanup must never raise
            logger.warning(
                "Failed to clean up documents for an evicted/closing session "
                "(tenant_id=%s, namespace=%s); orphaned graph entities may remain.",
                state["tenant_id"], state["namespace"],
            )

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            for state in list(sessions.values()):
                await _cleanup_session_documents(state)
            if shared_rag is not None:
                await shared_rag.close()

    app = FastAPI(title="Postgres GraphRAG Demo Studio", lifespan=_lifespan)
    # No CORS middleware is registered, deliberately -- this is a local,
    # single-operator demo tool, not a hosted multi-user service.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or DEFAULT_ALLOWED_HOSTS)
    # A plain ASGI middleware, not `@app.middleware("http")` (BaseHTTPMiddleware)
    # -- BaseHTTPMiddleware re-raises exceptions past a registered
    # `@app.exception_handler(Exception)` even after it has already produced
    # a response (a known Starlette ordering bug), which would defeat B12's
    # generic-error handling below.
    app.add_middleware(_IngestBodySizeLimitMiddleware)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    async def _sweep_sessions() -> None:
        now = time.monotonic()
        expired = [s for s, state in sessions.items() if now - state["last_used"] > _SESSION_TTL_SECONDS]
        for sid in expired:
            state = sessions.pop(sid, None)
            if state is not None:
                await _cleanup_session_documents(state)
        if len(sessions) > _MAX_SESSIONS:
            # Evict the least-recently-used sessions first, not an arbitrary
            # or insertion-ordered set -- bounds memory on a long-running
            # process without dropping whoever is actively using it.
            oldest = sorted(sessions.items(), key=lambda kv: kv[1]["last_used"])
            for sid, _ in oldest[: len(sessions) - _MAX_SESSIONS]:
                state = sessions.pop(sid, None)
                if state is not None:
                    await _cleanup_session_documents(state)

    async def _session(request: Request) -> tuple[str, Dict[str, Any]]:
        await _sweep_sessions()
        sid = request.cookies.get(SESSION_COOKIE)
        if not sid or sid not in sessions:
            sid = secrets.token_urlsafe(32)
            sessions[sid] = _new_session_state()
        sessions[sid]["last_used"] = time.monotonic()
        return sid, sessions[sid]

    def _render(request: Request, sid: str, session: Dict[str, Any]) -> HTMLResponse:
        response = templates.TemplateResponse(request, "index.html", {
            "session": session,
            "provider": provider,
            "provider_label": _PROVIDER_LABELS[provider],
            "has_document": bool(session["source_ids"]),
            "runtime_configured": bool(runtime_url),
        })
        response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        sid, session = await _session(request)
        return _render(request, sid, session)

    @app.post("/ingest", response_class=HTMLResponse)
    async def ingest(
        request: Request,
        text: str = Form(""),
        triplets_json: str = Form(""),
        file: Optional[UploadFile] = None,
    ):
        sid, session = await _session(request)
        async with session["lock"]:
            session["error"] = None

            has_text = bool(text.strip())
            has_file = file is not None and bool(file.filename)
            if has_text and has_file:
                session["error"] = "Pass either pasted text or an uploaded file, not both."
                return _render(request, sid, session)

            resolved_text = text
            if has_file:
                assert file is not None
                filename = (file.filename or "").lower()
                if not filename.endswith(_ALLOWED_UPLOAD_EXTENSIONS):
                    session["error"] = f"Only {'/'.join(_ALLOWED_UPLOAD_EXTENSIONS)} files are accepted."
                    return _render(request, sid, session)
                # Bounded read: never load more than the playground's own
                # character cap plus one byte into memory, so an oversized
                # upload is rejected cleanly instead of buffering the whole
                # file first and validating afterward.
                raw = await file.read(playground_service.PLAYGROUND_MAX_CHARS + 1)
                if len(raw) > playground_service.PLAYGROUND_MAX_CHARS:
                    session["error"] = (
                        f"Uploaded file exceeds the {playground_service.PLAYGROUND_MAX_CHARS}-character "
                        "playground limit; trim it and try again."
                    )
                    return _render(request, sid, session)
                resolved_text = raw.decode("utf-8", errors="replace")

            if shared_rag is None:
                session["error"] = "PGR_RUNTIME_URL is not configured on the server."
                return _render(request, sid, session)

            triplets = None
            if triplets_json.strip():
                import json
                try:
                    data = json.loads(triplets_json)
                    from .extractor import Triplet
                    triplets = [Triplet(**item) for item in data["triplets"]]
                except Exception as exc:  # noqa: BLE001 -- surfaced as a form error, not a 500
                    session["error"] = f"Invalid triplets JSON: {exc}"
                    return _render(request, sid, session)

            try:
                # shared_rag is not None (checked above) only when runtime_url
                # was truthy at app-creation time (see create_app()).
                assert runtime_url is not None
                scope = await playground_service.ingest_playground_document(
                    text=resolved_text,
                    provider=provider,
                    runtime_url=runtime_url,
                    triplets=triplets,
                    tenant_id=session["tenant_id"],
                    namespace=session["namespace"],
                    source_id=f"studio-doc-{len(session['source_ids']) + 1}",
                    rag=shared_rag,
                    litellm_embedding_dimension=litellm_embedding_dimension,
                )
            except PlaygroundInputError as exc:
                session["error"] = str(exc)
                return _render(request, sid, session)

            session["tenant_id"] = scope.tenant_id
            session["namespace"] = scope.namespace
            session["provider"] = scope.provider
            session["source_ids"].append(scope.source_id)
            session["warnings"].extend(scope.warnings)
            return _render(request, sid, session)

    def _current_scope(session: Dict[str, Any]) -> IngestedScope:
        return IngestedScope(
            tenant_id=session["tenant_id"],
            namespace=session["namespace"],
            provider=session["provider"],
            source_id=session["source_ids"][-1],
            ingest_report={},
            edge_count=0,
        )

    @app.post("/ask", response_class=HTMLResponse)
    async def ask(request: Request, question: str = Form(...), mode: str = Form("hybrid_graph")):
        sid, session = await _session(request)
        async with session["lock"]:
            session["error"] = None
            session["last_compare"] = None

            if not session["source_ids"]:
                session["error"] = "Add at least one document before asking a question."
                return _render(request, sid, session)

            try:
                # source_ids non-empty implies a prior successful /ingest, which
                # itself required shared_rag to be non-None.
                assert shared_rag is not None
                result = await playground_service.answer_playground_question(
                    scope=_current_scope(session), question=question, mode=mode, rag=shared_rag,
                )
            except PlaygroundInputError as exc:
                session["error"] = str(exc)
                return _render(request, sid, session)

            session["last_result"] = result
            return _render(request, sid, session)

    @app.post("/compare", response_class=HTMLResponse)
    async def compare(request: Request, question: str = Form(...)):
        """Free comparison: retrieval only, no answer-generation call, for
        all three modes -- see retrieve_playground_evidence(). Generating an
        actual answer for all three modes is a separate, explicit action
        (`/compare-answers`) precisely because it costs three real
        generation calls with a live provider, not one."""
        sid, session = await _session(request)
        async with session["lock"]:
            session["error"] = None
            session["last_result"] = None

            if not session["source_ids"]:
                session["error"] = "Add at least one document before comparing modes."
                return _render(request, sid, session)

            try:
                assert shared_rag is not None
                scope = _current_scope(session)
                results = {}
                for mode in ("vector", "hybrid", "hybrid_graph"):
                    results[mode] = await playground_service.retrieve_playground_evidence(
                        scope=scope, question=question, mode=mode, rag=shared_rag,
                    )
            except PlaygroundInputError as exc:
                session["error"] = str(exc)
                return _render(request, sid, session)

            session["last_compare"] = results
            return _render(request, sid, session)

    @app.post("/compare-answers", response_class=HTMLResponse)
    async def compare_answers(request: Request, question: str = Form(...)):
        """Explicit, separate opt-in: generates a real answer for all three
        modes (three live-provider generation calls for OpenAI/Gemini/LiteLLM), only
        ever run when the user clicks this specifically -- never the
        default comparison action."""
        sid, session = await _session(request)
        async with session["lock"]:
            session["error"] = None
            session["last_result"] = None

            if not session["source_ids"]:
                session["error"] = "Add at least one document before comparing modes."
                return _render(request, sid, session)

            try:
                assert shared_rag is not None
                scope = _current_scope(session)
                results = {}
                for mode in ("vector", "hybrid", "hybrid_graph"):
                    results[mode] = await playground_service.answer_playground_question(
                        scope=scope, question=question, mode=mode, rag=shared_rag,
                    )
            except PlaygroundInputError as exc:
                session["error"] = str(exc)
                return _render(request, sid, session)

            session["last_compare"] = results
            return _render(request, sid, session)

    @app.post("/clear", response_class=HTMLResponse)
    async def clear(request: Request):
        sid, session = await _session(request)
        async with session["lock"]:
            if shared_rag is not None and session["source_ids"] and session["tenant_id"]:
                try:
                    await playground_service.delete_playground_documents(
                        rag=shared_rag,
                        tenant_id=session["tenant_id"],
                        namespace=session["namespace"],
                        source_ids=session["source_ids"],
                    )
                except Exception:  # noqa: BLE001 -- reported honestly below, session state left intact
                    # logger.warning, not .exception -- a raw DB error's own
                    # message can echo connection details; only fixed,
                    # non-secret identifiers (tenant_id/namespace) go to the
                    # log, matching _cleanup_session_documents's policy.
                    logger.warning(
                        "Failed to remove documents for a /clear request (tenant_id=%s, namespace=%s)",
                        session["tenant_id"], session["namespace"],
                    )
                    session["error"] = (
                        "Failed to remove this session's documents from the database. Session "
                        "state was left unchanged -- try again, or restart the Studio."
                    )
                    return _render(request, sid, session)
            sessions[sid] = _new_session_state()
        return _render(request, sid, sessions[sid])

    @app.exception_handler(Exception)
    async def _unexpected_error_handler(request: Request, exc: Exception) -> PlainTextResponse:
        # Never reached by PlaygroundInputError (every route catches that
        # itself and turns it into a clean, specific message) -- only a
        # genuinely unexpected failure (e.g. a raw database error) lands
        # here. No traceback, DSN, key, SQL, or provider response body is
        # ever put in the response; a request id makes the matching server
        # log line findable without exposing its contents to the client.
        request_id = secrets.token_hex(8)
        logger.exception(
            "Unexpected error handling %s %s (request_id=%s)", request.method, request.url.path, request_id,
        )
        return PlainTextResponse(
            f"Something went wrong (request id: {request_id}). This has been logged; please try again.",
            status_code=500,
        )

    return app
