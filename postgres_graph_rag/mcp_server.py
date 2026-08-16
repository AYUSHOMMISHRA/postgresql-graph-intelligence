"""MCP server (v0.5) exposing PostgresGraphRAG's tenant-aware engine as
Model Context Protocol tools.

Optional dependency: importing this module requires `mcp` (`pip install
"postgres-graph-rag[mcp]"`); nothing else in the package imports it, so
installing without the `mcp` extra keeps working exactly as before.

Transports: stdio (default, for local IDE/agent use) and Streamable HTTP
(for remote deployment). Legacy SSE is intentionally not implemented — the
plan calls for Streamable HTTP as the modern replacement.

Tenant resolution differs by transport:
  - stdio receives a *statically configured* tenant at startup
    (`stdio_tenant_id`) — there is exactly one tenant per stdio process,
    matching how a local IDE integration is actually used (one user, one
    session).
  - Streamable HTTP requires a `tenant_resolver(ctx) -> uuid.UUID` callable
    supplied by the deployer, which is expected to look at whatever
    authentication middleware/headers the deployment puts in place (a JWT
    claim, a session lookup, etc.) and return the tenant it resolves to.
    This module does NOT implement an OAuth authorization server itself —
    that is a substantial separate undertaking (token issuance, client
    registration, consent flows) out of scope here. What it does enforce:
    HTTP mode refuses to start without either a `tenant_resolver` or an
    explicit `allow_unauthenticated_dev=True` *and* a loopback bind
    (127.0.0.1), so an accidental unauthenticated public listener is not
    the default failure mode.

Mutation/admin tools (`ingest_documents`, `delete_document`,
`merge_entities`, `refresh_communities`, `summarize_communities`) are
registered only when `enable_mutations=True` — read-only tools
(`retrieve`, `get_entity`, `find_paths`, `list_communities`,
`get_community_summary`, `capabilities`, `health`) are always available.

`split_entity` is intentionally not exposed: `SecureGraphStore` doesn't
implement it (see the note on `merge_entities` in tenancy.py — reversing a
merge needs mention-level provenance this schema doesn't keep), so there is
no real operation to wire up.
"""

import ipaddress
import logging
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from mcp.server.mcpserver import Context, MCPServer

from .communities import CommunityEngine
from .core import PostgresGraphRAG

logger = logging.getLogger("postgres_graph_rag")

TenantResolver = Callable[[Context], uuid.UUID]


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost",)


class _LifespanState:
    def __init__(self, rag: PostgresGraphRAG, communities: CommunityEngine, tenant_resolver: Optional[TenantResolver]):
        self.rag = rag
        self.communities = communities
        self.tenant_resolver = tenant_resolver


def _resolve_tenant(ctx: Context, stdio_tenant_id: Optional[uuid.UUID]) -> uuid.UUID:
    state: _LifespanState = ctx.request_context.lifespan_context
    if stdio_tenant_id is not None:
        return stdio_tenant_id
    if state.tenant_resolver is None:
        raise PermissionError(
            "No tenant could be resolved for this request: no static stdio "
            "tenant configured and no tenant_resolver supplied."
        )
    return state.tenant_resolver(ctx)


def build_server(
    rag: PostgresGraphRAG,
    *,
    name: str = "postgres-graph-rag",
    stdio_tenant_id: Optional[uuid.UUID] = None,
    tenant_resolver: Optional[TenantResolver] = None,
    enable_mutations: bool = False,
    allow_unauthenticated_dev: bool = False,
) -> MCPServer:
    """Builds (but doesn't run) an MCPServer wired to `rag`.

    Exactly one of `stdio_tenant_id` or `tenant_resolver` should be set
    depending on which transport you intend to run: `run_stdio_async()` or
    `run_streamable_http_async()`. Neither is validated as "matching" the
    transport you actually call — that check happens in the `run_*`
    wrappers below, where the transport is known.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_server: MCPServer):
        # `_get_or_create_store()` is the same construction path
        # `for_tenant()` uses -- the store itself isn't tenant-specific
        # (tenant scoping happens per-call via tenant_connection()), so no
        # tenant_id is needed up front here, but the lazy-init and
        # missing-runtime_url validation must be identical either way.
        store = rag._get_or_create_store()
        communities = CommunityEngine(store)
        yield _LifespanState(rag, communities, tenant_resolver)

    server = MCPServer(name=name, lifespan=lifespan)

    # ------------------------------------------------------------------
    # Read-only tools (always registered)
    # ------------------------------------------------------------------

    @server.tool()
    async def retrieve(question: str, namespace: str, top_k: int = 5, ctx: Context = None) -> Dict[str, Any]:
        """Retrieves evidence (hybrid chunk search + graph neighborhood) for a question."""
        tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
        engine = rag.for_tenant(tenant_id)
        result = await engine.retrieve(question, namespace, top_k=top_k)
        return {
            "chunks": [asdict(c) for c in result.chunks],
            "nodes": [asdict(n) for n in result.nodes],
            "edges": [asdict(e) for e in result.edges],
            "trace": asdict(result.trace) if result.trace else None,
        }

    @server.tool()
    async def answer(question: str, namespace: str, top_k: int = 5, ctx: Context = None) -> Dict[str, Any]:
        """Returns a grounded answer with citations validated against retrieved chunks."""
        tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
        result = await rag.for_tenant(tenant_id).answer(question, namespace, top_k=top_k)
        return {
            "answer": result.answer,
            "grounded": result.grounded,
            "citations": [asdict(c) for c in result.citations],
            "usage": result.usage,
            "latency_ms": result.latency_ms,
        }

    @server.tool()
    async def get_entity(name: str, namespace: str, ctx: Context = None) -> Dict[str, Any]:
        """Looks up an entity by name (nearest match) and its supporting documents."""
        tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
        state: _LifespanState = ctx.request_context.lifespan_context
        embedding = await rag.extractor.get_embedding(name)
        matches = await state.rag._secure_store.vector_search_nodes(tenant_id, namespace, embedding, top_k=1)
        if not matches:
            return {"found": False}
        node = matches[0]
        docs = await state.rag._secure_store.get_mentioning_documents(tenant_id, namespace, str(node["id"]))
        return {"found": True, "id": str(node["id"]), "content": node["content"], "documents": docs}

    @server.tool()
    async def find_paths(
        seed_entities: List[str], namespace: str, max_hops: int = 2,
        target_entities: Optional[List[str]] = None, top_k: int = 5,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """Returns ranked paths when targets are supplied; otherwise returns
        the legacy graph neighborhood around the seed entities."""
        tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
        state: _LifespanState = ctx.request_context.lifespan_context
        store = state.rag._secure_store
        engine = rag.for_tenant(tenant_id)
        if target_entities:
            return {
                "paths": await engine.find_paths(
                    seed_entities, target_entities, namespace,
                    max_hops=max_hops, top_k=top_k,
                )
            }
        seed_ids = []
        for name in seed_entities:
            embedding = await rag.extractor.get_embedding(name)
            matches = await store.vector_search_nodes(tenant_id, namespace, embedding, top_k=1)
            if matches:
                seed_ids.append(str(matches[0]["id"]))
        if not seed_ids:
            return {"nodes": [], "edges": []}
        graph = await store.traverse_graph(tenant_id, seed_ids, namespace=namespace, max_hops=max_hops)
        return {
            "nodes": [{"id": str(n["id"]), "content": n["content"], "hop_distance": n["hop_distance"]} for n in graph["nodes"]],
            "edges": [
                {"source": e["source_content"], "relation": e["relation"], "target": e["target_content"]}
                for e in graph["edges"]
            ],
        }

    @server.tool()
    async def explain_connection(
        source_entity: str, target_entity: str, namespace: str, max_hops: int = 3, ctx: Context = None
    ) -> Dict[str, Any]:
        """Reconstructs the actual reasoning path between two named
        entities (e.g. "how is Alice connected to Payments"), as an ordered
        sequence of hops — not just a bag of scored neighbors. Returns
        {"found": False} if either entity can't be resolved or no path
        exists within max_hops."""
        tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
        engine = rag.for_tenant(tenant_id)
        explained = await engine.explain_connection(source_entity, target_entity, namespace, max_hops=max_hops)
        if explained is None:
            return {"found": False}
        return {
            "found": True,
            "path": [{"node": s.node, "relation": s.relation} for s in explained.steps],
            "score": explained.score,
            "hop_distance": explained.hop_distance,
            "summary": str(explained),
        }

    @server.tool()
    async def list_communities(namespace: str, ctx: Context = None) -> List[Dict[str, Any]]:
        """Lists the most recently computed communities for a namespace."""
        tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
        state: _LifespanState = ctx.request_context.lifespan_context
        return await state.communities.list_communities(tenant_id, namespace)

    @server.tool()
    async def get_community_summary(namespace: str, community_id: str, ctx: Context = None) -> Dict[str, Any]:
        """Gets the stored summary for a specific community, if one exists."""
        tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
        state: _LifespanState = ctx.request_context.lifespan_context
        results = await state.communities.query_global(tenant_id, namespace, "", top_k=100)
        for r in results:
            if r["community_id"] == community_id:
                return r
        return {"found": False}

    @server.tool()
    async def capabilities(ctx: Context = None) -> Dict[str, Any]:
        """Reports which tools are enabled on this server instance."""
        return {
            "read_only_tools": ["retrieve", "answer", "get_entity", "find_paths", "explain_connection", "list_communities", "get_community_summary"],
            "mutation_tools_enabled": enable_mutations,
            "mutation_tools": (
                ["ingest_documents", "delete_document", "merge_entities", "refresh_communities", "summarize_communities"]
                if enable_mutations else []
            ),
        }

    @server.tool()
    async def health(ctx: Context = None) -> Dict[str, Any]:
        """Database liveness and secure-schema compatibility state."""
        try:
            if rag._secure_store is None:
                raise RuntimeError("secure store is not initialized")
            schema = await rag._secure_store.schema_status()
            return {"status": "ok" if schema.get("ready") else "degraded", "schema": schema}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "detail": str(exc)}

    # ------------------------------------------------------------------
    # Mutation/admin tools (opt-in)
    # ------------------------------------------------------------------

    if enable_mutations:
        @server.tool()
        async def ingest_documents(
            documents: List[Dict[str, str]], namespace: str, ctx: Context = None
        ) -> Dict[str, Any]:
            """Ingests documents: each item is {"source_id": ..., "text": ...}."""
            tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
            engine = rag.for_tenant(tenant_id)
            reports = []
            for doc in documents:
                report = await engine.add_document(doc["text"], namespace, doc["source_id"])
                reports.append({"source_id": doc["source_id"], **report})
            return {"results": reports}

        @server.tool()
        async def delete_document(namespace: str, source_id: str, ctx: Context = None) -> Dict[str, Any]:
            """Deletes a document and its chunks/mentions (entities remain if other documents still support them)."""
            tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
            engine = rag.for_tenant(tenant_id)
            await engine.delete_document(namespace, source_id)
            return {"deleted": True}

        @server.tool()
        async def merge_entities(
            namespace: str, source_content: str, target_content: str, ctx: Context = None
        ) -> Dict[str, Any]:
            """Manually merges one entity into another (corrects a missed automatic resolution)."""
            tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
            state: _LifespanState = ctx.request_context.lifespan_context
            return await state.rag._secure_store.merge_entities(tenant_id, namespace, source_content, target_content)

        @server.tool()
        async def refresh_communities(namespace: str, force: bool = False, ctx: Context = None) -> Dict[str, Any]:
            """Recomputes communities for a namespace (skipped if not dirty, unless force=True)."""
            tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
            state: _LifespanState = ctx.request_context.lifespan_context
            return await state.communities.refresh_communities(tenant_id, namespace, force=force)

        @server.tool()
        async def summarize_communities(namespace: str, ctx: Context = None) -> List[Dict[str, Any]]:
            """Generates/refreshes summaries for the namespace's current communities."""
            tenant_id = _resolve_tenant(ctx, stdio_tenant_id)
            state: _LifespanState = ctx.request_context.lifespan_context
            return await state.communities.summarize_communities(tenant_id, namespace, rag.extractor)

    return server


async def run_stdio(rag: PostgresGraphRAG, tenant_id: uuid.UUID, enable_mutations: bool = False) -> None:
    """Runs the server over stdio for exactly one statically-configured tenant."""
    server = build_server(rag, stdio_tenant_id=tenant_id, enable_mutations=enable_mutations)
    await server.run_stdio_async()


async def run_http(
    rag: PostgresGraphRAG,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    tenant_resolver: Optional[TenantResolver] = None,
    enable_mutations: bool = False,
    allow_unauthenticated_dev: bool = False,
) -> None:
    """Runs the server over Streamable HTTP.

    Refuses to start without a `tenant_resolver` unless the caller
    explicitly opts into `allow_unauthenticated_dev=True` *and* binds to a
    loopback address — an accidentally-public, unauthenticated multi-tenant
    endpoint is not something this should make easy to reach by omission.
    """
    if tenant_resolver is None and not allow_unauthenticated_dev:
        raise ValueError(
            "run_http() requires tenant_resolver (map an authenticated request to "
            "a tenant_id) unless allow_unauthenticated_dev=True is explicitly set."
        )
    if tenant_resolver is None and not _is_loopback(host):
        raise ValueError(
            f"Refusing to bind {host}:{port} without tenant_resolver: "
            "allow_unauthenticated_dev is only permitted on a loopback address."
        )

    server = build_server(
        rag, tenant_resolver=tenant_resolver, enable_mutations=enable_mutations,
        allow_unauthenticated_dev=allow_unauthenticated_dev,
    )
    await server.run_streamable_http_async(host=host, port=port)


def main() -> None:
    """Console entry point (`postgres-graph-rag-mcp`). Reads connection
    strings and mode from environment variables/CLI flags rather than
    requiring a Python launcher script for the common case."""
    import argparse
    import asyncio
    import os

    parser = argparse.ArgumentParser(prog="postgres-graph-rag-mcp")
    parser.add_argument("--runtime-url", default=os.getenv("PGR_RUNTIME_URL"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--google-api-key", default=os.getenv("GOOGLE_API_KEY"))
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--tenant-id", default=os.getenv("PGR_STDIO_TENANT_ID"), help="Required for --transport stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--enable-mutations", action="store_true")
    parser.add_argument("--allow-unauthenticated-dev", action="store_true")
    args = parser.parse_args()

    if not args.runtime_url:
        parser.error("--runtime-url (or PGR_RUNTIME_URL) is required")

    rag = PostgresGraphRAG(
        runtime_url=args.runtime_url,
        openai_api_key=args.openai_api_key,
        google_api_key=args.google_api_key,
    )

    async def _run():
        try:
            if args.transport == "stdio":
                if not args.tenant_id:
                    parser.error("--tenant-id is required for --transport stdio")
                await run_stdio(rag, uuid.UUID(args.tenant_id), enable_mutations=args.enable_mutations)
            else:
                await run_http(
                    rag, host=args.host, port=args.port, enable_mutations=args.enable_mutations,
                    allow_unauthenticated_dev=args.allow_unauthenticated_dev,
                )
        finally:
            await rag.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
