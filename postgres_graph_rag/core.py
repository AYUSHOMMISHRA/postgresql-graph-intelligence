import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Callable
from .database import DatabaseManager
from .extractor import LLMExtractor, Triplet
from .models import (
    ProviderConfig,
    OPENAI_DEFAULT_CONFIG,
    GOOGLE_DEFAULT_CONFIG,
    RetrievalConfig,
    DEFAULT_RETRIEVAL_CONFIG,
    IngestionConfig,
    DEFAULT_INGESTION_CONFIG,
)

logger = logging.getLogger("postgres_graph_rag")

# Hard ceiling on chunks accepted by a single add_texts() call. This isn't a
# tunable default: it bounds the blast radius of a mistaken call (e.g.
# accidentally passing a full corpus as one `texts` argument) regardless of
# ingestion_config, since that would otherwise fan out into thousands of
# concurrent/retried LLM calls and a single enormous transaction.
MAX_CHUNKS_PER_INGEST_CALL = 5000


def simple_chunker(
    text: str, size: int = 1000, overlap: int = 100
) -> List[str]:
    """Default simple character-based chunking."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


@dataclass
class RetrievedNode:
    id: str
    content: str
    metadata: Dict[str, Any]
    hop_distance: int
    score: float


@dataclass
class RetrievedEdge:
    source_id: str
    target_id: str
    source_content: str
    target_content: str
    relation: str
    weight: float
    metadata: Dict[str, Any]


@dataclass
class RetrievalResult:
    """Structured retrieval output: explainable evidence, not just a string.

    Nodes and edges are pre-sorted by relevance score (vector similarity of
    the seed(s) they were reached from, decayed per hop and scaled by edge
    weight), so consumers that want a custom context format, a UI graph
    view, or citations don't have to re-derive relevance themselves.
    """

    nodes: List[RetrievedNode] = field(default_factory=list)
    edges: List[RetrievedEdge] = field(default_factory=list)
    seed_ids: List[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        if not self.nodes and not self.edges:
            return "No relevant context found."

        parts = ["Relevant Entities and Relationships:"]

        if self.nodes:
            parts.append("\nEntities:")
            for n in self.nodes:
                parts.append(f"- {n.content} (score: {n.score:.2f}, hops: {n.hop_distance})")

        if self.edges:
            parts.append("\nRelationships:")
            for e in self.edges:
                parts.append(
                    f"- {e.source_content} --[{e.relation}]--> {e.target_content}"
                )

        return "\n".join(parts)


class PostgresGraphRAG:
    def __init__(
        self,
        postgres_url: str,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        config: Optional[ProviderConfig] = None,
        chunker: Optional[Callable[[str], List[str]]] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        ingestion_config: Optional[IngestionConfig] = None,
        runtime_url: Optional[str] = None,
        extractor: Optional[Any] = None,
    ):
        """
        Initializes the PostgresGraphRAG instance.
        This is a standard synchronous initialization.

        `runtime_url` is only needed for the multi-tenant, RLS-secured path
        (`for_tenant()` / `setup_secure()`): it must be the connection
        string for the restricted runtime role created by `setup_secure()`,
        never a superuser/owner URL — RLS does nothing for a role that can
        bypass it. `postgres_url` remains the single-tenant path unchanged.
        """
        self.db = DatabaseManager(postgres_url)
        self._runtime_url = runtime_url
        self._secure_store = None  # lazily created by for_tenant()

        if config is None:
            config = (
                GOOGLE_DEFAULT_CONFIG
                if google_api_key
                else OPENAI_DEFAULT_CONFIG
            )

        # Dependency injection keeps the production provider path unchanged
        # while allowing deterministic offline demos/tests without network or
        # paid model calls.
        self.extractor = extractor or LLMExtractor(
            config=config,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
        )
        self.chunker = chunker or simple_chunker
        self.retrieval_config: RetrievalConfig = {
            **DEFAULT_RETRIEVAL_CONFIG,
            **(retrieval_config or {}),
        }
        self.ingestion_config: IngestionConfig = {
            **DEFAULT_INGESTION_CONFIG,
            **(ingestion_config or {}),
        }

    async def __aenter__(self):
        """Supports async context manager usage."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes resources when exiting the context."""
        await self.close()

    async def close(self):
        """Manually closes the database connection pool(s)."""
        await self.db.close()
        if self._secure_store is not None:
            await self._secure_store.close()

    async def setup_secure(
        self,
        admin_url: str,
        runtime_role: str,
        runtime_password: str,
        migrate_legacy_data: bool = False,
    ):
        """Creates/upgrades the tenant-aware, RLS-secured schema (v0.3).

        Must be called with `admin_url` — a privileged (e.g. superuser)
        connection string, separate from `runtime_url` passed to the
        constructor — since this creates the restricted runtime role and
        installs RLS policies, which the runtime role itself must not be
        able to do. Safe to re-run.
        """
        from .tenancy import migrate_schema

        await migrate_schema(
            admin_url=admin_url,
            runtime_role=runtime_role,
            runtime_password=runtime_password,
            embedding_dimension=self.extractor.config["dimension"],
            migrate_legacy_data=migrate_legacy_data,
        )

    def for_tenant(self, tenant_id, event_bus=None):
        """Returns a `TenantGraphRAG` bound to `tenant_id` for the lifetime
        of the returned object — tenant_id is never a per-call argument
        again, so a single query can't accidentally target the wrong
        tenant. Requires `runtime_url` to have been passed to the
        constructor and `setup_secure()` to have been run at least once.

        `event_bus` (an `observability.EventBus`) is optional; omit it to
        get the default logging-only sink.
        """
        from .tenancy import SecureGraphStore, _vector_column_type as _vt
        from .tenant_engine import TenantGraphRAG

        if not self._runtime_url:
            raise ValueError(
                "for_tenant() requires PostgresGraphRAG to be constructed with "
                "runtime_url set to the restricted RLS runtime role's connection string."
            )
        if self._secure_store is None:
            vector_type = _vt(self.extractor.config["dimension"])
            self._secure_store = SecureGraphStore(self._runtime_url, vector_type=vector_type)
        return TenantGraphRAG(
            tenant_id=tenant_id,
            store=self._secure_store,
            extractor=self.extractor,
            chunker=self.chunker,
            ingestion_config=self.ingestion_config,
            retrieval_config=self.retrieval_config,
            event_bus=event_bus,
        )

    async def setup(self):
        """Initializes the migration-safe database schema."""
        dimension = self.extractor.config["dimension"]
        await self.db.setup_database(embedding_dimension=dimension)

    async def _extract_with_retry(self, chunk: str) -> Optional[List[Triplet]]:
        """Extracts triplets from a chunk, retrying transient failures with
        exponential backoff. Returns None (rather than raising) if all
        attempts fail, so one bad chunk doesn't abort an entire ingestion
        batch."""
        max_retries = self.ingestion_config["max_extraction_retries"]
        base_delay = self.ingestion_config["retry_base_delay"]
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                return await self.extractor.extract_triplets(chunk)
            except Exception as exc:  # noqa: BLE001 - provider errors vary by SDK
                last_exc = exc
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2**attempt))

        logger.warning(
            "Extraction failed for chunk (len=%d) after %d attempts: %s",
            len(chunk),
            max_retries,
            last_exc,
        )
        return None

    async def add_texts(
        self,
        texts: Union[str, List[str]],
        namespace: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Ingests one or more texts into a specific namespace.

        - Chunks are hashed and checked against previously-ingested content
          for this namespace so re-running the same source does not
          re-trigger (paid) LLM extraction.
        - Extraction runs with bounded concurrency and retries; a chunk that
          keeps failing is skipped (and logged) rather than aborting the
          whole batch.
        - Entities are resolved through a layered strategy (exact
          normalized match, then trigram + embedding fuzzy match, then new
          node) before a single bulk upsert.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_chunks: List[str] = []
        for text in texts:
            all_chunks.extend(self.chunker(text))

        if not all_chunks:
            return

        if len(all_chunks) > MAX_CHUNKS_PER_INGEST_CALL:
            raise ValueError(
                f"add_texts() received {len(all_chunks)} chunks, which exceeds "
                f"the hard limit of {MAX_CHUNKS_PER_INGEST_CALL} per call. "
                "Split the input across multiple add_texts() calls."
            )

        # Idempotency (skip-if-seen) is only safe when this call has no
        # per-call metadata to attach: we don't track which entities/edges a
        # given chunk previously produced (the same provenance gap noted on
        # `ingested_chunks` in database.py), so there is no node to merge
        # new metadata onto if we skip extraction. A call that supplies
        # metadata always runs fully, so metadata is never silently dropped.
        if self.ingestion_config["skip_duplicate_chunks"] and not metadata:
            chunks_to_process = await self.db.filter_new_chunks(
                all_chunks, namespace=namespace
            )
            skipped = len(all_chunks) - len(chunks_to_process)
            if skipped:
                logger.info(
                    "Skipping %d already-ingested chunk(s) in namespace '%s'",
                    skipped,
                    namespace,
                )
        else:
            chunks_to_process = all_chunks

        if not chunks_to_process:
            return

        # 1. Extraction (bounded concurrency across chunks, retried on failure)
        semaphore = asyncio.Semaphore(
            self.ingestion_config["max_concurrent_extractions"]
        )

        async def _bounded_extract(chunk: str):
            async with semaphore:
                return await self._extract_with_retry(chunk)

        extraction_results = await asyncio.gather(
            *[_bounded_extract(c) for c in chunks_to_process]
        )

        succeeded_chunks = []
        all_triplets: List[Triplet] = []
        unique_entities = set()
        for chunk, triplets in zip(chunks_to_process, extraction_results):
            if triplets is None:
                continue  # extraction failed after retries; skip, don't mark ingested
            succeeded_chunks.append((chunk, len(triplets)))
            for t in triplets:
                all_triplets.append(t)
                unique_entities.add(t.subject)
                unique_entities.add(t.object)

        if not unique_entities:
            if succeeded_chunks:
                await self.db.mark_chunks_ingested(
                    [c for c, _ in succeeded_chunks],
                    namespace=namespace,
                    triplet_counts=[n for _, n in succeeded_chunks],
                )
            return

        # 2. Batch Embedding Retrieval
        entity_list = list(unique_entities)
        embeddings = await self.extractor.get_embedding(entity_list)
        entity_to_emb = dict(zip(entity_list, embeddings))

        # 3. Entity resolution + batch DB write, all in one transaction
        await self.db._init_pool()
        async with self.db.pool.connection() as conn:
            entities_payload = [
                {
                    "content": entity,
                    "embedding": entity_to_emb[entity],
                    "metadata": metadata,
                }
                for entity in entity_list
            ]

            content_to_id = await self.db.resolve_and_upsert_nodes_batch(
                entities_payload,
                namespace=namespace,
                fuzzy=self.ingestion_config["fuzzy_entity_resolution"],
                trgm_threshold=self.ingestion_config["fuzzy_trgm_threshold"],
                embedding_threshold=self.ingestion_config["fuzzy_embedding_threshold"],
                connection=conn,
            )

            edges_data = [
                {
                    "source_id": content_to_id[t.subject],
                    "target_id": content_to_id[t.object],
                    "relation": t.predicate,
                    "metadata": metadata,
                }
                for t in all_triplets
                if t.subject != t.object
            ]

            await self.db.upsert_edges_batch(
                edges_data, namespace=namespace, connection=conn
            )

            if succeeded_chunks:
                await self.db.mark_chunks_ingested(
                    [c for c, _ in succeeded_chunks],
                    namespace=namespace,
                    triplet_counts=[n for _, n in succeeded_chunks],
                    connection=conn,
                )

            await conn.commit()

    async def query_structured(
        self,
        question: str,
        namespace: str = "default",
        **overrides: Any,
    ) -> RetrievalResult:
        """Runs the full retrieval pipeline and returns structured,
        explainable evidence (nodes/edges with scores, hop distances, and
        provenance metadata) instead of a pre-flattened string.

        Any field of ``RetrievalConfig`` (top_k, hops, directed,
        relation_types, exclude_relation_types, min_weight, score_decay,
        max_context_nodes, max_context_edges) can be overridden per call.
        """
        cfg: RetrievalConfig = {**self.retrieval_config, **overrides}

        query_emb = await self.extractor.get_embedding(question)
        if (
            isinstance(query_emb, list)
            and query_emb
            and isinstance(query_emb[0], list)
        ):
            # get_embedding's return type depends on its input type; this
            # branch should be unreachable for a single string question.
            query_emb = query_emb[0]

        await self.db._init_pool()
        async with self.db.pool.connection() as conn:
            seed_nodes = await self.db.vector_search(
                query_emb, namespace=namespace, top_k=cfg["top_k"], connection=conn
            )
            seed_ids = [n["id"] for n in seed_nodes]
            # cosine distance -> similarity, clamped to [0, 1] as the seed's
            # relevance score, which then decays outward during traversal.
            seed_scores = {
                n["id"]: max(0.0, min(1.0, 1.0 - n["distance"])) for n in seed_nodes
            }

            graph_data = await self.db.traverse_graph(
                seed_ids,
                namespace=namespace,
                max_hops=cfg["hops"],
                seed_scores=seed_scores,
                directed=cfg["directed"],
                relation_types=cfg["relation_types"],
                exclude_relation_types=cfg["exclude_relation_types"],
                min_weight=cfg["min_weight"],
                score_decay=cfg["score_decay"],
                max_neighbors_per_node=cfg["max_neighbors_per_node"],
                connection=conn,
            )

        nodes = [
            RetrievedNode(
                id=str(n["id"]),
                content=n["content"],
                metadata=n.get("metadata") or {},
                hop_distance=n["hop_distance"],
                score=n["score"] if n["score"] is not None else 0.0,
            )
            for n in graph_data["nodes"]
        ]
        nodes.sort(key=lambda n: n.score, reverse=True)
        nodes = nodes[: cfg["max_context_nodes"]]
        kept_ids = {n.id for n in nodes}

        edges = [
            RetrievedEdge(
                source_id=str(e["source_node_id"]),
                target_id=str(e["target_node_id"]),
                source_content=e["source_content"],
                target_content=e["target_content"],
                relation=e["relation"],
                weight=e["weight"],
                metadata=e.get("metadata") or {},
            )
            for e in graph_data["edges"]
            if str(e["source_node_id"]) in kept_ids
            and str(e["target_node_id"]) in kept_ids
        ]
        edges.sort(key=lambda e: e.weight, reverse=True)
        edges = edges[: cfg["max_context_edges"]]

        return RetrievalResult(nodes=nodes, edges=edges, seed_ids=seed_ids)

    async def query(
        self,
        question: str,
        namespace: str = "default",
        hops: Optional[int] = None,
        top_k: Optional[int] = None,
        **overrides: Any,
    ) -> str:
        """Searches the graph and returns enriched context as a string.

        Convenience wrapper around ``query_structured`` for callers that
        just want a prompt-ready context blob. Use ``query_structured`` to
        get scores, hop distances, and provenance for custom formatting.
        """
        if hops is not None:
            overrides["hops"] = hops
        if top_k is not None:
            overrides["top_k"] = top_k
        result = await self.query_structured(question, namespace=namespace, **overrides)
        return result.to_context_string()
