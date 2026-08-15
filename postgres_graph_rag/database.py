import hashlib
import json
import logging
import math
import re
from typing import List, Dict, Any, Optional, Tuple
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger("postgres_graph_rag")

_WHITESPACE_RE = re.compile(r"\s+")

# pgvector HNSW indexes only support up to 2000 dimensions on `vector` and up
# to 4000 on the half-precision `halfvec` type; beyond that there is no ANN
# index available and only exact (sequential-scan) search works.
# https://github.com/pgvector/pgvector#hnsw
_HNSW_VECTOR_MAX_DIM = 2000
_HNSW_HALFVEC_MAX_DIM = 4000


def _vector_column_type(embedding_dimension: int) -> str:
    """Picks the narrowest pgvector column type that can still get an HNSW
    index at this dimension, falling back to plain `vector` (no ANN index)
    beyond what pgvector supports at all."""
    if embedding_dimension <= _HNSW_VECTOR_MAX_DIM:
        return "vector"
    if embedding_dimension <= _HNSW_HALFVEC_MAX_DIM:
        return "halfvec"
    return "vector"


def normalize_entity(content: str) -> str:
    """Canonicalizes an entity string for duplicate detection.

    Collapses internal whitespace and trims leading/trailing whitespace so
    that trivially-different mentions of the same entity (e.g. extra spaces
    introduced by chunking) resolve to the same node. Case is preserved
    because entity casing is often meaningful for display (e.g. acronyms),
    and case-insensitive comparison is instead handled explicitly by the
    fuzzy-resolution path.
    """
    return _WHITESPACE_RE.sub(" ", content).strip()


def content_hash(text: str) -> str:
    """Stable hash used to detect whether a chunk has already been ingested."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_float_list(value: Any) -> List[float]:
    """Normalizes a value read back from a `vector` column into a plain
    list of floats. psycopg has no built-in pgvector codec, so without an
    explicit (connection-scoped) adapter registration these come back as
    the raw Postgres text representation, e.g. "[0.1,0.2,0.3]"."""
    if isinstance(value, str):
        return [float(x) for x in value.strip("[]").split(",")]
    return list(value)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Hard safety limits, independent of any per-call config, that bound the
# blast radius of a single traversal/write regardless of caller input.
MAX_HOPS_HARD_LIMIT = 5
MAX_ROWS_PER_STATEMENT = 1000  # rows per multi-row INSERT before we chunk


class DatabaseManager:
    def __init__(
        self,
        connection_url: str,
        pool_min_size: int = 1,
        pool_max_size: int = 10,
        pool_timeout_s: float = 30.0,
        pool_max_idle_s: float = 300.0,
        pool_max_lifetime_s: float = 3600.0,
        statement_timeout_ms: int = 30_000,
    ):
        self.connection_url = connection_url
        self.pool: Optional[AsyncConnectionPool] = None
        self.embedding_dimension: Optional[int] = None
        self.vector_type: str = "vector"
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool_timeout_s = pool_timeout_s
        self._pool_max_idle_s = pool_max_idle_s
        self._pool_max_lifetime_s = pool_max_lifetime_s
        self._statement_timeout_ms = statement_timeout_ms

    async def _init_pool(self):
        """Lazily initializes the connection pool if it doesn't exist."""
        if self.pool is None:

            async def _configure(conn: psycopg.AsyncConnection):
                # The pool's `configure` contract requires connections to be
                # handed back idle (not mid-transaction); SET alone leaves
                # one open under psycopg's default autocommit=False, so we
                # must commit it explicitly or the pool discards and retries
                # this connection forever.
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"SET statement_timeout = {int(self._statement_timeout_ms)}"
                    )
                await conn.commit()

            self.pool = AsyncConnectionPool(
                self.connection_url,
                open=False,  # Wait for explicit open
                kwargs={"row_factory": dict_row},
                min_size=self._pool_min_size,
                max_size=self._pool_max_size,
                timeout=self._pool_timeout_s,
                max_idle=self._pool_max_idle_s,
                max_lifetime=self._pool_max_lifetime_s,
                configure=_configure,
            )
            await self.pool.open()

    async def close(self):
        """Closes the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def setup_database(self, embedding_dimension: int = 1536):
        """Creates the core schema (idempotent via CREATE TABLE/INDEX IF NOT
        EXISTS). This is not a migration system: core columns (namespace,
        content, embedding type/dimension, relation) are fixed once created
        and are not auto-migrated if you need to change them later."""
        # Defense in depth: embedding_dimension is interpolated into DDL
        # below (Postgres does not support parameterizing a type modifier).
        # It should only ever originate from a developer-supplied
        # ProviderConfig, but we validate it strictly regardless so a
        # malformed/attacker-influenced config can't inject SQL via this path.
        if not isinstance(embedding_dimension, int) or not (
            0 < embedding_dimension <= 16000
        ):
            raise ValueError(
                f"Invalid embedding_dimension: {embedding_dimension!r}. "
                "Must be a positive integer (pgvector max is 16000)."
            )

        vector_type = _vector_column_type(embedding_dimension)
        use_ann_index = embedding_dimension <= _HNSW_HALFVEC_MAX_DIM
        if not use_ann_index:
            logger.warning(
                "embedding_dimension=%d exceeds pgvector's HNSW limit (%d for "
                "halfvec). No ANN index will be created; vector_search will "
                "fall back to an exact sequential scan, which does not scale "
                "past small graphs. Consider a smaller-dimension embedding model.",
                embedding_dimension,
                _HNSW_HALFVEC_MAX_DIM,
            )

        await self._init_pool()
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Enable pgvector + pg_trgm (fuzzy entity resolution) extensions
                # Extensions must not be installed into a caller-provided
                # search_path (the test suite and some managed services use a
                # tenant/temp schema as the first path entry).  Keeping the
                # extension objects in public makes subsequent connections
                # resolve pgvector types consistently.
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
                await cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public")

                # Create graph_nodes table (The Entities)
                await cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS graph_nodes (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        namespace VARCHAR(255) NOT NULL,
                        content TEXT NOT NULL,
                        embedding {vector_type}({embedding_dimension}),
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Check if embedding dimension AND column type match if the
                # table already existed (a dimension can map to either
                # vector/halfvec depending on which side of the HNSW cutoff
                # it falls on; changing the embedding model must not silently
                # reuse an incompatible column).
                await cur.execute(
                    """
                    SELECT format_type(atttypid, atttypmod) AS full_type
                    FROM pg_attribute
                    WHERE attrelid = 'graph_nodes'::regclass
                      AND attname = 'embedding'
                """
                )
                row = await cur.fetchone()
                if row:
                    full_type = row["full_type"] if isinstance(row, dict) else row[0]
                    expected = f"{vector_type}({embedding_dimension})"
                    if full_type != expected:
                        raise ValueError(
                            f"Database has 'graph_nodes.embedding' typed as {full_type!r}, "
                            f"but current config expects {expected!r}. "
                            "Please drop the table or use a compatible embedding model."
                        )

                self.embedding_dimension = embedding_dimension
                self.vector_type = vector_type

                # Create graph_edges table (The Relationships)
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS graph_edges (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        namespace VARCHAR(255) NOT NULL,
                        source_node_id UUID REFERENCES graph_nodes(id) ON DELETE CASCADE,
                        target_node_id UUID REFERENCES graph_nodes(id) ON DELETE CASCADE,
                        relation TEXT NOT NULL,
                        weight FLOAT DEFAULT 1.0,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(namespace, source_node_id, target_node_id, relation)
                    )
                """
                )

                # Tracks which source chunks have already been extracted so
                # that re-ingesting the same content is a cheap no-op instead
                # of re-running (paid) LLM extraction calls.
                #
                # KNOWN LIMITATION: this is keyed on (namespace, chunk_hash)
                # only, with no document identity. If the same sentence
                # appears in two different documents in the same namespace,
                # the second document's occurrence of that chunk is treated
                # as already-ingested and its extraction is skipped (the
                # resulting graph facts are still correct, since it's the
                # same text -> same triplets -> same nodes/edges, so this is
                # a provenance gap, not a retrieval-correctness bug). It does
                # mean there is currently no way to answer "which documents
                # mention X" or to safely delete one document's contribution
                # without affecting another's. Fixing this properly requires
                # a documents/chunks/mentions evidence schema (tracked as a
                # planned v0.3-scope change) rather than a patch here.
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ingested_chunks (
                        namespace VARCHAR(255) NOT NULL,
                        chunk_hash CHAR(64) NOT NULL,
                        triplet_count INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (namespace, chunk_hash)
                    )
                """
                )

                # Indices for fast lookups and namespacing
                await cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_nodes_namespace_content ON graph_nodes (namespace, content)"
                )
                if use_ann_index:
                    ops_class = (
                        "vector_cosine_ops"
                        if vector_type == "vector"
                        else "halfvec_cosine_ops"
                    )
                    await cur.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_graph_nodes_embedding "
                        f"ON graph_nodes USING hnsw (embedding {ops_class})"
                    )
                await cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_graph_nodes_content_trgm ON graph_nodes USING gin (content gin_trgm_ops)"
                )
                await cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges (source_node_id)"
                )
                await cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges (target_node_id)"
                )
                await cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_graph_edges_namespace ON graph_edges (namespace)"
                )

                await conn.commit()

    # ------------------------------------------------------------------
    # Idempotent ingestion bookkeeping
    # ------------------------------------------------------------------

    async def filter_new_chunks(
        self,
        chunks: List[str],
        namespace: str = "default",
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> List[str]:
        """Returns the subset of `chunks` that have not already been ingested
        (by content hash) into this namespace, so callers can skip paying for
        LLM extraction on unchanged content when a source is re-ingested."""
        if not chunks:
            return []

        hashes = [content_hash(c) for c in chunks]

        async def _query(conn: psycopg.AsyncConnection) -> List[str]:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT chunk_hash FROM ingested_chunks WHERE namespace = %s AND chunk_hash = ANY(%s)",
                    (namespace, hashes),
                )
                rows = await cur.fetchall()
                seen = {r["chunk_hash"] for r in rows}
            return [c for c, h in zip(chunks, hashes) if h not in seen]

        if connection:
            return await _query(connection)

        await self._init_pool()
        async with self.pool.connection() as conn:
            return await _query(conn)

    async def mark_chunks_ingested(
        self,
        chunks: List[str],
        namespace: str,
        triplet_counts: Optional[List[int]] = None,
        connection: Optional[psycopg.AsyncConnection] = None,
    ):
        """Records that `chunks` have been processed, so future ingestion of
        the same source can skip re-extraction."""
        if not chunks:
            return
        triplet_counts = triplet_counts or [0] * len(chunks)
        rows = [
            (namespace, content_hash(c), count)
            for c, count in zip(chunks, triplet_counts)
        ]

        async def _insert(conn: psycopg.AsyncConnection):
            async with conn.cursor() as cur:
                await cur.executemany(
                    """
                    INSERT INTO ingested_chunks (namespace, chunk_hash, triplet_count)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (namespace, chunk_hash) DO UPDATE SET
                        triplet_count = EXCLUDED.triplet_count
                    """,
                    rows,
                )

        if connection:
            await _insert(connection)
            return

        await self._init_pool()
        async with self.pool.connection() as conn:
            await _insert(conn)
            await conn.commit()

    # ------------------------------------------------------------------
    # Node / edge writes
    # ------------------------------------------------------------------

    async def upsert_node(
        self,
        content: str,
        embedding: List[float],
        namespace: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> str:
        """Inserts or updates a single node within a namespace and returns its ID."""
        ids = await self.upsert_nodes_batch(
            [{"content": content, "embedding": embedding, "metadata": metadata}],
            namespace=namespace,
            connection=connection,
        )
        return ids[0]

    async def upsert_nodes_batch(
        self,
        nodes: List[Dict[str, Any]],
        namespace: str = "default",
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> List[str]:
        """Upserts multiple nodes with multi-row INSERTs, chunked to at most
        `MAX_ROWS_PER_STATEMENT` rows per statement (still far fewer round
        trips than one-row-at-a-time, while keeping any single statement's
        parameter count and planning cost bounded regardless of how large a
        batch a caller passes in), and returns their IDs in input order.

        Nodes sharing the same content are de-duplicated first: Postgres'
        ON CONFLICT DO UPDATE cannot touch the same row twice within one
        statement."""
        if not nodes:
            return []

        deduped: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            content = node["content"]
            if content not in deduped:
                deduped[content] = {"embedding": node["embedding"], "metadata": {}}
            deduped[content]["metadata"] = {
                **deduped[content]["metadata"],
                **(node.get("metadata") or {}),
            }
            # Last embedding wins if content repeats within a batch; the
            # embeddings should be identical/near-identical for the same
            # string in practice.
            deduped[content]["embedding"] = node["embedding"]

        items = list(deduped.items())

        async def _run(conn: psycopg.AsyncConnection) -> List[str]:
            by_content: Dict[str, str] = {}
            async with conn.cursor() as cur:
                for start in range(0, len(items), MAX_ROWS_PER_STATEMENT):
                    chunk = items[start : start + MAX_ROWS_PER_STATEMENT]
                    values_sql = ", ".join(["(%s, %s, %s, %s)"] * len(chunk))
                    params: List[Any] = []
                    for content, data in chunk:
                        params.extend(
                            [namespace, content, data["embedding"], json.dumps(data["metadata"])]
                        )
                    await cur.execute(
                        f"""
                        INSERT INTO graph_nodes (namespace, content, embedding, metadata)
                        VALUES {values_sql}
                        ON CONFLICT (namespace, content) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            metadata = graph_nodes.metadata || EXCLUDED.metadata
                        RETURNING id, content
                        """,
                        params,
                    )
                    for r in await cur.fetchall():
                        by_content[r["content"]] = str(r["id"])
            return [by_content[n["content"]] for n in nodes]

        if connection:
            return await _run(connection)

        await self._init_pool()
        async with self.pool.connection() as conn:
            res = await _run(conn)
            await conn.commit()
            return res

    async def resolve_and_upsert_nodes_batch(
        self,
        entities: List[Dict[str, Any]],
        namespace: str = "default",
        fuzzy: bool = True,
        trgm_threshold: float = 0.4,
        embedding_threshold: float = 0.90,
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> Dict[str, str]:
        """Layered entity resolution + upsert.

        Each entity in `entities` is a dict with `content`, `embedding`, and
        optional `metadata`. `content` is first normalized (whitespace only).
        Resolution proceeds in layers:

          1. Exact match on the normalized string within the namespace.
          2. (if `fuzzy`) Trigram-similarity candidate lookup, confirmed by
             embedding cosine similarity, to catch near-duplicate mentions
             (e.g. "Apple Inc." vs "Apple") without merging unrelated
             entities that merely share short substrings.
          3. Otherwise, a brand new node is created.

        Returns a mapping from the *original* (pre-normalization) content
        string to the resolved node UUID.
        """
        if not entities:
            return {}

        normalized = [
            (e["content"], normalize_entity(e["content"]), e["embedding"], e.get("metadata"))
            for e in entities
        ]

        async def _run(conn: psycopg.AsyncConnection) -> Dict[str, str]:
            original_to_norm: Dict[str, str] = {}
            norm_to_embedding: Dict[str, List[float]] = {}
            norm_to_metadata: Dict[str, Optional[Dict[str, Any]]] = {}
            for original, norm, embedding, metadata in normalized:
                original_to_norm[original] = norm
                norm_to_embedding[norm] = embedding
                norm_to_metadata[norm] = metadata

            unique_norms = list(dict.fromkeys(original_to_norm.values()))
            resolved: Dict[str, str] = {}  # norm -> existing node id

            # Layer 1: exact match
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT id, content FROM graph_nodes WHERE namespace = %s AND content = ANY(%s)",
                    (namespace, unique_norms),
                )
                for row in await cur.fetchall():
                    resolved[row["content"]] = str(row["id"])

            # Refresh embedding and merge metadata onto exact-match nodes,
            # mirroring upsert_nodes_batch's ON CONFLICT DO UPDATE semantics.
            # Without this, calling add_texts() again on already-known
            # content with new metadata would resolve to the existing node
            # via this exact-match path and then silently do nothing with
            # the new metadata (caught by test_ingest_does_not_skip_when_
            # metadata_provided + a live end-to-end test failure against a
            # real database).
            if resolved:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """
                        UPDATE graph_nodes SET
                            embedding = %s,
                            metadata = metadata || %s
                        WHERE id = %s
                        """,
                        [
                            (
                                norm_to_embedding[norm],
                                json.dumps(norm_to_metadata[norm] or {}),
                                node_id,
                            )
                            for norm, node_id in resolved.items()
                        ],
                    )

            unmatched = [n for n in unique_norms if n not in resolved]

            # Layer 2: fuzzy match (trigram candidate + embedding confirmation)
            fuzzy_merges: Dict[str, str] = {}
            if fuzzy and unmatched:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT q.content AS query_content, c.id, c.content AS matched_content,
                               c.embedding AS matched_embedding, c.sim
                        FROM unnest(%(unmatched)s::text[]) AS q(content)
                        LEFT JOIN LATERAL (
                            SELECT id, content, embedding, similarity(content, q.content) AS sim
                            FROM graph_nodes
                            WHERE namespace = %(namespace)s AND content %% q.content
                            ORDER BY sim DESC
                            LIMIT 1
                        ) c ON true
                        WHERE c.sim >= %(trgm_threshold)s
                        """,
                        {
                            "unmatched": unmatched,
                            "namespace": namespace,
                            "trgm_threshold": trgm_threshold,
                        },
                    )
                    candidates = await cur.fetchall()

                for cand in candidates:
                    query_content = cand["query_content"]
                    query_embedding = norm_to_embedding[query_content]
                    matched_embedding = _as_float_list(cand["matched_embedding"])
                    cos_sim = _cosine_similarity(query_embedding, matched_embedding)
                    if cos_sim >= embedding_threshold:
                        fuzzy_merges[query_content] = str(cand["id"])

            still_unmatched = [n for n in unmatched if n not in fuzzy_merges]

            # Layer 3: create new nodes for anything left over
            new_ids: Dict[str, str] = {}
            if still_unmatched:
                new_nodes = [
                    {
                        "content": n,
                        "embedding": norm_to_embedding[n],
                        "metadata": norm_to_metadata[n],
                    }
                    for n in still_unmatched
                ]
                ids = await self.upsert_nodes_batch(
                    new_nodes, namespace=namespace, connection=conn
                )
                new_ids = dict(zip(still_unmatched, ids))

            # Merge resolved metadata onto fuzzy-merged existing nodes so
            # provenance/context accumulates instead of being silently dropped.
            if fuzzy_merges:
                merge_rows = [
                    (
                        fuzzy_merges[n],
                        json.dumps(
                            {
                                **(norm_to_metadata[n] or {}),
                                "_resolution": {
                                    "merged_from": n,
                                    "method": "fuzzy_trgm_embedding",
                                },
                            }
                        ),
                    )
                    for n in fuzzy_merges
                ]
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "UPDATE graph_nodes SET metadata = metadata || %s WHERE id = %s",
                        [(meta, node_id) for node_id, meta in merge_rows],
                    )

            norm_to_id = {**resolved, **fuzzy_merges, **new_ids}
            return {
                original: norm_to_id[norm] for original, norm in original_to_norm.items()
            }

        if connection:
            return await _run(connection)

        await self._init_pool()
        async with self.pool.connection() as conn:
            res = await _run(conn)
            await conn.commit()
            return res

    async def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        namespace: str = "default",
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        connection: Optional[psycopg.AsyncConnection] = None,
    ):
        """Inserts or updates a single edge within a namespace."""
        await self.upsert_edges_batch(
            [
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation": relation,
                    "weight": weight,
                    "metadata": metadata,
                }
            ],
            namespace=namespace,
            connection=connection,
        )

    async def upsert_edges_batch(
        self,
        edges: List[Dict[str, Any]],
        namespace: str = "default",
        connection: Optional[psycopg.AsyncConnection] = None,
    ):
        """Upserts multiple edges with a single multi-row INSERT. Repeated
        mentions of the same (source, target, relation) triple increment the
        edge weight, giving frequently-restated relationships a higher score
        during retrieval instead of being silently deduplicated.

        Edges within the same call that share a (source, target, relation)
        key are pre-aggregated in Python: Postgres' ON CONFLICT DO UPDATE
        cannot touch the same row twice within one statement, and this also
        happens to be exactly the "mention frequency" signal we want to
        fold into weight."""
        if not edges:
            return

        aggregated: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for edge in edges:
            key = (edge["source_id"], edge["target_id"], edge["relation"])
            if key not in aggregated:
                aggregated[key] = {
                    "weight": 0.0,
                    "metadata": {},
                }
            aggregated[key]["weight"] += edge.get("weight", 1.0)
            aggregated[key]["metadata"] = {
                **aggregated[key]["metadata"],
                **(edge.get("metadata") or {}),
            }

        items = list(aggregated.items())

        async def _run(conn: psycopg.AsyncConnection):
            async with conn.cursor() as cur:
                for start in range(0, len(items), MAX_ROWS_PER_STATEMENT):
                    chunk = items[start : start + MAX_ROWS_PER_STATEMENT]
                    values_sql = ", ".join(["(%s, %s, %s, %s, %s, %s)"] * len(chunk))
                    params: List[Any] = []
                    for (source_id, target_id, relation), agg in chunk:
                        params.extend(
                            [
                                namespace,
                                source_id,
                                target_id,
                                relation,
                                agg["weight"],
                                json.dumps(agg["metadata"]),
                            ]
                        )
                    await cur.execute(
                        f"""
                        INSERT INTO graph_edges (namespace, source_node_id, target_node_id, relation, weight, metadata)
                        VALUES {values_sql}
                        ON CONFLICT (namespace, source_node_id, target_node_id, relation) DO UPDATE SET
                            weight = graph_edges.weight + EXCLUDED.weight,
                            metadata = graph_edges.metadata || EXCLUDED.metadata
                        """,
                        params,
                    )

        if connection:
            await _run(connection)
            return

        await self._init_pool()
        async with self.pool.connection() as conn:
            await _run(conn)
            await conn.commit()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        query_embedding: List[float],
        namespace: str = "default",
        top_k: int = 5,
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> List[Dict[str, Any]]:
        """Finds nodes within a namespace most similar to the query embedding."""
        if connection:
            return await self._vector_search_with_conn(
                connection, query_embedding, namespace, top_k
            )

        await self._init_pool()
        async with self.pool.connection() as conn:
            return await self._vector_search_with_conn(
                conn, query_embedding, namespace, top_k
            )

    async def _vector_search_with_conn(
        self,
        conn: psycopg.AsyncConnection,
        query_embedding: List[float],
        namespace: str = "default",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT id, content, metadata, (embedding <=> %s::{self.vector_type}) as distance
                FROM graph_nodes
                WHERE namespace = %s
                ORDER BY distance ASC
                LIMIT %s
            """,
                (query_embedding, namespace, top_k),
            )
            return await cur.fetchall()

    async def traverse_graph(
        self,
        seed_node_ids: List[str],
        namespace: str = "default",
        max_hops: int = 2,
        seed_scores: Optional[Dict[str, float]] = None,
        directed: bool = False,
        relation_types: Optional[List[str]] = None,
        exclude_relation_types: Optional[List[str]] = None,
        min_weight: float = 0.0,
        score_decay: float = 0.7,
        max_neighbors_per_node: int = 20,
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Performs a namespaced recursive traversal to find neighbors within
        N hops, with explainable per-node scores and hop distances.

        Args:
            directed: if True, only follows edges in their source->target
                direction (e.g. "A depends_on B" will reach B from A but not
                A from B). If False (default), edges are treated as
                undirected for traversal purposes.
            relation_types: if set, only traverse edges whose relation is in
                this allow-list.
            exclude_relation_types: if set, never traverse edges whose
                relation is in this deny-list.
            min_weight: skip edges with weight below this threshold.
            score_decay: per-hop multiplicative decay applied to a seed's
                relevance score as it propagates outward, combined with edge
                weight, so evidence closer to a relevant seed (via stronger
                edges) ranks higher than a distant, weakly-connected node.
            max_neighbors_per_node: caps fan-out per node at each hop to the
                `max_neighbors_per_node` highest-weight edges, so a single
                hub node (or an adversarial/degenerate graph) can't blow up
                the size of the expansion combinatorially across hops.

        Raises:
            ValueError: if `max_hops` exceeds the hard safety ceiling
                (`MAX_HOPS_HARD_LIMIT`) — deep recursive traversal on a dense
                graph grows the candidate set combinatorially, so this isn't
                just a UX default, it protects the database from a runaway
                query.
        """
        if max_hops > MAX_HOPS_HARD_LIMIT:
            raise ValueError(
                f"max_hops={max_hops} exceeds the hard limit of "
                f"{MAX_HOPS_HARD_LIMIT}. Deep traversal on a dense graph "
                "grows combinatorially; use a smaller max_hops or narrow "
                "relation_types/min_weight instead."
            )

        if not seed_node_ids:
            return {"nodes": [], "edges": []}

        seed_scores = seed_scores or {sid: 1.0 for sid in seed_node_ids}
        scores_array = [seed_scores.get(sid, 1.0) for sid in seed_node_ids]

        if connection:
            return await self._traverse_graph_with_conn(
                connection,
                seed_node_ids,
                scores_array,
                namespace,
                max_hops,
                directed,
                relation_types,
                exclude_relation_types,
                min_weight,
                score_decay,
                max_neighbors_per_node,
            )

        await self._init_pool()
        async with self.pool.connection() as conn:
            return await self._traverse_graph_with_conn(
                conn,
                seed_node_ids,
                scores_array,
                namespace,
                max_hops,
                directed,
                relation_types,
                exclude_relation_types,
                min_weight,
                score_decay,
                max_neighbors_per_node,
            )

    async def _traverse_graph_with_conn(
        self,
        conn: psycopg.AsyncConnection,
        seed_node_ids: List[str],
        seed_scores: List[float],
        namespace: str,
        max_hops: int,
        directed: bool,
        relation_types: Optional[List[str]],
        exclude_relation_types: Optional[List[str]],
        min_weight: float,
        score_decay: float,
        max_neighbors_per_node: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                WITH RECURSIVE seeds AS (
                    SELECT unnest(%(seed_ids)s::uuid[]) AS id,
                           unnest(%(seed_scores)s::float[]) AS seed_score
                ),
                graph_expansion AS (
                    -- Base case: seed nodes
                    SELECT n.id, n.content, n.metadata, 0 AS depth,
                           ARRAY[n.id] AS visited, s.seed_score AS score
                    FROM graph_nodes n
                    JOIN seeds s ON s.id = n.id
                    WHERE n.namespace = %(namespace)s

                    UNION ALL

                    -- Recursive step: walk to each node's highest-weight
                    -- neighbors (capped at max_neighbors_per_node via the
                    -- LATERAL's ORDER BY + LIMIT), respecting direction/filters.
                    -- Capping fan-out here — not just via min_weight — is what
                    -- keeps a hub node from making the expansion blow up
                    -- combinatorially across hops.
                    SELECT n.id, n.content, n.metadata, ge.depth + 1,
                           ge.visited || n.id,
                           -- nb.weight is an unbounded mention count (see
                           -- upsert_edges_batch); (1 - exp(-weight)) saturates
                           -- it to (0, 1) so a heavily-repeated edge can't
                           -- blow the running score past a normalized range.
                           ge.score * %(decay)s * (1 - exp(-nb.weight)) AS score
                    FROM graph_expansion ge
                    CROSS JOIN LATERAL (
                        SELECT
                            CASE WHEN e.source_node_id = ge.id
                                 THEN e.target_node_id ELSE e.source_node_id END AS neighbor_id,
                            e.weight
                        FROM graph_edges e
                        WHERE e.namespace = %(namespace)s
                          AND (e.source_node_id = ge.id OR (NOT %(directed)s AND e.target_node_id = ge.id))
                          AND e.weight >= %(min_weight)s
                          AND (%(relation_types)s::text[] IS NULL OR e.relation = ANY(%(relation_types)s))
                          AND (%(exclude_relation_types)s::text[] IS NULL OR NOT (e.relation = ANY(%(exclude_relation_types)s)))
                        ORDER BY e.weight DESC
                        LIMIT %(max_neighbors)s
                    ) nb
                    JOIN graph_nodes n ON n.id = nb.neighbor_id
                    WHERE ge.depth < %(max_hops)s
                      AND n.id != ge.id
                      AND NOT (n.id = ANY(ge.visited))
                )
                SELECT id, content, metadata,
                       MIN(depth) AS hop_distance,
                       MAX(score) AS score
                FROM graph_expansion
                GROUP BY id, content, metadata
                ORDER BY score DESC NULLS LAST
                """,
                {
                    "seed_ids": seed_node_ids,
                    "seed_scores": seed_scores,
                    "namespace": namespace,
                    "max_hops": max_hops,
                    "directed": directed,
                    "relation_types": relation_types,
                    "exclude_relation_types": exclude_relation_types,
                    "min_weight": min_weight,
                    "decay": score_decay,
                    "max_neighbors": max_neighbors_per_node,
                },
            )
            nodes = await cur.fetchall()
            node_ids = [n["id"] for n in nodes]

            if node_ids:
                await cur.execute(
                    """
                    SELECT e.source_node_id, e.target_node_id, e.relation, e.metadata, e.weight,
                           s.content as source_content, t.content as target_content
                    FROM graph_edges e
                    JOIN graph_nodes s ON e.source_node_id = s.id
                    JOIN graph_nodes t ON e.target_node_id = t.id
                    WHERE e.source_node_id = ANY(%s)
                      AND e.target_node_id = ANY(%s)
                      AND e.namespace = %s
                """,
                    (node_ids, node_ids, namespace),
                )
                edges = await cur.fetchall()
            else:
                edges = []

            return {"nodes": nodes, "edges": edges}
