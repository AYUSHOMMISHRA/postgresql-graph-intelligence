"""Tenant-aware, RLS-secured storage layer (v0.3).

`SecureGraphStore` here is the v0.3 evidence-grounded, tenant-isolated
engine — new tables live in the `postgres_graph_rag` Postgres schema,
separate from the legacy `public.graph_nodes`/`graph_edges` tables that a
prior single-tenant deployment may still hold (see `migrate_legacy_data`
below for backfilling that data in).

Security model
---------------
`tenant_id` is the hard security boundary, enforced at the database level via
Row-Level Security (RLS), not just by applications remembering to filter on
it:

  * Every domain table has `tenant_id` and an RLS policy comparing it against
    `current_setting('postgres_graph_rag.tenant_id', true)` — the `true`
    ("missing_ok") means a session with no tenant context set gets NULL,
    and `tenant_id = NULL` never matches any row, so missing context fails
    closed (denies everything) rather than open.
  * The tenant GUC is set with `set_config(..., true)` — transaction-local,
    not session-local — because connections come from a pool and are reused
    across tenants; a session-level setting would leak from one tenant's
    request into the next one that happens to reuse the same physical
    connection.
  * RLS only restricts non-owner roles (or the owner too, if
    `FORCE ROW LEVEL SECURITY` is set, which we do for defense in depth).
    Migrations run as an admin/owner role; the *runtime* role used for actual
    queries must be a separate, non-superuser role without the BYPASSRLS
    attribute, or none of this does anything. `migrate_schema()` creates that
    role and `SecureGraphStore` is meant to connect as it, never as the
    admin/owner role.
"""

import hashlib
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ._db_utils import (
    _HNSW_HALFVEC_MAX_DIM,
    MAX_HOPS_HARD_LIMIT,
    MAX_ROWS_PER_STATEMENT,
    _as_float_list,
    _cosine_similarity,
    _vector_column_type,
    normalize_entity,
    content_hash,
)
from .filters import MetadataFilter, compile_metadata_filter

logger = logging.getLogger("postgres_graph_rag")

SCHEMA = "postgres_graph_rag"
TENANT_GUC = "postgres_graph_rag.tenant_id"

# documents.extraction_status: enforced here (SecureGraphStore.set_extraction_status
# raises before ever sending an invalid value to Postgres) and again by the
# CHECK constraint on the column itself (defense in depth — anything writing
# to this column outside set_extraction_status still can't violate it).
_VALID_EXTRACTION_STATUSES = frozenset({"pending", "ready", "partial", "failed"})

# Fixed identifier for data that predates multi-tenancy. Existing
# public.graph_nodes/graph_edges rows are backfilled under this tenant by
# migrate_schema(migrate_legacy_data=True) rather than being silently
# orphaned or discarded.
LEGACY_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

DEFAULT_LEASE_SECONDS = 120

# `websearch_to_tsquery` AND-conjoins every plain term in the input,
# regardless of text-search config. The `simple` config was chosen
# deliberately (see hybrid_search's docstring) so identifiers and
# non-English tokens survive tokenization unstemmed -- but `simple` also
# has no stopword list, so a natural-language question like "Who is the
# on-call contact for the team that owns the service affected by
# CVE-2026-1188?" becomes a query requiring "who" AND "is" AND "the" AND
# "for" AND "that" AND "by" (among the real content words) to ALL appear
# verbatim in the same short document. Against terse, factual chunks that
# essentially never happens, so the lexical half of "hybrid" search was
# silently contributing nothing for real questions -- confirmed directly:
# `websearch_to_tsquery('simple', question)` for that exact question
# returned zero matches against a small corpus that plainly contained the
# named identifier.
#
# The fix has two parts, and stopword-stripping alone turned out not to be
# enough -- also verified directly, not assumed: even after removing filler
# words, websearch_to_tsquery still AND-conjoins the remaining content
# words by default, and a genuinely multi-hop question's terms are often
# spread across several documents rather than co-occurring in one (e.g.
# "on-call" + "team" + "owns" + "service" + "CVE-2026-1188" each live in a
# *different* chunk here) -- so the AND'd query still matched nothing.
# Joining the remaining terms with explicit "OR" instead (which
# websearch_to_tsquery does honor as boolean OR) lets a chunk matching even
# one distinctive term register in the lexical ranking instead of
# contributing nothing at all to RRF fusion. This trades lexical precision
# for recall deliberately: the lexical branch here is a *candidate signal*
# fused with semantic similarity via RRF, not a standalone precise search,
# and precision is still enforced downstream by citation/anchor validation
# in answer(). `simple` stays the tokenization config either way, so
# hyphenated/numeric identifiers like "CVE-2026-1188" are preserved exactly
# as before.
_LEXICAL_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "but", "nor",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing",
    "has", "have", "had", "having",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how",
    "with", "by", "from", "as", "into", "onto", "about", "than", "then",
    "if", "so", "not", "no", "yes", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "there", "here",
})


def _lexical_query_terms(question: str) -> str:
    """Strips common English filler words from `question`, then joins the
    remaining content words/identifiers with explicit "OR" so
    `websearch_to_tsquery` treats them as candidates (any one matching is
    enough) rather than requiring every one to co-occur in the same chunk
    — see the module-level comment above `_LEXICAL_STOPWORDS` for why both
    steps turned out to be necessary. Falls back to the original text if
    stripping would leave nothing (a question that's entirely filler words
    is a degenerate case, not one worth over-engineering for)."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_.]*", question)
    kept = [w for w in words if w.lower() not in _LEXICAL_STOPWORDS]
    return " OR ".join(kept) if kept else question


def _tenant_rls_policy(table: str) -> str:
    """Fail-closed tenant isolation policy: a session with no tenant context
    set gets `current_setting(...) IS NULL`, and `tenant_id = NULL` is never
    true for any row, so missing context denies all access rather than
    accidentally granting it."""
    return f"""
        ALTER TABLE {SCHEMA}.{table} ENABLE ROW LEVEL SECURITY;
        ALTER TABLE {SCHEMA}.{table} FORCE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS tenant_isolation ON {SCHEMA}.{table};
        CREATE POLICY tenant_isolation ON {SCHEMA}.{table}
            USING (tenant_id = NULLIF(current_setting('{TENANT_GUC}', true), '')::uuid)
            WITH CHECK (tenant_id = NULLIF(current_setting('{TENANT_GUC}', true), '')::uuid);
    """


def _as_merge_list(existing: Any) -> List[str]:
    """Normalizes a node's `_merged_from` metadata value to a list. Rows
    written before this normalization existed may have it as a bare string
    (the original single-value-overwrite behavior) — read-and-normalize on
    the fly rather than requiring a data migration."""
    if existing is None:
        return []
    if isinstance(existing, list):
        return existing
    return [existing]


# Bumped whenever migrate_schema's DDL changes in a way worth recording in
# schema_migrations. This is NOT a versioned migration framework (no
# discrete numbered migration scripts, no down-migrations) — it's a single
# idempotent DDL script (CREATE ... IF NOT EXISTS throughout) that also
# leaves a queryable record of when it last ran and under what version, and
# refuses to run concurrently with itself. A real step-wise migration
# system (each schema change as its own tracked, individually-appliable
# script) remains a larger follow-up.
SCHEMA_VERSION = 7
MIGRATION_DESCRIPTION = (
    "documents.extraction_status gains a CHECK constraint restricting it to "
    "'pending'/'ready'/'partial'/'failed' (defense in depth alongside "
    "set_extraction_status()'s own application-level validation)"
)

# ----------------------------------------------------------------------
# Migration DDL: each fixed-text statement lives in exactly one constant
# (or, where a statement is parameterized per deployment, one small
# function) used both by _migrate_schema_locked() to actually run it and
# by MIGRATION_DDL_CHECKSUM to hash it — one shared source, so the hashed
# text can never drift from the executed text.
#
# Deliberately excluded from the hash: CREATE EXTENSION statements
# (environment bootstrap, not this migration's own shape), the one-time
# v2->v3 edge-weight backfill UPDATE (data, not schema shape), role
# CREATE/ALTER/PASSWORD and GRANT statements (operational, and the role
# statement embeds a plaintext secret that must never be hashed or
# logged), and the legacy-data-migration branch (operational, opt-in).
# ----------------------------------------------------------------------

_VECTOR_COL_PLACEHOLDER = "__VECTOR_COL__"
_VECTOR_OPS_PLACEHOLDER = "__VECTOR_OPS__"

_RLS_TABLES = (
    "documents",
    "document_chunks",
    "chunk_extractions",
    "graph_nodes",
    "graph_edges",
    "entity_mentions",
    "edge_mentions",
    "community_runs",
    "community_memberships",
    "community_dirty",
    "community_summaries",
)

_SCHEMA_MIGRATIONS_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.schema_migrations (
        version INT PRIMARY KEY,
        applied_at TIMESTAMPTZ DEFAULT now(),
        description TEXT,
        checksum CHAR(64)
    )
    """
_SCHEMA_MIGRATIONS_CHECKSUM_COLUMN_DDL = (
    f"ALTER TABLE {SCHEMA}.schema_migrations ADD COLUMN IF NOT EXISTS checksum CHAR(64)"
)
_SCHEMA_SETTINGS_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.schema_settings (
        singleton BOOLEAN PRIMARY KEY DEFAULT true CHECK (singleton),
        embedding_dimension INT NOT NULL,
        vector_type TEXT NOT NULL,
        schema_version INT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """
_DOCUMENTS_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id UUID NOT NULL,
        namespace VARCHAR(255) NOT NULL,
        source_id TEXT NOT NULL,
        content_hash CHAR(64) NOT NULL,
        metadata JSONB DEFAULT '{{}}',
        status TEXT NOT NULL DEFAULT 'active',
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE (tenant_id, id),
        UNIQUE (tenant_id, namespace, source_id)
    )
    """
# Hash/chunk publication is atomic (see upsert_document + replace_chunks
# sharing one transaction in tenant_engine.py), but graph extraction is a
# separate, eventually-consistent stage that runs after publication — these
# columns are what let a caller (or retrieval) tell "graph-ready" apart from
# "text-searchable but extraction still pending/incomplete", instead of the
# two being indistinguishable as they were before.
_DOCUMENTS_EXTRACTION_STATUS_COLUMN_DDL = (
    f"ALTER TABLE {SCHEMA}.documents ADD COLUMN IF NOT EXISTS "
    "extraction_status TEXT NOT NULL DEFAULT 'pending'"
)
_DOCUMENTS_EXTRACTION_ERROR_COLUMN_DDL = (
    f"ALTER TABLE {SCHEMA}.documents ADD COLUMN IF NOT EXISTS extraction_error TEXT"
)
_DOCUMENTS_EXTRACTION_ATTEMPTS_COLUMN_DDL = (
    f"ALTER TABLE {SCHEMA}.documents ADD COLUMN IF NOT EXISTS "
    "extraction_attempts INT NOT NULL DEFAULT 0"
)
# Postgres has no `ADD CONSTRAINT IF NOT EXISTS`, so this drops and recreates
# the constraint every migration run (idempotent, matching the DROP
# TRIGGER IF EXISTS + CREATE TRIGGER pattern used below for the edge-support
# trigger) — application code already validates in set_extraction_status(),
# this is defense in depth against anything that writes the column directly.
_DOCUMENTS_EXTRACTION_STATUS_CHECK_DDL = f"""
    ALTER TABLE {SCHEMA}.documents DROP CONSTRAINT IF EXISTS documents_extraction_status_check;
    ALTER TABLE {SCHEMA}.documents ADD CONSTRAINT documents_extraction_status_check
        CHECK (extraction_status IN ('pending', 'ready', 'partial', 'failed'));
    """
_CHUNK_EXTRACTIONS_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.chunk_extractions (
        tenant_id UUID NOT NULL,
        content_hash CHAR(64) NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        result JSONB,
        triplet_count INT NOT NULL DEFAULT 0,
        lease_owner TEXT,
        lease_expires_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (tenant_id, content_hash, provider, model, prompt_version)
    )
    """
_GRAPH_EDGES_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.graph_edges (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id UUID NOT NULL,
        namespace VARCHAR(255) NOT NULL,
        source_node_id UUID NOT NULL,
        target_node_id UUID NOT NULL,
        relation TEXT NOT NULL,
        weight FLOAT DEFAULT 1.0,
        manual_weight FLOAT NOT NULL DEFAULT 0.0,
        support_count INT NOT NULL DEFAULT 0,
        metadata JSONB DEFAULT '{{}}',
        created_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE (tenant_id, namespace, source_node_id, target_node_id, relation),
        UNIQUE (tenant_id, id),
        FOREIGN KEY (tenant_id, source_node_id)
            REFERENCES {SCHEMA}.graph_nodes (tenant_id, id) ON DELETE CASCADE,
        FOREIGN KEY (tenant_id, target_node_id)
            REFERENCES {SCHEMA}.graph_nodes (tenant_id, id) ON DELETE CASCADE
    )
    """
_GRAPH_EDGES_MANUAL_WEIGHT_COLUMN_DDL = (
    f"ALTER TABLE {SCHEMA}.graph_edges ADD COLUMN IF NOT EXISTS manual_weight FLOAT NOT NULL DEFAULT 0.0"
)
_GRAPH_EDGES_SUPPORT_COUNT_COLUMN_DDL = (
    f"ALTER TABLE {SCHEMA}.graph_edges ADD COLUMN IF NOT EXISTS support_count INT NOT NULL DEFAULT 0"
)
_ENTITY_MENTIONS_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.entity_mentions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id UUID NOT NULL,
        chunk_id UUID NOT NULL,
        node_id UUID NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE (tenant_id, chunk_id, node_id),
        FOREIGN KEY (tenant_id, chunk_id)
            REFERENCES {SCHEMA}.document_chunks (tenant_id, id) ON DELETE CASCADE,
        FOREIGN KEY (tenant_id, node_id)
            REFERENCES {SCHEMA}.graph_nodes (tenant_id, id) ON DELETE CASCADE
    )
    """
_EDGE_MENTIONS_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.edge_mentions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id UUID NOT NULL,
        chunk_id UUID NOT NULL,
        edge_id UUID NOT NULL,
        confidence FLOAT,
        created_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE (tenant_id, chunk_id, edge_id),
        FOREIGN KEY (tenant_id, chunk_id)
            REFERENCES {SCHEMA}.document_chunks (tenant_id, id) ON DELETE CASCADE,
        FOREIGN KEY (tenant_id, edge_id)
            REFERENCES {SCHEMA}.graph_edges (tenant_id, id) ON DELETE CASCADE
    )
    """
_COMMUNITY_RUNS_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.community_runs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id UUID NOT NULL,
        namespace VARCHAR(255) NOT NULL,
        converged BOOLEAN NOT NULL,
        iterations INT NOT NULL,
        node_count INT NOT NULL,
        community_count INT NOT NULL,
        duration_ms INT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE (tenant_id, id)
    )
    """
_COMMUNITY_MEMBERSHIPS_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.community_memberships (
        tenant_id UUID NOT NULL,
        run_id UUID NOT NULL,
        node_id UUID NOT NULL,
        community_id UUID NOT NULL,
        PRIMARY KEY (tenant_id, run_id, node_id),
        FOREIGN KEY (tenant_id, run_id)
            REFERENCES {SCHEMA}.community_runs (tenant_id, id) ON DELETE CASCADE
    )
    """
_COMMUNITY_DIRTY_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.community_dirty (
        tenant_id UUID NOT NULL,
        namespace VARCHAR(255) NOT NULL,
        marked_dirty_at TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (tenant_id, namespace)
    )
    """
_COMMUNITY_SUMMARIES_TABLE_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.community_summaries (
        tenant_id UUID NOT NULL,
        run_id UUID NOT NULL,
        community_id UUID NOT NULL,
        summary TEXT NOT NULL,
        evidence_hash CHAR(64) NOT NULL,
        member_count INT NOT NULL,
        model TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (tenant_id, run_id, community_id),
        FOREIGN KEY (tenant_id, run_id)
            REFERENCES {SCHEMA}.community_runs (tenant_id, id) ON DELETE CASCADE
    )
    """

_INDEX_DDL_STATEMENTS = [
    f"CREATE INDEX IF NOT EXISTS idx_pgr_chunks_namespace ON {SCHEMA}.document_chunks (tenant_id, namespace)",
    f"CREATE INDEX IF NOT EXISTS idx_pgr_community_memberships_community ON {SCHEMA}.community_memberships (tenant_id, run_id, community_id)",
    f"CREATE INDEX IF NOT EXISTS idx_pgr_community_runs_namespace ON {SCHEMA}.community_runs (tenant_id, namespace, created_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_pgr_chunks_tsv ON {SCHEMA}.document_chunks USING gin (tsv)",
    f"CREATE INDEX IF NOT EXISTS idx_pgr_chunks_hash ON {SCHEMA}.document_chunks (tenant_id, content_hash)",
    f"CREATE INDEX IF NOT EXISTS idx_pgr_nodes_trgm ON {SCHEMA}.graph_nodes USING gin (content gin_trgm_ops)",
    f"CREATE INDEX IF NOT EXISTS idx_pgr_edges_source ON {SCHEMA}.graph_edges (tenant_id, source_node_id)",
    f"CREATE INDEX IF NOT EXISTS idx_pgr_edges_target ON {SCHEMA}.graph_edges (tenant_id, target_node_id)",
    # Needed for the composite FK from edge_mentions on databases created
    # before the explicit UNIQUE (tenant_id, id) was added to graph_edges.
    f"CREATE UNIQUE INDEX IF NOT EXISTS idx_pgr_edges_tenant_id_id ON {SCHEMA}.graph_edges (tenant_id, id)",
    f"CREATE INDEX IF NOT EXISTS idx_pgr_mentions_node ON {SCHEMA}.entity_mentions (tenant_id, node_id)",
    f"CREATE INDEX IF NOT EXISTS idx_pgr_edge_mentions_edge ON {SCHEMA}.edge_mentions (tenant_id, edge_id)",
]

_RECOMPUTE_EDGE_SUPPORT_TRIGGER_DDL = f"""
    CREATE OR REPLACE FUNCTION {SCHEMA}.recompute_edge_support()
    RETURNS trigger LANGUAGE plpgsql AS $func$
    DECLARE
        v_tenant UUID := COALESCE(NEW.tenant_id, OLD.tenant_id);
        v_edge UUID := COALESCE(NEW.edge_id, OLD.edge_id);
        v_support INT;
        v_manual FLOAT;
    BEGIN
        SELECT count(*)::int INTO v_support
        FROM {SCHEMA}.edge_mentions
        WHERE tenant_id = v_tenant AND edge_id = v_edge;

        SELECT manual_weight INTO v_manual
        FROM {SCHEMA}.graph_edges
        WHERE tenant_id = v_tenant AND id = v_edge
        FOR UPDATE;

        IF v_manual IS NULL THEN
            RETURN NULL;
        END IF;

        IF v_support = 0 AND v_manual <= 0 THEN
            DELETE FROM {SCHEMA}.graph_edges
            WHERE tenant_id = v_tenant AND id = v_edge;
        ELSE
            UPDATE {SCHEMA}.graph_edges
            SET support_count = v_support,
                weight = v_manual + v_support
            WHERE tenant_id = v_tenant AND id = v_edge;
        END IF;
        RETURN NULL;
    END;
    $func$;
    DROP TRIGGER IF EXISTS edge_mentions_recompute_support ON {SCHEMA}.edge_mentions;
    CREATE CONSTRAINT TRIGGER edge_mentions_recompute_support
    AFTER INSERT OR DELETE ON {SCHEMA}.edge_mentions
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.recompute_edge_support();
    """


def _document_chunks_ddl(vector_col: str) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.document_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id UUID NOT NULL,
            document_id UUID NOT NULL,
            namespace VARCHAR(255) NOT NULL,
            ordinal INT NOT NULL,
            content TEXT NOT NULL,
            content_hash CHAR(64) NOT NULL,
            embedding {vector_col},
            tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
            metadata JSONB DEFAULT '{{}}',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMPTZ DEFAULT now(),
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, document_id, ordinal),
            FOREIGN KEY (tenant_id, document_id)
                REFERENCES {SCHEMA}.documents (tenant_id, id) ON DELETE CASCADE
        )
        """


def _graph_nodes_ddl(vector_col: str) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.graph_nodes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id UUID NOT NULL,
            namespace VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            embedding {vector_col},
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMPTZ DEFAULT now(),
            UNIQUE (tenant_id, id),
            UNIQUE (tenant_id, namespace, content)
        )
        """


def _ann_index_ddl(ops_class: str) -> List[str]:
    return [
        f"CREATE INDEX IF NOT EXISTS idx_pgr_chunks_embedding ON {SCHEMA}.document_chunks USING hnsw (embedding {ops_class})",
        f"CREATE INDEX IF NOT EXISTS idx_pgr_nodes_embedding ON {SCHEMA}.graph_nodes USING hnsw (embedding {ops_class})",
    ]


_DDL_SHAPE_STATEMENTS: List[str] = [
    f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}",
    _SCHEMA_MIGRATIONS_TABLE_DDL,
    _SCHEMA_MIGRATIONS_CHECKSUM_COLUMN_DDL,
    _SCHEMA_SETTINGS_TABLE_DDL,
    _DOCUMENTS_TABLE_DDL,
    _DOCUMENTS_EXTRACTION_STATUS_COLUMN_DDL,
    _DOCUMENTS_EXTRACTION_ERROR_COLUMN_DDL,
    _DOCUMENTS_EXTRACTION_ATTEMPTS_COLUMN_DDL,
    _DOCUMENTS_EXTRACTION_STATUS_CHECK_DDL,
    _document_chunks_ddl(_VECTOR_COL_PLACEHOLDER),
    _CHUNK_EXTRACTIONS_TABLE_DDL,
    _graph_nodes_ddl(_VECTOR_COL_PLACEHOLDER),
    _GRAPH_EDGES_TABLE_DDL,
    _GRAPH_EDGES_MANUAL_WEIGHT_COLUMN_DDL,
    _GRAPH_EDGES_SUPPORT_COUNT_COLUMN_DDL,
    _ENTITY_MENTIONS_TABLE_DDL,
    _EDGE_MENTIONS_TABLE_DDL,
    _COMMUNITY_RUNS_TABLE_DDL,
    _COMMUNITY_MEMBERSHIPS_TABLE_DDL,
    _COMMUNITY_DIRTY_TABLE_DDL,
    _COMMUNITY_SUMMARIES_TABLE_DDL,
    *_INDEX_DDL_STATEMENTS,
    *_ann_index_ddl(_VECTOR_OPS_PLACEHOLDER),
    _RECOMPUTE_EDGE_SUPPORT_TRIGGER_DDL,
    *[_tenant_rls_policy(t) for t in _RLS_TABLES],
]

MIGRATION_DDL_CHECKSUM = hashlib.sha256(
    "\n".join(_DDL_SHAPE_STATEMENTS).encode("utf-8")
).hexdigest()

# Any fixed int identifies this advisory lock class; the specific value has
# no meaning beyond not colliding with other advisory locks this process
# might take.
_MIGRATION_LOCK_KEY = 837123


class SchemaCompatibilityError(RuntimeError):
    """The configured embedding schema cannot safely use existing data."""


class InsecureRuntimeRoleError(RuntimeError):
    """The connecting role has rolsuper or rolbypassrls set, so RLS would not apply to it."""


async def _assert_role_cannot_bypass_rls(conn: psycopg.AsyncConnection) -> None:
    """Raises InsecureRuntimeRoleError if the connecting role is a superuser
    or has BYPASSRLS — either one makes every RLS policy in this schema a
    no-op for that role, regardless of what the policies say. pg_roles is
    readable by any authenticated role (unlike pg_authid), so this needs no
    special privilege to check."""
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        )
        row = await cur.fetchone()
    # No matching pg_roles row should be impossible (current_user must be a
    # real role to have connected at all) — fail closed rather than assume
    # safety if this row is ever missing.
    if row is None or row["rolsuper"] or row["rolbypassrls"]:
        raise InsecureRuntimeRoleError(
            "SecureGraphStore must connect as a NOSUPERUSER/NOBYPASSRLS role; "
            f"current_user has rolsuper={row and row['rolsuper']!r}, "
            f"rolbypassrls={row and row['rolbypassrls']!r}. Pass the runtime "
            "role's URL created by migrate_schema(), never the admin/owner URL."
        )


async def migrate_schema(
    admin_url: str,
    runtime_role: str,
    runtime_password: str,
    embedding_dimension: int,
    migrate_legacy_data: bool = False,
) -> None:
    """Creates/upgrades the tenant-aware schema. Must be run with an
    admin/owner connection (e.g. the Postgres superuser), never with the
    restricted runtime role — this is a privileged operation, kept
    deliberately separate from `SecureGraphStore`'s tenant-scoped runtime
    queries.

    Idempotent: safe to re-run (CREATE ... IF NOT EXISTS / DROP POLICY IF
    EXISTS + CREATE POLICY throughout). Takes a session-level advisory lock
    for its duration and fails fast (rather than blocking) if another
    migration is already in progress against the same database — concurrent
    DDL (two deployments migrating at once) is a real production hazard,
    not just a theoretical one.
    """
    if not isinstance(embedding_dimension, int) or not (0 < embedding_dimension <= 16000):
        raise ValueError(f"Invalid embedding_dimension: {embedding_dimension!r}")

    vector_type = _vector_column_type(embedding_dimension)
    use_ann_index = embedding_dimension <= _HNSW_HALFVEC_MAX_DIM

    conn = await psycopg.AsyncConnection.connect(admin_url, row_factory=dict_row)
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT pg_try_advisory_lock(%s)", (_MIGRATION_LOCK_KEY,))
            got_lock = (await cur.fetchone())["pg_try_advisory_lock"]
            if not got_lock:
                raise RuntimeError(
                    "Another migrate_schema() run holds the migration advisory lock "
                    f"({_MIGRATION_LOCK_KEY}) on this database; refusing to run concurrently."
                )
        try:
            try:
                await _migrate_schema_locked(
                    conn, runtime_role, runtime_password, embedding_dimension,
                    vector_type, use_ann_index, migrate_legacy_data,
                )
            except Exception:
                # DDL errors abort the transaction; rollback before trying to
                # release the advisory lock so the original exception is not
                # masked by InFailedSqlTransaction.
                await conn.rollback()
                raise
        finally:
            async with conn.cursor() as cur:
                await cur.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_KEY,))
    finally:
        await conn.close()


async def _migrate_schema_locked(
    conn: psycopg.AsyncConnection,
    runtime_role: str,
    runtime_password: str,
    embedding_dimension: int,
    vector_type: str,
    use_ann_index: bool,
    migrate_legacy_data: bool,
) -> None:
    async with conn.cursor() as cur:
        # Keep extension objects out of a tenant/temp search_path.  The
        # unqualified vector/pg_trgm types and operators used below then
        # resolve consistently for both admin migrations and runtime pools.
        await cur.execute("SET LOCAL search_path = public, pg_catalog")
        await cur.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        await cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public")
        await cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

        # CREATE TABLE IF NOT EXISTS cannot change a pgvector dimension.
        # Detect incompatibility before any domain-table DDL/write instead of
        # reporting a successful setup followed by a later DataException.
        await cur.execute(
            "SELECT to_regclass(%s) AS table_name",
            (f"{SCHEMA}.graph_nodes",),
        )
        existing_nodes = await cur.fetchone()
        if existing_nodes and existing_nodes["table_name"]:
            await cur.execute(
                f"""
                SELECT format_type(atttypid, atttypmod) AS full_type
                FROM pg_attribute
                WHERE attrelid = '{SCHEMA}.graph_nodes'::regclass
                  AND attname = 'embedding' AND NOT attisdropped
                """
            )
            type_row = await cur.fetchone()
            expected_type = f"{vector_type}({embedding_dimension})"
            actual_type = type_row["full_type"] if type_row else None
            if actual_type != expected_type:
                raise SchemaCompatibilityError(
                    "Secure schema embedding type is incompatible: "
                    f"database={actual_type!r}, configured={expected_type!r}. "
                    "Create a new schema and re-embed the source documents; "
                    "vector dimensions cannot be changed safely in place."
                )

        await cur.execute(_SCHEMA_MIGRATIONS_TABLE_DDL)
        await cur.execute(_SCHEMA_MIGRATIONS_CHECKSUM_COLUMN_DDL)
        await cur.execute(_SCHEMA_SETTINGS_TABLE_DDL)
        await cur.execute(
            f"SELECT checksum FROM {SCHEMA}.schema_migrations WHERE version=%s",
            (SCHEMA_VERSION,),
        )
        existing_migration = await cur.fetchone()
        if (
            existing_migration
            and existing_migration["checksum"]
            and existing_migration["checksum"] != MIGRATION_DDL_CHECKSUM
        ):
            raise SchemaCompatibilityError(
                f"Migration {SCHEMA_VERSION} checksum differs from the applied migration; "
                "never edit an applied migration in place."
            )

        await cur.execute(_DOCUMENTS_TABLE_DDL)
        await cur.execute(_DOCUMENTS_EXTRACTION_STATUS_COLUMN_DDL)
        await cur.execute(_DOCUMENTS_EXTRACTION_ERROR_COLUMN_DDL)
        await cur.execute(_DOCUMENTS_EXTRACTION_ATTEMPTS_COLUMN_DDL)
        await cur.execute(_DOCUMENTS_EXTRACTION_STATUS_CHECK_DDL)

        await cur.execute(_document_chunks_ddl(f"{vector_type}({embedding_dimension})"))

        # Lease-based extraction cache: concurrent ingestion of the same
        # chunk content (even across documents/tenant-namespaces sharing
        # a hash) claims a lease before calling the LLM, so only one
        # worker pays for the request; others poll `status`/
        # `lease_expires_at` instead of duplicating the call.
        await cur.execute(_CHUNK_EXTRACTIONS_TABLE_DDL)

        await cur.execute(_graph_nodes_ddl(f"{vector_type}({embedding_dimension})"))

        await cur.execute(_GRAPH_EDGES_TABLE_DDL)
        # `CREATE TABLE IF NOT EXISTS` does not upgrade a v2 installation;
        # add the lifecycle columns explicitly before backfilling them.
        await cur.execute(_GRAPH_EDGES_MANUAL_WEIGHT_COLUMN_DDL)
        await cur.execute(_GRAPH_EDGES_SUPPORT_COUNT_COLUMN_DDL)

        # Chunk -> entity evidence. This is the provenance link the
        # single-tenant schema (database.py) is missing: given a node,
        # you can trace every chunk (and therefore document) that
        # mentioned it. Deleting a document cascades to its chunks,
        # which cascades to mentions, but NOT to the entity/edge itself
        # (other documents may still support it) — mirroring the plan's
        # "delete only mentions, keep entities that are still supported".
        await cur.execute(_ENTITY_MENTIONS_TABLE_DDL)

        # Chunk -> relationship evidence. Unlike entity_mentions, this keeps
        # the exact edge asserted by a chunk, so two documents can support
        # different (including conflicting) relationships without losing
        # their individual provenance.
        await cur.execute(_EDGE_MENTIONS_TABLE_DDL)

        # Community detection (weighted label propagation) run history +
        # membership snapshots. Runs are immutable once written — a
        # namespace's *current* communities are simply the memberships
        # belonging to its most recent run_id.
        await cur.execute(_COMMUNITY_RUNS_TABLE_DDL)
        await cur.execute(_COMMUNITY_MEMBERSHIPS_TABLE_DDL)
        # Presence of a row marks a (tenant, namespace) as needing
        # recomputation; refresh_communities() deletes it on success.
        await cur.execute(_COMMUNITY_DIRTY_TABLE_DDL)
        await cur.execute(_COMMUNITY_SUMMARIES_TABLE_DDL)

        # Indexes
        for index_ddl in _INDEX_DDL_STATEMENTS:
            await cur.execute(index_ddl)

        # Existing v2 edges used `weight` as an ingestion counter.  Convert
        # evidence-backed edges to the new support-derived representation and
        # preserve evidence-free legacy edges as manual/legacy weight so a
        # forward migration never silently deletes knowledge.
        await cur.execute(
            f"""
            UPDATE {SCHEMA}.graph_edges e
            SET support_count = s.support_count,
                manual_weight = CASE WHEN s.support_count > 0 THEN 0.0 ELSE e.weight END,
                weight = CASE WHEN s.support_count > 0 THEN s.support_count::float ELSE e.weight END
            FROM (
                SELECT tenant_id, edge_id, count(*)::int AS support_count
                FROM {SCHEMA}.edge_mentions
                GROUP BY tenant_id, edge_id
            ) s
            WHERE e.tenant_id = s.tenant_id AND e.id = s.edge_id
            """
        )

        # Keep edge weight correct when document chunks cascade-delete their
        # edge evidence.  Unsupported extracted edges are removed; explicit
        # deterministic edges survive through manual_weight.
        await cur.execute(_RECOMPUTE_EDGE_SUPPORT_TRIGGER_DDL)
        if use_ann_index:
            ops_class = "vector_cosine_ops" if vector_type == "vector" else "halfvec_cosine_ops"
            for index_ddl in _ann_index_ddl(ops_class):
                await cur.execute(index_ddl)
        else:
            logger.warning(
                "embedding_dimension=%d exceeds pgvector's HNSW limit; "
                "secure-schema vector search will use exact scans.",
                embedding_dimension,
            )

        # RLS: fail-closed tenant isolation on every domain table.
        for table in _RLS_TABLES:
            await cur.execute(_tenant_rls_policy(table))

        # Runtime role: NOSUPERUSER + NOBYPASSRLS is what makes RLS
        # actually apply to it. It must not own these tables (ownership
        # itself doesn't bypass RLS once FORCE is set, but keeping
        # ownership with the admin role keeps privilege escalation paths
        # like ALTER TABLE out of the runtime role's reach entirely).
        await cur.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (runtime_role,)
        )
        role_exists = await cur.fetchone()
        # CREATE/ALTER ROLE's PASSWORD clause is DDL and cannot take a
        # bind parameter; sql.Literal safely quotes it (escaping any
        # embedded quotes) instead of relying on the query protocol.
        role_stmt = sql.SQL(
            "{verb} ROLE {role} WITH LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD {password}"
        ).format(
            verb=sql.SQL("ALTER" if role_exists else "CREATE"),
            role=sql.Identifier(runtime_role),
            password=sql.Literal(runtime_password),
        )
        await cur.execute(role_stmt)

        await cur.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(SCHEMA), sql.Identifier(runtime_role)
            )
        )
        await cur.execute(
            sql.SQL(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {} TO {}"
            ).format(sql.Identifier(SCHEMA), sql.Identifier(runtime_role))
        )

        if migrate_legacy_data:
            await _migrate_legacy_data(cur, vector_type, embedding_dimension)

        await cur.execute(
            f"""
            INSERT INTO {SCHEMA}.schema_migrations (version, description, checksum)
            VALUES (%s, %s, %s)
            ON CONFLICT (version) DO UPDATE SET
                description = EXCLUDED.description,
                checksum = COALESCE({SCHEMA}.schema_migrations.checksum, EXCLUDED.checksum)
            """,
            (SCHEMA_VERSION, MIGRATION_DESCRIPTION, MIGRATION_DDL_CHECKSUM),
        )
        await cur.execute(
            f"""
            INSERT INTO {SCHEMA}.schema_settings
                (singleton, embedding_dimension, vector_type, schema_version)
            VALUES (true, %s, %s, %s)
            ON CONFLICT (singleton) DO UPDATE SET
                embedding_dimension = EXCLUDED.embedding_dimension,
                vector_type = EXCLUDED.vector_type,
                schema_version = EXCLUDED.schema_version,
                updated_at = now()
            """,
            (embedding_dimension, vector_type, SCHEMA_VERSION),
        )

        await conn.commit()


async def _migrate_legacy_data(cur, vector_type: str, embedding_dimension: int) -> None:
    """Backfills pre-multi-tenancy `public.graph_nodes`/`graph_edges` rows
    (from database.py's single-tenant schema) into the tenant-aware schema
    under LEGACY_TENANT_ID, preserving IDs. Idempotent via ON CONFLICT DO
    NOTHING keyed on the preserved id. Existing data becomes graph-only —
    there is no source document to backfill a chunk/mention for, since the
    legacy schema never recorded provenance."""
    await cur.execute(
        "SELECT to_regclass('public.graph_nodes') AS t"
    )
    row = await cur.fetchone()
    if not row or not row["t"]:
        logger.info("No legacy public.graph_nodes table found; skipping data migration.")
        return

    await cur.execute(
        """
        SELECT format_type(atttypid, atttypmod) AS full_type
        FROM pg_attribute
        WHERE attrelid = 'public.graph_nodes'::regclass AND attname = 'embedding'
        """
    )
    legacy_type_row = await cur.fetchone()
    expected = f"{vector_type}({embedding_dimension})"
    if legacy_type_row and legacy_type_row["full_type"] != expected:
        raise ValueError(
            f"Legacy public.graph_nodes.embedding is {legacy_type_row['full_type']!r}, "
            f"but the secure schema expects {expected!r}. Migrate with a matching "
            "embedding_dimension, or migrate_legacy_data=False to skip."
        )

    await cur.execute(
        f"""
        INSERT INTO {SCHEMA}.graph_nodes (id, tenant_id, namespace, content, embedding, metadata, created_at)
        SELECT id, %s, namespace, content, embedding, metadata, created_at
        FROM public.graph_nodes
        ON CONFLICT (tenant_id, id) DO NOTHING
        """,
        (str(LEGACY_TENANT_ID),),
    )
    # manual_weight is set from the legacy weight (not left at its default
    # 0.0): a migrated edge has no document evidence at all (the legacy
    # schema never recorded provenance), so support_count correctly stays
    # 0 -- but leaving manual_weight at 0 too, with only `weight` carrying
    # the old value, violates the weight = manual_weight + support_count
    # invariant every other write path maintains. Worse, it silently drops
    # the edge out of traversal entirely: the read-time defense-in-depth
    # filter added for the unsupported-edge leak (support_count > 0 OR
    # manual_weight > 0) would treat every migrated edge as unsupported,
    # since a bare `weight` column satisfying `weight >= min_weight` alone
    # doesn't pass that check. Copying weight into manual_weight is exactly
    # how a deterministic, non-evidence-backed edge is represented
    # elsewhere (see upsert_edges' evidence_backed=False branch) --
    # migrated legacy edges are precisely that: manually-asserted,
    # unsupported-by-any-chunk facts.
    await cur.execute(
        f"""
        INSERT INTO {SCHEMA}.graph_edges
            (id, tenant_id, namespace, source_node_id, target_node_id, relation, weight, manual_weight, metadata, created_at)
        SELECT id, %s, namespace, source_node_id, target_node_id, relation, weight, weight, metadata, created_at
        FROM public.graph_edges
        ON CONFLICT (tenant_id, namespace, source_node_id, target_node_id, relation) DO NOTHING
        """,
        (str(LEGACY_TENANT_ID),),
    )
    logger.info("Legacy public.graph_nodes/graph_edges backfilled under tenant %s.", LEGACY_TENANT_ID)


class SecureGraphStore:
    """Tenant-isolated runtime storage layer. Must be constructed with the
    restricted runtime role's connection URL (created by `migrate_schema`),
    never the admin/owner URL — RLS is a no-op for a superuser or any role
    with BYPASSRLS."""

    def __init__(
        self,
        runtime_url: str,
        vector_type: str = "vector",
        pool_min_size: int = 1,
        pool_max_size: int = 10,
        pool_timeout_s: float = 30.0,
        statement_timeout_ms: int = 30_000,
    ):
        self.runtime_url = runtime_url
        # Must match whatever migrate_schema() actually created the
        # embedding columns as (see _vector_column_type) — the caller knows
        # the embedding dimension and is responsible for keeping this in
        # sync; SecureGraphStore has no schema-introspection step of its own.
        self.vector_type = vector_type
        self.pool: Optional[AsyncConnectionPool] = None
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool_timeout_s = pool_timeout_s
        self._statement_timeout_ms = statement_timeout_ms

    async def _init_pool(self):
        if self.pool is None:
            # Fail fast, before ever constructing a pool of (possibly
            # insecure) connections: a plain one-off connection, independent
            # of the pool's configure/retry machinery, so a bad role raises
            # InsecureRuntimeRoleError immediately and cleanly rather than
            # being swallowed by the pool's background reconnect-with-backoff
            # loop (psycopg_pool's _add_connection catches and retries *any*
            # exception raised from `configure` rather than propagating it).
            probe = await psycopg.AsyncConnection.connect(self.runtime_url, row_factory=dict_row)
            try:
                await _assert_role_cannot_bypass_rls(probe)
            finally:
                await probe.close()

            async def _configure(conn: psycopg.AsyncConnection):
                # Defense in depth: re-checked on every new pooled
                # connection, not just the one probed above, in case the
                # pool grows later or role privileges change at runtime. If
                # this check ever fails here, psycopg_pool retries
                # connecting with backoff rather than propagating the
                # exception out of pool.open()/pool.connection() — callers
                # will observe repeated warning logs and, eventually,
                # PoolTimeout on checkout, not a silent RLS bypass.
                await _assert_role_cannot_bypass_rls(conn)
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"SET statement_timeout = {int(self._statement_timeout_ms)}"
                    )
                await conn.commit()

            self.pool = AsyncConnectionPool(
                self.runtime_url,
                open=False,
                kwargs={"row_factory": dict_row},
                min_size=self._pool_min_size,
                max_size=self._pool_max_size,
                timeout=self._pool_timeout_s,
                configure=_configure,
            )
            await self.pool.open()

    async def close(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def schema_status(self) -> Dict[str, Any]:
        """Read compatibility metadata without requiring tenant context."""
        await self._init_pool()
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT embedding_dimension, vector_type, schema_version, updated_at "
                    f"FROM {SCHEMA}.schema_settings WHERE singleton=true"
                )
                row = await cur.fetchone()
        if not row:
            return {"ready": False, "reason": "schema_settings_missing"}
        return {
            "ready": True,
            "embedding_dimension": row["embedding_dimension"],
            "vector_type": row["vector_type"],
            "schema_version": row["schema_version"],
            "updated_at": row["updated_at"].isoformat(),
        }

    @asynccontextmanager
    async def tenant_connection(
        self, tenant_id: uuid.UUID
    ) -> AsyncIterator[psycopg.AsyncConnection]:
        """Yields a connection with the tenant GUC set *for the duration of
        one transaction only*. Commits on clean exit, rolls back on
        exception — either way the transaction ends before the connection
        goes back to the pool, so the transaction-local `set_config` value
        can never leak into whatever tenant borrows this physical connection
        next."""
        await self._init_pool()
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT set_config('{TENANT_GUC}', %s, true)", (str(tenant_id),)
                )
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Lease-based extraction cache
    # ------------------------------------------------------------------

    async def claim_extraction(
        self,
        tenant_id: uuid.UUID,
        chunk_hash: str,
        provider: str,
        model: str,
        prompt_version: str,
        lease_owner: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> Dict[str, Any]:
        """Attempts to claim the right to run extraction for this exact
        content hash. Returns one of:

          {"status": "claimed"} — caller must run extraction and call
              complete_extraction() (or fail_extraction() on error).
          {"status": "done", "result": [...], "triplet_count": N} — another
              worker already finished this; use the cached result.
          {"status": "in_progress"} — another worker holds a live lease;
              caller should back off and poll again, not duplicate the call.

        This is safe under concurrent claimants: the UPDATE's WHERE clause
        only succeeds for one racing transaction (Postgres row-level
        locking on the conflicting row serializes the competing UPDATEs),
        so exactly one caller gets {"status": "claimed"}.
        """
        params = {
            "tenant_id": str(tenant_id),
            "content_hash": chunk_hash,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "lease_owner": lease_owner,
            "lease_seconds": lease_seconds,
        }

        async def _run(conn: psycopg.AsyncConnection) -> Dict[str, Any]:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO {SCHEMA}.chunk_extractions
                        (tenant_id, content_hash, provider, model, prompt_version,
                         status, lease_owner, lease_expires_at)
                    VALUES
                        (%(tenant_id)s, %(content_hash)s, %(provider)s, %(model)s, %(prompt_version)s,
                         'in_progress', %(lease_owner)s, now() + make_interval(secs => %(lease_seconds)s))
                    ON CONFLICT (tenant_id, content_hash, provider, model, prompt_version)
                    DO UPDATE SET
                        status = 'in_progress',
                        lease_owner = EXCLUDED.lease_owner,
                        lease_expires_at = EXCLUDED.lease_expires_at,
                        updated_at = now()
                    WHERE {SCHEMA}.chunk_extractions.status <> 'done'
                      AND ({SCHEMA}.chunk_extractions.lease_expires_at IS NULL
                           OR {SCHEMA}.chunk_extractions.lease_expires_at < now())
                    RETURNING 1
                    """,
                    params,
                )
                claimed = await cur.fetchone()
                if claimed:
                    return {"status": "claimed"}

                await cur.execute(
                    f"""
                    SELECT status, result, triplet_count FROM {SCHEMA}.chunk_extractions
                    WHERE tenant_id = %(tenant_id)s AND content_hash = %(content_hash)s
                      AND provider = %(provider)s AND model = %(model)s
                      AND prompt_version = %(prompt_version)s
                    """,
                    params,
                )
                existing = await cur.fetchone()

            if existing and existing["status"] == "done":
                return {
                    "status": "done",
                    "result": existing["result"],
                    "triplet_count": existing["triplet_count"],
                }
            return {"status": "in_progress"}

        if connection:
            return await _run(connection)
        await self._init_pool()
        async with self.tenant_connection(tenant_id) as conn:
            return await _run(conn)

    async def complete_extraction(
        self,
        tenant_id: uuid.UUID,
        chunk_hash: str,
        provider: str,
        model: str,
        prompt_version: str,
        lease_owner: str,
        result: List[Dict[str, Any]],
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> None:
        """Marks a claimed extraction as done, caching `result` for future
        callers. Guarded by `lease_owner` matching, so a lease that expired
        and was reclaimed by someone else can't have its result clobbered by
        the original (slow/stale) holder finishing late."""
        params = (
            json.dumps(result),
            len(result),
            str(tenant_id),
            chunk_hash,
            provider,
            model,
            prompt_version,
            lease_owner,
        )

        async def _run(conn: psycopg.AsyncConnection):
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE {SCHEMA}.chunk_extractions SET
                        status = 'done', result = %s, triplet_count = %s,
                        lease_owner = NULL, lease_expires_at = NULL, updated_at = now()
                    WHERE tenant_id = %s AND content_hash = %s AND provider = %s
                      AND model = %s AND prompt_version = %s AND lease_owner = %s
                    """,
                    params,
                )

        if connection:
            await _run(connection)
            return
        async with self.tenant_connection(tenant_id) as conn:
            await _run(conn)

    async def fail_extraction(
        self,
        tenant_id: uuid.UUID,
        chunk_hash: str,
        provider: str,
        model: str,
        prompt_version: str,
        lease_owner: str,
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> None:
        """Releases a claimed lease after an extraction failure, so the next
        attempt (this call's retry, or another worker) can claim it
        immediately instead of waiting out the full lease timeout."""
        params = (str(tenant_id), chunk_hash, provider, model, prompt_version, lease_owner)

        async def _run(conn: psycopg.AsyncConnection):
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE {SCHEMA}.chunk_extractions SET
                        status = 'pending', lease_owner = NULL, lease_expires_at = NULL, updated_at = now()
                    WHERE tenant_id = %s AND content_hash = %s AND provider = %s
                      AND model = %s AND prompt_version = %s AND lease_owner = %s
                    """,
                    params,
                )

        if connection:
            await _run(connection)
            return
        async with self.tenant_connection(tenant_id) as conn:
            await _run(conn)

    # ------------------------------------------------------------------
    # Documents / chunks (evidence-grounded ingestion)
    # ------------------------------------------------------------------

    async def lock_document_for_publication(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        source_id: str,
        connection: psycopg.AsyncConnection,
    ) -> None:
        """Transaction-scoped advisory lock (auto-released at commit/
        rollback via pg_advisory_xact_lock) keyed to this exact (tenant,
        namespace, source_id). Must be called with the *same* connection
        the caller then uses for upsert_document()/replace_chunks(), before
        either of them, to serialize concurrent publishers of the same
        logical document.

        This matters even though (tenant_id, namespace, source_id) already
        has a UNIQUE constraint: upsert_document()'s `content_changed` check
        reads the row's *previous* hash via a CTE that (like any read under
        READ COMMITTED) is evaluated against the snapshot at that
        statement's start. If two INSERT ... ON CONFLICT statements race on
        a brand-new document, the second one blocks on the row lock, but —
        having already taken its snapshot before the first transaction
        committed — its CTE still sees "no previous row" once unblocked,
        so it *also* reports content_changed=True for identical content
        (verified against a real concurrent test, not hypothetical). The
        advisory lock forces the second publisher to wait for the first to
        fully commit before it even starts its own statement, so its read
        is fresh rather than stale.
        """
        # hashtextextended's 64-bit output fits pg_advisory_xact_lock's
        # bigint key directly; a hash collision between two different
        # documents only costs extra (harmless) serialization, never
        # incorrectness, since the lock is purely a concurrency gate.
        key = f"{tenant_id}:{namespace}:{source_id}"
        async with connection.cursor() as cur:
            await cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (key,))

    async def upsert_document(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        source_id: str,
        doc_content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> str:
        """Upserts a document by (tenant, namespace, source_id). Returns the
        document id and whether its content actually changed since the last
        ingestion, via the row's content_hash — callers use this to decide
        whether to replace chunks (see `replace_chunks`).

        Callers publishing a document (as opposed to test setup poking the
        store directly) should call `lock_document_for_publication()` on the
        same connection first — see its docstring for why the UNIQUE
        constraint alone isn't sufficient under concurrency."""
        params = (str(tenant_id), namespace, source_id, str(tenant_id), namespace, source_id, doc_content_hash, json.dumps(metadata or {}))

        async def _run(conn: psycopg.AsyncConnection) -> Dict[str, Any]:
            async with conn.cursor() as cur:
                # `old` is evaluated against the pre-statement snapshot (the
                # same-statement-CTE rule), so it correctly captures the row
                # as it was *before* this upsert overwrites content_hash —
                # including the "no previous row" case, where a NULL
                # comparison via IS DISTINCT FROM correctly reports a new
                # document as content_changed=True.
                await cur.execute(
                    f"""
                    WITH old AS (
                        SELECT content_hash FROM {SCHEMA}.documents
                        WHERE tenant_id = %s AND namespace = %s AND source_id = %s
                    )
                    INSERT INTO {SCHEMA}.documents (tenant_id, namespace, source_id, content_hash, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, namespace, source_id) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash,
                        metadata = {SCHEMA}.documents.metadata || EXCLUDED.metadata,
                        updated_at = now(),
                        -- A changed hash means new chunks are about to be
                        -- published and re-extracted; any extraction_status
                        -- left over from the *previous* content no longer
                        -- describes anything real, so reset it here rather
                        -- than let a stale 'ready'/'failed' outlive its hash.
                        extraction_status = CASE
                            WHEN {SCHEMA}.documents.content_hash IS DISTINCT FROM EXCLUDED.content_hash
                            THEN 'pending' ELSE {SCHEMA}.documents.extraction_status END,
                        extraction_error = CASE
                            WHEN {SCHEMA}.documents.content_hash IS DISTINCT FROM EXCLUDED.content_hash
                            THEN NULL ELSE {SCHEMA}.documents.extraction_error END
                    RETURNING id,
                        (SELECT content_hash FROM old) IS DISTINCT FROM content_hash AS content_changed
                    """,
                    params,
                )
                row = await cur.fetchone()
            return {"id": str(row["id"]), "content_changed": row["content_changed"]}

        if connection:
            return await _run(connection)
        async with self.tenant_connection(tenant_id) as conn:
            result = await _run(conn)
            return result

    async def get_document_content_hash(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        source_id: str,
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> Optional[str]:
        params = (str(tenant_id), namespace, source_id)

        async def _run(conn: psycopg.AsyncConnection) -> Optional[str]:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT content_hash FROM {SCHEMA}.documents WHERE tenant_id=%s AND namespace=%s AND source_id=%s",
                    params,
                )
                row = await cur.fetchone()
            return row["content_hash"] if row else None

        if connection:
            return await _run(connection)
        async with self.tenant_connection(tenant_id) as conn:
            return await _run(conn)

    async def set_extraction_status(
        self,
        tenant_id: uuid.UUID,
        document_id: str,
        status: str,
        error: Optional[str] = None,
        increment_attempts: bool = False,
        expected_content_hash: Optional[str] = None,
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> bool:
        """Records how far graph extraction got for one document, *after*
        its hash/chunks were already published atomically. Extraction itself
        remains a separate, eventually-consistent stage (LLM calls are not
        run inside a database transaction) — this is what lets a caller
        distinguish "text-searchable, graph pending/partial" from
        "graph-ready" instead of the two looking identical from outside.

        status: one of 'pending' | 'ready' | 'partial' | 'failed'.

        `expected_content_hash`, when given, makes this a compare-and-set:
        the update only applies if the document's *current* content_hash
        still matches it. Without this, a worker whose extraction is still
        running against revision B's chunks when a concurrent revision C is
        published (resetting status to 'pending' for C) could finish late
        and overwrite C's status with a verdict that was only ever true of
        B — even though filter_existing_chunk_ids() already stopped it from
        attaching B's facts to C's chunks, the status row itself has no
        such guard unless this is passed. Returns whether the row was
        actually updated (False means a caller-supplied expected hash no
        longer matched — i.e. this worker was superseded).
        """
        if status not in _VALID_EXTRACTION_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_EXTRACTION_STATUSES)}, got {status!r}"
            )
        params: List[Any] = [status, error, 1 if increment_attempts else 0, str(tenant_id), document_id]
        hash_clause = ""
        if expected_content_hash is not None:
            hash_clause = "AND content_hash = %s"
            params.append(expected_content_hash)

        async def _run(conn: psycopg.AsyncConnection) -> bool:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    UPDATE {SCHEMA}.documents
                    SET extraction_status = %s,
                        extraction_error = %s,
                        extraction_attempts = extraction_attempts + %s,
                        updated_at = now()
                    WHERE tenant_id = %s AND id = %s {hash_clause}
                    RETURNING id
                    """,
                    params,
                )
                row = await cur.fetchone()
            return row is not None

        if connection:
            return await _run(connection)
        async with self.tenant_connection(tenant_id) as conn:
            return await _run(conn)

    async def filter_existing_chunk_ids(
        self,
        tenant_id: uuid.UUID,
        chunk_ids: List[str],
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> List[str]:
        """Returns the subset of `chunk_ids` that still exist in
        `document_chunks`. Used to guard against the stale-extraction race:
        a chunk's extraction can still be in flight (an LLM call takes
        seconds) after a *concurrent* re-ingestion has already replaced that
        document's chunks (`replace_chunks` deletes the old rows outright),
        so the late-finishing worker must not attach entity/edge mentions to
        chunk ids that no longer belong to the current revision."""
        if not chunk_ids:
            return []

        async def _run(conn: psycopg.AsyncConnection) -> List[str]:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT id FROM {SCHEMA}.document_chunks WHERE tenant_id = %s AND id = ANY(%s)",
                    (str(tenant_id), chunk_ids),
                )
                rows = await cur.fetchall()
            return [str(r["id"]) for r in rows]

        if connection:
            return await _run(connection)
        async with self.tenant_connection(tenant_id) as conn:
            return await _run(conn)

    async def prune_unsupported_edges(
        self,
        tenant_id: uuid.UUID,
        edge_ids: List[str],
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> None:
        """Deletes any of `edge_ids` that ended up with zero support: an
        evidence-backed edge is created by `upsert_edges` *before* its
        `edge_mentions` row is recorded, so a failure in between (e.g. the
        stale-chunk race `filter_existing_chunk_ids` guards against, or any
        other error after edge creation but before mention recording) can
        otherwise leave a zero-weight, zero-mention edge permanently
        traversable — `graph_edges.weight` defaults to `0.0` and traversal's
        `min_weight` defaults to `0.0` too, so it wouldn't be filtered out by
        default. Mirrors the condition `recompute_edge_support()` already
        uses to delete edges when their last mention is removed; this just
        also covers edges that never got a first mention at all."""
        if not edge_ids:
            return

        async def _run(conn: psycopg.AsyncConnection) -> None:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    DELETE FROM {SCHEMA}.graph_edges
                    WHERE tenant_id = %s AND id = ANY(%s)
                      AND support_count = 0 AND manual_weight <= 0
                    """,
                    (str(tenant_id), edge_ids),
                )

        if connection:
            await _run(connection)
            return
        async with self.tenant_connection(tenant_id) as conn:
            await _run(conn)

    async def replace_chunks(
        self,
        tenant_id: uuid.UUID,
        document_id: str,
        namespace: str,
        chunks: List[Dict[str, Any]],
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> List[str]:
        """Atomically replaces all chunks for a document: deletes the old
        set (cascading to entity_mentions) and inserts the new one. This is
        the "document update replaces stale chunk occurrences
        transactionally" behavior — simpler than diffing old vs new chunks,
        at the cost of re-writing unchanged chunks too on a partial edit."""

        async def _run(conn: psycopg.AsyncConnection) -> List[str]:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {SCHEMA}.document_chunks WHERE tenant_id=%s AND document_id=%s",
                    (str(tenant_id), document_id),
                )
                ids: List[str] = []
                for start in range(0, len(chunks), MAX_ROWS_PER_STATEMENT):
                    batch = chunks[start : start + MAX_ROWS_PER_STATEMENT]
                    values_sql = ", ".join(["(%s,%s,%s,%s,%s,%s,%s,%s)"] * len(batch))
                    params: List[Any] = []
                    for i, c in enumerate(batch):
                        params.extend(
                            [
                                str(tenant_id),
                                document_id,
                                namespace,
                                start + i,
                                c["content"],
                                content_hash(c["content"]),
                                c["embedding"],
                                json.dumps(c.get("metadata") or {}),
                            ]
                        )
                    await cur.execute(
                        f"""
                        INSERT INTO {SCHEMA}.document_chunks
                            (tenant_id, document_id, namespace, ordinal, content, content_hash, embedding, metadata)
                        VALUES {values_sql}
                        RETURNING id
                        """,
                        params,
                    )
                    ids.extend(str(r["id"]) for r in await cur.fetchall())
            return ids

        if connection:
            return await _run(connection)
        async with self.tenant_connection(tenant_id) as conn:
            return await _run(conn)

    async def delete_document(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        source_id: str,
    ) -> None:
        """Deletes a document and its chunks/mentions (cascade). Entities
        and edges are intentionally NOT deleted — they may still be
        supported by other documents. Fully-orphaned nodes/edges (no
        remaining mentions from any document) are left in place rather than
        proactively pruned; a periodic sweep for zero-mention nodes is a
        reasonable follow-up if storage growth from orphans matters, but
        pruning inline here risks deleting a node whose only *other*
        evidence is being written concurrently in a still-open transaction."""
        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"DELETE FROM {SCHEMA}.documents WHERE tenant_id=%s AND namespace=%s AND source_id=%s",
                    (str(tenant_id), namespace, source_id),
                )

    async def find_unfinished_chunks(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        source_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Finds chunks that are durably stored but whose extraction never
        completed (no matching 'done' row in `chunk_extractions` for their
        content hash) — the retry candidate set for `retry_failed_chunks()`.

        When provider/model/prompt are supplied, only a matching successful
        extraction satisfies the retry check. This prevents a model upgrade
        from silently reusing an older provider's result.
        """
        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT dc.id, dc.content, dc.metadata, dc.document_id, d.source_id, d.content_hash
                    FROM {SCHEMA}.document_chunks dc
                    JOIN {SCHEMA}.documents d ON d.tenant_id = dc.tenant_id AND d.id = dc.document_id
                    WHERE dc.tenant_id = %(tenant_id)s AND dc.namespace = %(namespace)s
                      AND (%(source_id)s::text IS NULL OR d.source_id = %(source_id)s)
                      AND NOT EXISTS (
                          SELECT 1 FROM {SCHEMA}.chunk_extractions ce
                          WHERE ce.tenant_id = dc.tenant_id AND ce.content_hash = dc.content_hash AND ce.status = 'done'
                            AND (%(provider)s::text IS NULL OR ce.provider = %(provider)s)
                            AND (%(model)s::text IS NULL OR ce.model = %(model)s)
                            AND (%(prompt_version)s::text IS NULL OR ce.prompt_version = %(prompt_version)s)
                      )
                    ORDER BY dc.document_id, dc.ordinal
                    """,
                    {
                        "tenant_id": str(tenant_id), "namespace": namespace,
                        "source_id": source_id, "provider": provider,
                        "model": model, "prompt_version": prompt_version,
                    },
                )
                rows = await cur.fetchall()
        return [
            {
                "id": str(r["id"]), "content": r["content"], "metadata": r["metadata"],
                "document_id": str(r["document_id"]), "source_id": r["source_id"],
                # Captured *now*, at candidate-fetch time — this is the
                # revision these specific chunks/results actually belong to.
                # retry_failed_chunks() binds its final status write to this
                # exact hash rather than re-reading "whatever's current" at
                # write time, which would silently re-target a newer,
                # concurrently-published revision instead of detecting the
                # mismatch (see set_extraction_status()'s expected_content_hash).
                "document_content_hash": r["content_hash"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Entity resolution + provenance (tenant-scoped equivalent of
    # database.py's resolve_and_upsert_nodes_batch, plus mention recording)
    # ------------------------------------------------------------------

    async def resolve_and_upsert_nodes(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        entities: List[Dict[str, Any]],
        fuzzy: bool = True,
        trgm_threshold: float = 0.4,
        embedding_threshold: float = 0.90,
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> Dict[str, str]:
        """Same layered resolution as `database.py`'s single-tenant version
        (exact normalized match -> trigram+embedding fuzzy match -> new
        node), scoped to `{SCHEMA}.graph_nodes` and tenant_id."""
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
            resolved: Dict[str, str] = {}

            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT id, content FROM {SCHEMA}.graph_nodes WHERE tenant_id=%s AND namespace=%s AND content = ANY(%s)",
                    (str(tenant_id), namespace, unique_norms),
                )
                for row in await cur.fetchall():
                    resolved[row["content"]] = str(row["id"])

            if resolved:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        f"UPDATE {SCHEMA}.graph_nodes SET embedding=%s, metadata = metadata || %s WHERE tenant_id=%s AND id=%s",
                        [
                            (
                                norm_to_embedding[norm],
                                json.dumps(norm_to_metadata[norm] or {}),
                                str(tenant_id),
                                node_id,
                            )
                            for norm, node_id in resolved.items()
                        ],
                    )

            unmatched = [n for n in unique_norms if n not in resolved]

            fuzzy_merges: Dict[str, str] = {}
            # Versioned service, deployment, incident, and runbook identifiers
            # are exact identities. Fuzzy-merging `service-001` into
            # `service-002` creates unsupported cross-document graph paths.
            # Human names without digits can still use the two-signal resolver.
            fuzzy_candidates = [
                value for value in unmatched if not any(char.isdigit() for char in value)
            ]
            if fuzzy and fuzzy_candidates:
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""
                        SELECT q.content AS query_content, c.id, c.embedding AS matched_embedding, c.sim
                        FROM unnest(%(unmatched)s::text[]) AS q(content)
                        LEFT JOIN LATERAL (
                            SELECT id, embedding, similarity(content, q.content) AS sim
                            FROM {SCHEMA}.graph_nodes
                            WHERE tenant_id = %(tenant_id)s AND namespace = %(namespace)s AND content %% q.content
                            ORDER BY sim DESC
                            LIMIT 1
                        ) c ON true
                        WHERE c.sim >= %(trgm_threshold)s
                        """,
                        {
                            "unmatched": fuzzy_candidates,
                            "tenant_id": str(tenant_id),
                            "namespace": namespace,
                            "trgm_threshold": trgm_threshold,
                        },
                    )
                    candidates = await cur.fetchall()
                for cand in candidates:
                    q = cand["query_content"]
                    cos_sim = _cosine_similarity(
                        norm_to_embedding[q], _as_float_list(cand["matched_embedding"])
                    )
                    if cos_sim >= embedding_threshold:
                        fuzzy_merges[q] = str(cand["id"])

            still_unmatched = [n for n in unmatched if n not in fuzzy_merges]
            new_ids: Dict[str, str] = {}
            if still_unmatched:
                values_sql = ", ".join(["(%s,%s,%s,%s,%s)"] * len(still_unmatched))
                params: List[Any] = []
                for n in still_unmatched:
                    params.extend(
                        [str(tenant_id), namespace, n, norm_to_embedding[n], json.dumps(norm_to_metadata[n] or {})]
                    )
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""
                        INSERT INTO {SCHEMA}.graph_nodes (tenant_id, namespace, content, embedding, metadata)
                        VALUES {values_sql}
                        ON CONFLICT (tenant_id, namespace, content) DO UPDATE SET
                            embedding = EXCLUDED.embedding, metadata = {SCHEMA}.graph_nodes.metadata || EXCLUDED.metadata
                        RETURNING id, content
                        """,
                        params,
                    )
                    rows = await cur.fetchall()
                by_content = {r["content"]: str(r["id"]) for r in rows}
                new_ids = {n: by_content[n] for n in still_unmatched}

            if fuzzy_merges:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        f"UPDATE {SCHEMA}.graph_nodes SET metadata = metadata || %s WHERE tenant_id=%s AND id=%s",
                        [
                            (
                                json.dumps({**(norm_to_metadata[n] or {}), "_resolution": {"merged_from": n, "method": "fuzzy_trgm_embedding"}}),
                                str(tenant_id),
                                node_id,
                            )
                            for n, node_id in fuzzy_merges.items()
                        ],
                    )

            norm_to_id = {**resolved, **fuzzy_merges, **new_ids}

            # A new or merged node changes graph structure just as much as
            # a new edge does — community detection needs to see it in the
            # next run too, not just edge changes. Needs its own cursor:
            # every `cur` above belongs to an `async with conn.cursor()`
            # block that has already exited (and closed it) by this point.
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO {SCHEMA}.community_dirty (tenant_id, namespace)
                    VALUES (%s, %s) ON CONFLICT (tenant_id, namespace) DO NOTHING
                    """,
                    (str(tenant_id), namespace),
                )

            return {original: norm_to_id[norm] for original, norm in original_to_norm.items()}

        if connection:
            return await _run(connection)
        async with self.tenant_connection(tenant_id) as conn:
            return await _run(conn)

    async def upsert_edges(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        edges: List[Dict[str, Any]],
        connection: Optional[psycopg.AsyncConnection] = None,
        evidence_backed: bool = False,
    ) -> Dict[tuple, str]:
        if not edges:
            return {}

        aggregated: Dict[Any, Dict[str, Any]] = {}
        for edge in edges:
            key = (edge["source_id"], edge["target_id"], edge["relation"])
            if key not in aggregated:
                aggregated[key] = {"weight": 0.0, "metadata": {}}
            aggregated[key]["weight"] += edge.get("weight", 1.0)
            aggregated[key]["metadata"] = {**aggregated[key]["metadata"], **(edge.get("metadata") or {})}

        items = list(aggregated.items())

        async def _run(conn: psycopg.AsyncConnection):
            edge_ids: Dict[tuple, str] = {}
            async with conn.cursor() as cur:
                for start in range(0, len(items), MAX_ROWS_PER_STATEMENT):
                    chunk = items[start : start + MAX_ROWS_PER_STATEMENT]
                    values_sql = ", ".join(["(%s,%s,%s,%s,%s,%s,%s,%s)"] * len(chunk))
                    params: List[Any] = []
                    for (source_id, target_id, relation), agg in chunk:
                        # Extracted relationships derive their strength from
                        # edge_mentions. Deterministic caller-supplied edges
                        # retain the old additive weight behavior as manual
                        # support for compatibility.
                        manual_weight = 0.0 if evidence_backed else agg["weight"]
                        params.extend([
                            str(tenant_id), namespace, source_id, target_id,
                            relation, agg["weight"], manual_weight,
                        ])
                        params.append(json.dumps(agg["metadata"]))
                    await cur.execute(
                        f"""
                        INSERT INTO {SCHEMA}.graph_edges
                            (tenant_id, namespace, source_node_id, target_node_id, relation, weight, manual_weight, metadata)
                        VALUES {values_sql}
                        ON CONFLICT (tenant_id, namespace, source_node_id, target_node_id, relation) DO UPDATE SET
                            manual_weight = {SCHEMA}.graph_edges.manual_weight + EXCLUDED.manual_weight,
                            weight = {SCHEMA}.graph_edges.manual_weight + EXCLUDED.manual_weight + {SCHEMA}.graph_edges.support_count,
                            metadata = {SCHEMA}.graph_edges.metadata || EXCLUDED.metadata
                        RETURNING id, source_node_id, target_node_id, relation
                        """,
                        params,
                    )
                    for row in await cur.fetchall():
                        edge_ids[(str(row["source_node_id"]), str(row["target_node_id"]), row["relation"])] = str(row["id"])
                await cur.execute(
                    f"""
                    INSERT INTO {SCHEMA}.community_dirty (tenant_id, namespace)
                    VALUES (%s, %s) ON CONFLICT (tenant_id, namespace) DO NOTHING
                    """,
                    (str(tenant_id), namespace),
                )
            return edge_ids

        if connection:
            return await _run(connection)
        async with self.tenant_connection(tenant_id) as conn:
            return await _run(conn)

    async def record_entity_mentions(
        self,
        tenant_id: uuid.UUID,
        chunk_id: str,
        node_ids: List[str],
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> None:
        """Records which chunk mentioned which entities — the provenance
        link the single-tenant schema in database.py doesn't have. Lets you
        answer "which documents mention X" and safely delete one document's
        contribution without affecting another's."""
        if not node_ids:
            return
        rows = [(str(tenant_id), chunk_id, node_id) for node_id in set(node_ids)]

        async def _run(conn: psycopg.AsyncConnection):
            async with conn.cursor() as cur:
                await cur.executemany(
                    f"""
                    INSERT INTO {SCHEMA}.entity_mentions (tenant_id, chunk_id, node_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (tenant_id, chunk_id, node_id) DO NOTHING
                    """,
                    rows,
                )

        if connection:
            await _run(connection)
            return
        async with self.tenant_connection(tenant_id) as conn:
            await _run(conn)

    async def record_edge_mentions(
        self,
        tenant_id: uuid.UUID,
        chunk_id: str,
        edge_ids: List[str],
        connection: Optional[psycopg.AsyncConnection] = None,
    ) -> None:
        """Records the exact relationships asserted by a chunk."""
        if not edge_ids:
            return
        rows = [(str(tenant_id), chunk_id, edge_id) for edge_id in set(edge_ids)]

        async def _run(conn: psycopg.AsyncConnection):
            async with conn.cursor() as cur:
                await cur.executemany(
                    f"""
                    INSERT INTO {SCHEMA}.edge_mentions (tenant_id, chunk_id, edge_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (tenant_id, chunk_id, edge_id) DO NOTHING
                    """,
                    rows,
                )

        if connection:
            await _run(connection)
            return
        async with self.tenant_connection(tenant_id) as conn:
            await _run(conn)

    async def get_edge_evidence(
        self, tenant_id: uuid.UUID, namespace: str, edge_id: str
    ) -> List[Dict[str, Any]]:
        """Returns documents/chunks that explicitly asserted an edge."""
        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT em.edge_id, em.confidence,
                           d.id AS document_id, d.source_id,
                           dc.id AS chunk_id, dc.ordinal, dc.content,
                           dc.metadata
                    FROM {SCHEMA}.edge_mentions em
                    JOIN {SCHEMA}.document_chunks dc
                      ON dc.tenant_id = em.tenant_id AND dc.id = em.chunk_id
                    JOIN {SCHEMA}.documents d
                      ON d.tenant_id = dc.tenant_id AND d.id = dc.document_id
                    WHERE em.tenant_id = %s AND em.edge_id = %s
                      AND dc.namespace = %s
                    ORDER BY d.source_id, dc.ordinal
                    """,
                    (str(tenant_id), edge_id, namespace),
                )
                rows = await cur.fetchall()
                return [
                    {
                        **row,
                        "edge_id": str(row["edge_id"]),
                        "document_id": str(row["document_id"]),
                        "chunk_id": str(row["chunk_id"]),
                    }
                    for row in rows
                ]

    async def get_edges_evidence(
        self, tenant_id: uuid.UUID, namespace: str, edge_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Batch-load citable chunks supporting traversed edges."""
        if not edge_ids:
            return []
        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT em.edge_id, em.confidence,
                           d.id AS document_id, d.source_id,
                           dc.id AS chunk_id, dc.ordinal, dc.content,
                           dc.metadata
                    FROM {SCHEMA}.edge_mentions em
                    JOIN {SCHEMA}.document_chunks dc
                      ON dc.tenant_id = em.tenant_id AND dc.id = em.chunk_id
                    JOIN {SCHEMA}.documents d
                      ON d.tenant_id = dc.tenant_id AND d.id = dc.document_id
                    WHERE em.tenant_id = %s AND em.edge_id = ANY(%s::uuid[])
                      AND dc.namespace = %s
                    ORDER BY em.edge_id, d.source_id, dc.ordinal
                    """,
                    (str(tenant_id), edge_ids, namespace),
                )
                rows = await cur.fetchall()
        return [
            {
                **row,
                "edge_id": str(row["edge_id"]),
                "document_id": str(row["document_id"]),
                "chunk_id": str(row["chunk_id"]),
            }
            for row in rows
        ]

    async def get_mentioning_documents(
        self, tenant_id: uuid.UUID, namespace: str, node_id: str
    ) -> List[Dict[str, Any]]:
        """Answers "which documents mention this entity" — the provenance
        query the legacy schema structurally cannot support."""
        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT DISTINCT d.id, d.source_id, dc.id AS chunk_id, dc.ordinal
                    FROM {SCHEMA}.entity_mentions em
                    JOIN {SCHEMA}.document_chunks dc ON dc.tenant_id = em.tenant_id AND dc.id = em.chunk_id
                    JOIN {SCHEMA}.documents d ON d.tenant_id = dc.tenant_id AND d.id = dc.document_id
                    WHERE em.tenant_id = %s AND em.node_id = %s AND dc.namespace = %s
                    ORDER BY d.source_id, dc.ordinal
                    """,
                    (str(tenant_id), node_id, namespace),
                )
                return await cur.fetchall()

    # ------------------------------------------------------------------
    # Hybrid retrieval: lexical (FTS) + semantic (vector), fused via RRF
    # ------------------------------------------------------------------

    async def semantic_search_chunks(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        query_embedding: List[float],
        top_chunks: int = 10,
        metadata_filter: MetadataFilter = None,
    ) -> List[Dict[str, Any]]:
        """Semantic-only chunk retrieval used as an explicit evaluation
        baseline. It intentionally returns the same result shape as
        ``hybrid_search`` so callers can compare strategies without special
        casing their result handling."""
        if not 1 <= top_chunks <= 100:
            raise ValueError("top_chunks must be between 1 and 100")
        filter_sql, filter_params = compile_metadata_filter(
            metadata_filter, column="dc.metadata", param_prefix="semantic"
        )
        params = {
            "tenant_id": str(tenant_id),
            "namespace": namespace,
            "query_embedding": query_embedding,
            "top_chunks": top_chunks,
            **filter_params,
        }
        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    WITH ranked AS (
                        SELECT dc.id,
                               (dc.embedding <=> %(query_embedding)s::{self.vector_type}) AS distance,
                               row_number() OVER (
                                   ORDER BY (dc.embedding <=> %(query_embedding)s::{self.vector_type}) ASC
                               ) AS sem_rank
                        FROM {SCHEMA}.document_chunks dc
                        WHERE dc.tenant_id = %(tenant_id)s
                          AND dc.namespace = %(namespace)s
                          AND dc.status = 'active'
                          AND ({filter_sql})
                        ORDER BY distance ASC
                        LIMIT %(top_chunks)s
                    )
                    SELECT dc.id, dc.document_id, d.source_id, dc.ordinal, dc.content,
                           1.0 / (60.0 + ranked.sem_rank) AS rrf_score,
                           NULL::bigint AS lex_rank, ranked.sem_rank
                    FROM ranked
                    JOIN {SCHEMA}.document_chunks dc
                      ON dc.tenant_id = %(tenant_id)s AND dc.id = ranked.id
                    JOIN {SCHEMA}.documents d
                      ON d.tenant_id = dc.tenant_id AND d.id = dc.document_id
                    ORDER BY ranked.sem_rank ASC
                    """,
                    params,
                )
                return await cur.fetchall()

    async def hybrid_search(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        query_text: str,
        query_embedding: List[float],
        semantic_candidates: int = 40,
        lexical_candidates: int = 40,
        rrf_k: int = 60,
        top_chunks: int = 10,
        metadata_filter: MetadataFilter = None,
    ) -> List[Dict[str, Any]]:
        """Searches chunks (not entity labels) via two independent rankings
        — Postgres full-text search (`websearch_to_tsquery` + `ts_rank_cd`
        over a generated `tsvector`, `simple` config so identifiers and
        non-English tokens aren't stemmed away) and pgvector cosine
        similarity — fused with Reciprocal Rank Fusion so a chunk that
        ranks decently on *both* signals outranks one that's a perfect hit
        on only one. This is genuinely hybrid full-text-search + vector,
        not Okapi BM25 (Postgres's built-in ranking is not BM25); see the
        README for that distinction.

        `metadata_filter` accepts either a plain dict (JSONB containment,
        `metadata @> filter` — the original behavior) or a list of typed
        clauses (`[{"field": "updated_at", "op": "gte", "value":
        "2024-01-01"}]`) for range/comparison filtering — see
        `filters.compile_metadata_filter` for the full DSL.

        Returns chunks with their individual lexical rank, semantic rank,
        and fused RRF score preserved separately (not just the final
        score), so a caller can see *why* a chunk was retrieved.
        """
        filter_sql, filter_params = compile_metadata_filter(metadata_filter, column="metadata")
        lexical_query_text = _lexical_query_terms(query_text)
        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    WITH lexical AS (
                        SELECT id, ts_rank_cd(tsv, query) AS lex_score,
                               row_number() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS lex_rank
                        FROM {SCHEMA}.document_chunks, websearch_to_tsquery('simple', %(lexical_query_text)s) AS query
                        WHERE tenant_id = %(tenant_id)s AND namespace = %(namespace)s
                          AND status = 'active' AND tsv @@ query
                          AND ({filter_sql})
                        ORDER BY lex_score DESC
                        LIMIT %(lexical_candidates)s
                    ),
                    semantic AS (
                        SELECT id, (embedding <=> %(query_embedding)s::{self.vector_type}) AS distance,
                               row_number() OVER (ORDER BY (embedding <=> %(query_embedding)s::{self.vector_type}) ASC) AS sem_rank
                        FROM {SCHEMA}.document_chunks
                        WHERE tenant_id = %(tenant_id)s AND namespace = %(namespace)s AND status = 'active'
                          AND ({filter_sql})
                        ORDER BY distance ASC
                        LIMIT %(semantic_candidates)s
                    ),
                    fused AS (
                        SELECT COALESCE(l.id, s.id) AS id,
                               COALESCE(1.0 / (%(rrf_k)s + l.lex_rank), 0.0)
                                 + COALESCE(1.0 / (%(rrf_k)s + s.sem_rank), 0.0) AS rrf_score,
                               l.lex_rank, s.sem_rank
                        FROM lexical l
                        FULL OUTER JOIN semantic s ON l.id = s.id
                    )
                    SELECT dc.id, dc.document_id, dc.ordinal, dc.content, dc.metadata, d.source_id,
                           f.rrf_score, f.lex_rank, f.sem_rank
                    FROM fused f
                    JOIN {SCHEMA}.document_chunks dc
                        ON dc.tenant_id = %(tenant_id)s AND dc.id = f.id
                    JOIN {SCHEMA}.documents d
                        ON d.tenant_id = dc.tenant_id AND d.id = dc.document_id
                    ORDER BY f.rrf_score DESC
                    LIMIT %(top_chunks)s
                    """,
                    {
                        "tenant_id": str(tenant_id),
                        "namespace": namespace,
                        "lexical_query_text": lexical_query_text,
                        "query_embedding": query_embedding,
                        "semantic_candidates": semantic_candidates,
                        "lexical_candidates": lexical_candidates,
                        "rrf_k": rrf_k,
                        "top_chunks": top_chunks,
                        **filter_params,
                    },
                )
                return await cur.fetchall()

    async def mentioned_node_ids_for_chunks(
        self, tenant_id: uuid.UUID, chunk_ids: List[str]
    ) -> List[str]:
        """Seeds for graph traversal: entities mentioned by the chunks
        hybrid search surfaced, so the graph expansion step starts from
        actual retrieval evidence instead of only a fresh vector_search
        over entity-label embeddings."""
        if not chunk_ids:
            return []
        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT DISTINCT node_id FROM {SCHEMA}.entity_mentions WHERE tenant_id=%s AND chunk_id = ANY(%s)",
                    (str(tenant_id), chunk_ids),
                )
                return [str(r["node_id"]) for r in await cur.fetchall()]

    async def mentioned_node_scores_for_chunks(
        self, tenant_id: uuid.UUID, chunk_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Aggregate retrieval relevance from evidence chunks to graph seeds.

        A node mentioned by several high-ranking chunks receives the highest
        supporting chunk score. Scores are normalized so the best seed is 1.
        """
        if not chunk_scores:
            return {}
        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT chunk_id, node_id FROM {SCHEMA}.entity_mentions "
                    "WHERE tenant_id=%s AND chunk_id = ANY(%s::uuid[])",
                    (str(tenant_id), list(chunk_scores)),
                )
                rows = await cur.fetchall()
        scores: Dict[str, float] = {}
        for row in rows:
            node_id = str(row["node_id"])
            score = float(chunk_scores.get(str(row["chunk_id"]), 0.0))
            scores[node_id] = max(scores.get(node_id, 0.0), score)
        maximum = max(scores.values(), default=0.0)
        if maximum <= 0:
            return {node_id: 1.0 for node_id in scores}
        return {node_id: score / maximum for node_id, score in scores.items()}

    # ------------------------------------------------------------------
    # Graph traversal (tenant-scoped equivalent of database.py's version)
    # ------------------------------------------------------------------

    async def traverse_graph(
        self,
        tenant_id: uuid.UUID,
        seed_node_ids: List[str],
        namespace: str,
        max_hops: int = 2,
        seed_scores: Optional[Dict[str, float]] = None,
        directed: bool = False,
        relation_types: Optional[List[str]] = None,
        exclude_relation_types: Optional[List[str]] = None,
        min_weight: float = 0.0,
        score_decay: float = 0.7,
        max_neighbors_per_node: int = 20,
        metadata_filter: MetadataFilter = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """`metadata_filter` accepts either a plain dict (JSONB containment
        — the original behavior) or a list of typed clauses for
        range/comparison filtering (see `filters.compile_metadata_filter`).
        Applied to seeds too, so a seed that doesn't match never enters the
        expansion. This filters *nodes* only, not edges; there's no
        equivalent "only follow edges whose supporting evidence matches"
        check (that would need edge-level confidence/evidence-mention data
        this schema doesn't track per edge)."""
        if max_hops > MAX_HOPS_HARD_LIMIT:
            raise ValueError(f"max_hops={max_hops} exceeds the hard limit of {MAX_HOPS_HARD_LIMIT}.")
        if not seed_node_ids:
            return {"nodes": [], "edges": []}

        seed_scores = seed_scores or {sid: 1.0 for sid in seed_node_ids}
        scores_array = [seed_scores.get(sid, 1.0) for sid in seed_node_ids]
        filter_sql, filter_params = compile_metadata_filter(metadata_filter, column="n.metadata")

        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    WITH RECURSIVE seeds AS (
                        SELECT unnest(%(seed_ids)s::uuid[]) AS id,
                               unnest(%(seed_scores)s::float[]) AS seed_score
                    ),
                    graph_expansion AS (
                        SELECT n.id, n.content, n.metadata, 0 AS depth,
                               ARRAY[n.id] AS visited, s.seed_score AS score
                        FROM {SCHEMA}.graph_nodes n
                        JOIN seeds s ON s.id = n.id
                        WHERE n.tenant_id = %(tenant_id)s AND n.namespace = %(namespace)s
                          AND ({filter_sql})

                        UNION ALL

                        SELECT n.id, n.content, n.metadata, ge.depth + 1,
                               ge.visited || n.id,
                               ge.score * %(decay)s * (1 - exp(-nb.weight)) AS score
                        FROM graph_expansion ge
                        CROSS JOIN LATERAL (
                            SELECT
                                CASE WHEN e.source_node_id = ge.id
                                     THEN e.target_node_id ELSE e.source_node_id END AS neighbor_id,
                                e.weight
                            FROM {SCHEMA}.graph_edges e
                            WHERE e.tenant_id = %(tenant_id)s
                              AND (e.source_node_id = ge.id OR (NOT %(directed)s AND e.target_node_id = ge.id))
                              AND e.weight >= %(min_weight)s
                              -- Defense in depth alongside prune_unsupported_edges()
                              -- (tenant_engine.py): an evidence-backed edge is created
                              -- before its first mention is recorded, so a failure in
                              -- between can — until that cleanup runs — leave a
                              -- zero-weight edge that e.weight >= min_weight (default
                              -- 0.0) would not filter out on its own. Traversal
                              -- correctness should not depend entirely on that cleanup
                              -- having already run.
                              AND (e.support_count > 0 OR e.manual_weight > 0)
                              AND (%(relation_types)s::text[] IS NULL OR e.relation = ANY(%(relation_types)s))
                              AND (%(exclude_relation_types)s::text[] IS NULL OR NOT (e.relation = ANY(%(exclude_relation_types)s)))
                            ORDER BY e.weight DESC
                            LIMIT %(max_neighbors)s
                        ) nb
                        JOIN {SCHEMA}.graph_nodes n ON n.tenant_id = %(tenant_id)s AND n.id = nb.neighbor_id
                        WHERE ge.depth < %(max_hops)s
                          AND n.id != ge.id
                          AND NOT (n.id = ANY(ge.visited))
                          AND ({filter_sql})
                    )
                    SELECT id, content, metadata, MIN(depth) AS hop_distance, MAX(score) AS score
                    FROM graph_expansion
                    GROUP BY id, content, metadata
                    ORDER BY score DESC NULLS LAST
                    """,
                    {
                        "seed_ids": seed_node_ids,
                        "seed_scores": scores_array,
                        "tenant_id": str(tenant_id),
                        "namespace": namespace,
                        "max_hops": max_hops,
                        "directed": directed,
                        "relation_types": relation_types,
                        "exclude_relation_types": exclude_relation_types,
                        "min_weight": min_weight,
                        "decay": score_decay,
                        "max_neighbors": max_neighbors_per_node,
                        **filter_params,
                    },
                )
                nodes = await cur.fetchall()
                node_ids = [n["id"] for n in nodes]

                if node_ids:
                    # This is an *induced* edge fetch — every edge between
                    # any two reached nodes, not just the ones the recursive
                    # expansion above actually walked across to reach them.
                    # Without the same filters the expansion applies, a node
                    # pair that was reached via other, supported paths could
                    # still surface an edge between them that has zero
                    # support (see prune_unsupported_edges()'s docstring)
                    # or that the caller explicitly asked to exclude —
                    # silently reintroducing exactly what those filters are
                    # meant to keep out of the returned context/citations.
                    await cur.execute(
                        f"""
                        SELECT e.id, e.source_node_id, e.target_node_id, e.relation, e.metadata, e.weight,
                               s.content AS source_content, t.content AS target_content
                        FROM {SCHEMA}.graph_edges e
                        JOIN {SCHEMA}.graph_nodes s ON s.tenant_id = %(tenant_id)s AND e.source_node_id = s.id
                        JOIN {SCHEMA}.graph_nodes t ON t.tenant_id = %(tenant_id)s AND e.target_node_id = t.id
                        WHERE e.tenant_id = %(tenant_id)s
                          AND e.source_node_id = ANY(%(node_ids)s) AND e.target_node_id = ANY(%(node_ids)s)
                          AND (e.support_count > 0 OR e.manual_weight > 0)
                          AND e.weight >= %(min_weight)s
                          AND (%(relation_types)s::text[] IS NULL OR e.relation = ANY(%(relation_types)s))
                          AND (%(exclude_relation_types)s::text[] IS NULL OR NOT (e.relation = ANY(%(exclude_relation_types)s)))
                        """,
                        {
                            "tenant_id": str(tenant_id), "node_ids": node_ids,
                            "min_weight": min_weight, "relation_types": relation_types,
                            "exclude_relation_types": exclude_relation_types,
                        },
                    )
                    edges = await cur.fetchall()
                else:
                    edges = []

                return {"nodes": nodes, "edges": edges}

    async def vector_search_nodes(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Entity-label vector search, kept for callers that want to seed
        traversal directly from query/entity similarity rather than (or in
        addition to) hybrid chunk search."""
        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT id, content, metadata, (embedding <=> %s::{self.vector_type}) AS distance
                    FROM {SCHEMA}.graph_nodes
                    WHERE tenant_id = %s AND namespace = %s
                    ORDER BY distance ASC
                    LIMIT %s
                    """,
                    (query_embedding, str(tenant_id), namespace, top_k),
                )
                return await cur.fetchall()

    # ------------------------------------------------------------------
    # Administrative entity resolution corrections
    # ------------------------------------------------------------------

    async def merge_entities(
        self, tenant_id: uuid.UUID, namespace: str, source_content: str, target_content: str
    ) -> Dict[str, Any]:
        """Manually merges `source_content`'s node into `target_content`'s:
        repoints its edges and entity_mentions onto the target, merges
        metadata, and deletes the now-empty source node. For correcting a
        missed automatic merge (layered resolution decided not to merge two
        mentions of the same real-world entity) — the inverse operation,
        `split_entity()`, is NOT implemented: undoing a merge requires
        knowing which original mentions came from which side, which this
        schema's entity_mentions table (chunk -> node, not chunk -> alias)
        doesn't preserve once a merge has happened. Splitting is left as an
        explicit gap rather than a fabricated best-effort implementation.
        """
        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT id, metadata FROM {SCHEMA}.graph_nodes WHERE tenant_id=%s AND namespace=%s AND content=%s",
                    (str(tenant_id), namespace, source_content),
                )
                source = await cur.fetchone()
                await cur.execute(
                    f"SELECT id, metadata FROM {SCHEMA}.graph_nodes WHERE tenant_id=%s AND namespace=%s AND content=%s",
                    (str(tenant_id), namespace, target_content),
                )
                target = await cur.fetchone()
                if not source or not target:
                    raise ValueError(f"Both entities must exist in namespace {namespace!r} to merge them.")
                if source["id"] == target["id"]:
                    return {"merged": False, "reason": "source and target are already the same node"}

                # Repoint edges. ON CONFLICT can happen if the target already
                # has an edge with the same (relation, other-endpoint); fold
                # manual_weight in rather than erroring. `weight` itself is
                # never trusted directly here — it's recomputed below from
                # manual_weight + actual support_count, after evidence
                # (edge_mentions) is repointed too. Previously this only
                # carried `weight` forward and left manual_weight/
                # support_count at their INSERT defaults (0), and never
                # repointed edge_mentions at all — so a repointed edge's
                # accumulated evidence was silently discarded, which
                # traversal's zero-support filter now makes visible (an edge
                # with no real mentions and manual_weight=0 is, correctly,
                # no longer traversable — that's the point of that filter).
                for col in ("source_node_id", "target_node_id"):
                    other_col = "target_node_id" if col == "source_node_id" else "source_node_id"
                    await cur.execute(
                        f"""
                        SELECT id, {other_col} AS other_id, relation, manual_weight, metadata
                        FROM {SCHEMA}.graph_edges WHERE tenant_id=%s AND {col}=%s
                        """,
                        (str(tenant_id), source["id"]),
                    )
                    rows = await cur.fetchall()
                    for row in rows:
                        await cur.execute(
                            f"""
                            INSERT INTO {SCHEMA}.graph_edges (tenant_id, namespace, source_node_id, target_node_id, relation, weight, manual_weight, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (tenant_id, namespace, source_node_id, target_node_id, relation) DO UPDATE SET
                                manual_weight = {SCHEMA}.graph_edges.manual_weight + EXCLUDED.manual_weight
                            RETURNING id
                            """,
                            (
                                str(tenant_id), namespace,
                                target["id"] if col == "source_node_id" else row["other_id"],
                                row["other_id"] if col == "source_node_id" else target["id"],
                                row["relation"], row["manual_weight"], row["manual_weight"],
                                json.dumps(row["metadata"] or {}),
                            ),
                        )
                        surviving_edge_id = (await cur.fetchone())["id"]
                        # edge_mentions.recompute_edge_support fires on
                        # INSERT/DELETE, not UPDATE, so repointing edge_id
                        # here doesn't trigger it automatically — the
                        # explicit recompute right after does that job.
                        await cur.execute(
                            f"""
                            UPDATE {SCHEMA}.edge_mentions em SET edge_id = %s
                            WHERE em.tenant_id = %s AND em.edge_id = %s
                              AND NOT EXISTS (
                                  SELECT 1 FROM {SCHEMA}.edge_mentions em2
                                  WHERE em2.tenant_id = em.tenant_id AND em2.chunk_id = em.chunk_id
                                    AND em2.edge_id = %s
                              )
                            """,
                            (surviving_edge_id, str(tenant_id), row["id"], surviving_edge_id),
                        )
                        await cur.execute(
                            f"""
                            UPDATE {SCHEMA}.graph_edges e
                            SET support_count = sub.support_count,
                                weight = e.manual_weight + sub.support_count
                            FROM (
                                SELECT count(*)::int AS support_count
                                FROM {SCHEMA}.edge_mentions WHERE tenant_id = %s AND edge_id = %s
                            ) sub
                            WHERE e.tenant_id = %s AND e.id = %s
                            """,
                            (str(tenant_id), surviving_edge_id, str(tenant_id), surviving_edge_id),
                        )
                    await cur.execute(f"DELETE FROM {SCHEMA}.graph_edges WHERE tenant_id=%s AND {col}=%s", (str(tenant_id), source["id"]))

                # Repoint mentions (ON CONFLICT DO NOTHING: a chunk may
                # already mention the target directly).
                await cur.execute(
                    f"""
                    UPDATE {SCHEMA}.entity_mentions em SET node_id = %s
                    WHERE em.tenant_id = %s AND em.node_id = %s
                      AND NOT EXISTS (
                          SELECT 1 FROM {SCHEMA}.entity_mentions em2
                          WHERE em2.tenant_id = em.tenant_id AND em2.chunk_id = em.chunk_id AND em2.node_id = %s
                      )
                    """,
                    (target["id"], str(tenant_id), source["id"], target["id"]),
                )
                await cur.execute(f"DELETE FROM {SCHEMA}.entity_mentions WHERE tenant_id=%s AND node_id=%s", (str(tenant_id), source["id"]))

                target_history = _as_merge_list((target["metadata"] or {}).get("_merged_from"))
                source_history = _as_merge_list((source["metadata"] or {}).get("_merged_from"))
                merged_metadata = {
                    **(target["metadata"] or {}),
                    "_merged_from": target_history + source_history + [source_content],
                }
                await cur.execute(
                    f"UPDATE {SCHEMA}.graph_nodes SET metadata=%s WHERE tenant_id=%s AND id=%s",
                    (json.dumps(merged_metadata), str(tenant_id), target["id"]),
                )
                await cur.execute(f"DELETE FROM {SCHEMA}.graph_nodes WHERE tenant_id=%s AND id=%s", (str(tenant_id), source["id"]))

        return {"merged": True, "target_id": str(target["id"])}

    # ------------------------------------------------------------------
    # Explainability: point-to-point path reconstruction
    # ------------------------------------------------------------------

    async def _decorate_paths_with_evidence(
        self, tenant_id: uuid.UUID, namespace: str, paths: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Attach document/chunk evidence to every traversed edge.

        Path search stays a single bounded SQL operation; evidence is loaded
        afterward only for the returned paths, keeping the recursive query's
        row shape small and making the citation contract explicit.
        """
        for result in paths:
            for step in result.get("path", [])[1:]:
                edge_id = step.get("edge_id")
                if edge_id:
                    step["evidence"] = await self.get_edge_evidence(
                        tenant_id, namespace, str(edge_id)
                    )
        return paths

    async def find_path(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        source_id: str,
        target_id: str,
        max_hops: int = 3,
        directed: bool = False,
        relation_types: Optional[List[str]] = None,
        exclude_relation_types: Optional[List[str]] = None,
        min_weight: float = 0.0,
        max_neighbors_per_node: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """Reconstructs the single best (highest-scoring, then shortest)
        path from `source_id` to `target_id`, for explainability — this is
        the "why did the system connect Node A to Node C" answer that a
        bag of (node, score, hop_distance) tuples from `traverse_graph`
        can't give you: that method deliberately discards *which specific*
        path produced a node's best score once nodes are grouped, since a
        broad multi-seed retrieval doesn't have a single meaningful
        source/target pair to reconstruct a path between.

        Returns `None` if no path exists within `max_hops` (an explicit,
        honest "not connected" answer, not an empty/ambiguous result), or
        `{"path": [{"node": str, "relation": str | None}, ...], "score":
        float, "hop_distance": int}` — `path[0]["relation"]` is always
        `None` (the source has nothing "arriving" at it); each subsequent
        step's `relation` is the edge that was crossed to reach that node.

        Uses the same hard hop-count ceiling and per-node fan-out cap as
        `traverse_graph`, for the same reason: unbounded path search on a
        dense graph is a real cost/availability risk, not just a
        theoretical one.
        """
        if max_hops > MAX_HOPS_HARD_LIMIT:
            raise ValueError(f"max_hops={max_hops} exceeds the hard limit of {MAX_HOPS_HARD_LIMIT}.")
        if source_id == target_id:
            return None

        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    WITH RECURSIVE path_search AS (
                        SELECT n.id, 0 AS depth, ARRAY[n.id] AS visited,
                               jsonb_build_array(jsonb_build_object(
                                   'node_id', n.id, 'node', n.content,
                                   'relation', NULL::text, 'edge_id', NULL::uuid
                               )) AS path,
                               1.0::float AS score
                        FROM {SCHEMA}.graph_nodes n
                        WHERE n.tenant_id = %(tenant_id)s AND n.namespace = %(namespace)s AND n.id = %(source_id)s

                        UNION ALL

                        SELECT n.id, ps.depth + 1, ps.visited || n.id,
                               ps.path || jsonb_build_array(jsonb_build_object(
                                   'node_id', n.id, 'node', n.content,
                                   'relation', nb.relation, 'edge_id', nb.edge_id
                               )),
                               ps.score * (1 - exp(-nb.weight))
                        FROM path_search ps
                        CROSS JOIN LATERAL (
                            SELECT CASE WHEN e.source_node_id = ps.id
                                        THEN e.target_node_id ELSE e.source_node_id END AS neighbor_id,
                                   e.id AS edge_id, e.relation, e.weight
                            FROM {SCHEMA}.graph_edges e
                            WHERE e.tenant_id = %(tenant_id)s
                              AND (e.source_node_id = ps.id OR (NOT %(directed)s AND e.target_node_id = ps.id))
                              AND e.weight >= %(min_weight)s
                              -- Defense in depth alongside prune_unsupported_edges()
                              -- (tenant_engine.py): an evidence-backed edge is created
                              -- before its first mention is recorded, so a failure in
                              -- between can — until that cleanup runs — leave a
                              -- zero-weight edge that e.weight >= min_weight (default
                              -- 0.0) would not filter out on its own. Traversal
                              -- correctness should not depend entirely on that cleanup
                              -- having already run.
                              AND (e.support_count > 0 OR e.manual_weight > 0)
                              AND (%(relation_types)s::text[] IS NULL OR e.relation = ANY(%(relation_types)s))
                              AND (%(exclude_relation_types)s::text[] IS NULL OR NOT (e.relation = ANY(%(exclude_relation_types)s)))
                            ORDER BY e.weight DESC
                            LIMIT %(max_neighbors)s
                        ) nb
                        JOIN {SCHEMA}.graph_nodes n ON n.tenant_id = %(tenant_id)s AND n.id = nb.neighbor_id
                        WHERE ps.depth < %(max_hops)s
                          AND NOT (n.id = ANY(ps.visited))
                    )
                    SELECT path, score, depth
                    FROM path_search
                    WHERE id = %(target_id)s
                    ORDER BY score DESC, depth ASC
                    LIMIT 1
                    """,
                    {
                        "tenant_id": str(tenant_id),
                        "namespace": namespace,
                        "source_id": source_id,
                        "target_id": target_id,
                        "max_hops": max_hops,
                        "directed": directed,
                        "relation_types": relation_types,
                        "exclude_relation_types": exclude_relation_types,
                        "min_weight": min_weight,
                        "max_neighbors": max_neighbors_per_node,
                    },
                )
                row = await cur.fetchone()

        if not row:
            return None
        result = {
            "path": row["path"],
            "score": row["score"],
            "hop_distance": row["depth"],
            "source_id": source_id,
            "target_id": target_id,
        }
        await self._decorate_paths_with_evidence(tenant_id, namespace, [result])
        return result

    async def find_paths(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        source_ids: List[str],
        target_ids: List[str],
        max_hops: int = 3,
        top_k: int = 5,
        directed: bool = False,
        relation_types: Optional[List[str]] = None,
        exclude_relation_types: Optional[List[str]] = None,
        min_weight: float = 0.0,
        max_neighbors_per_node: int = 20,
    ) -> List[Dict[str, Any]]:
        """Returns the globally best bounded paths across source/target sets.

        This intentionally returns the top K paths globally. It does not try
        to implement one-best-per-target or Yen-style alternate-route
        enumeration; those are different contracts with different cost
        characteristics.
        """
        if max_hops > MAX_HOPS_HARD_LIMIT:
            raise ValueError(f"max_hops={max_hops} exceeds the hard limit of {MAX_HOPS_HARD_LIMIT}.")
        if top_k <= 0 or not source_ids or not target_ids:
            return []

        async with self.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    WITH RECURSIVE path_search AS (
                        SELECT n.id, n.id AS source_id, 0 AS depth,
                               ARRAY[n.id] AS visited,
                               jsonb_build_array(jsonb_build_object(
                                   'node_id', n.id, 'node', n.content,
                                   'relation', NULL::text, 'edge_id', NULL::uuid
                               )) AS path,
                               1.0::float AS score
                        FROM {SCHEMA}.graph_nodes n
                        WHERE n.tenant_id = %(tenant_id)s
                          AND n.namespace = %(namespace)s
                          AND n.id = ANY(%(source_ids)s::uuid[])

                        UNION ALL

                        SELECT n.id, ps.source_id, ps.depth + 1,
                               ps.visited || n.id,
                               ps.path || jsonb_build_array(jsonb_build_object(
                                   'node_id', n.id, 'node', n.content,
                                   'relation', nb.relation, 'edge_id', nb.edge_id
                               )),
                               ps.score * (1 - exp(-nb.weight))
                        FROM path_search ps
                        CROSS JOIN LATERAL (
                            SELECT CASE WHEN e.source_node_id = ps.id
                                        THEN e.target_node_id ELSE e.source_node_id END AS neighbor_id,
                                   e.id AS edge_id, e.relation, e.weight
                            FROM {SCHEMA}.graph_edges e
                            WHERE e.tenant_id = %(tenant_id)s
                              AND (e.source_node_id = ps.id OR (NOT %(directed)s AND e.target_node_id = ps.id))
                              AND e.weight >= %(min_weight)s
                              -- Defense in depth alongside prune_unsupported_edges()
                              -- (tenant_engine.py): an evidence-backed edge is created
                              -- before its first mention is recorded, so a failure in
                              -- between can — until that cleanup runs — leave a
                              -- zero-weight edge that e.weight >= min_weight (default
                              -- 0.0) would not filter out on its own. Traversal
                              -- correctness should not depend entirely on that cleanup
                              -- having already run.
                              AND (e.support_count > 0 OR e.manual_weight > 0)
                              AND (%(relation_types)s::text[] IS NULL OR e.relation = ANY(%(relation_types)s))
                              AND (%(exclude_relation_types)s::text[] IS NULL OR NOT (e.relation = ANY(%(exclude_relation_types)s)))
                            ORDER BY e.weight DESC
                            LIMIT %(max_neighbors)s
                        ) nb
                        JOIN {SCHEMA}.graph_nodes n
                          ON n.tenant_id = %(tenant_id)s AND n.id = nb.neighbor_id
                        WHERE ps.depth < %(max_hops)s
                          AND NOT (n.id = ANY(ps.visited))
                    )
                    SELECT source_id, id AS target_id, path, score, depth
                    FROM path_search
                    WHERE id = ANY(%(target_ids)s::uuid[])
                    ORDER BY score DESC, depth ASC
                    LIMIT %(top_k)s
                    """,
                    {
                        "tenant_id": str(tenant_id),
                        "namespace": namespace,
                        "source_ids": source_ids,
                        "target_ids": target_ids,
                        "max_hops": max_hops,
                        "top_k": top_k,
                        "directed": directed,
                        "relation_types": relation_types,
                        "exclude_relation_types": exclude_relation_types,
                        "min_weight": min_weight,
                        "max_neighbors": max_neighbors_per_node,
                    },
                )
                rows = await cur.fetchall()

        results = [
            {
                "path": row["path"],
                "score": row["score"],
                "hop_distance": row["depth"],
                "source_id": str(row["source_id"]),
                "target_id": str(row["target_id"]),
            }
            for row in rows
        ]
        return await self._decorate_paths_with_evidence(tenant_id, namespace, results)
