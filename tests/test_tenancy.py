"""Tests for the v0.3 tenant-aware, RLS-secured storage layer.

Requires POSTGRES_URL (an admin/superuser connection, to run migrate_schema
and create the restricted runtime role) — skipped otherwise. No LLM key
needed; embeddings are supplied directly.

Test-safety note: these tests DROP/CREATE the `postgres_graph_rag` schema
and a `pgr_test_runtime` role directly, because `tenancy.py` currently
hardcodes its schema name rather than accepting it as a parameter. This is
acceptable today because that schema is new (no production deployment has
ever used it yet), but parameterizing the schema name is a reasonable
fast-follow if this ever needs to run against a database that also hosts
real tenant data under that name.
"""
import os
import uuid

import pytest
import pytest_asyncio
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

from postgres_graph_rag.tenancy import (
    SCHEMA,
    InsecureRuntimeRoleError,
    SecureGraphStore,
    migrate_schema,
)

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
DIM = 8
RUNTIME_ROLE = "pgr_test_runtime"
RUNTIME_PASSWORD = "pgr_test_runtime_pw"

pytestmark = pytest.mark.skipif(not POSTGRES_URL, reason="POSTGRES_URL not set")


def _runtime_url() -> str:
    # Swap the admin credentials in POSTGRES_URL for the restricted runtime
    # role's, keeping host/port/db the same.
    import urllib.parse as up

    parsed = up.urlparse(POSTGRES_URL)
    netloc = f"{RUNTIME_ROLE}:{RUNTIME_PASSWORD}@{parsed.hostname}:{parsed.port or 5432}"
    return up.urlunparse(parsed._replace(netloc=netloc))


def tid() -> uuid.UUID:
    return uuid.uuid4()


def vec(*pos, dim=DIM):
    v = [0.0] * dim
    for p in pos:
        v[p] = 1.0
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


@pytest_asyncio.fixture
async def admin_conn():
    conn = await psycopg.AsyncConnection.connect(POSTGRES_URL)
    yield conn
    await conn.close()


@pytest_asyncio.fixture
async def store(admin_conn):
    async with admin_conn.cursor() as cur:
        await cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        await admin_conn.commit()

    await migrate_schema(
        admin_url=POSTGRES_URL,
        runtime_role=RUNTIME_ROLE,
        runtime_password=RUNTIME_PASSWORD,
        embedding_dimension=DIM,
        migrate_legacy_data=False,
    )

    s = SecureGraphStore(_runtime_url(), vector_type="vector")
    yield s
    await s.close()


# ----------------------------------------------------------------------
# RLS security: the core, non-negotiable guarantees
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_role_cannot_bypass_rls_attributes(store):
    """Sanity check on the role itself: if this ever regresses (e.g. someone
    grants BYPASSRLS or SUPERUSER to the runtime role), every other RLS test
    in this file would start silently passing for the wrong reason."""
    async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = %s",
                (RUNTIME_ROLE,),
            )
            row = await cur.fetchone()
    assert row[0] is False, "runtime role must not be a superuser"
    assert row[1] is False, "runtime role must not have BYPASSRLS"


@pytest.mark.asyncio
async def test_secure_store_refuses_a_superuser_connection(store):
    """SecureGraphStore's entire tenant-isolation guarantee depends on the
    connecting role lacking BYPASSRLS/SUPERUSER. If it's ever handed the
    admin/owner URL by mistake (a plausible config error), it must refuse
    loudly rather than silently operating with RLS disabled."""
    insecure = SecureGraphStore(POSTGRES_URL, vector_type="vector")
    try:
        with pytest.raises(InsecureRuntimeRoleError):
            await insecure._init_pool()
    finally:
        await insecure.close()


@pytest.mark.asyncio
async def test_missing_tenant_context_fails_closed_on_read(store):
    tenant = tid()
    await store.resolve_and_upsert_nodes(tenant, "ns", [{"content": "Secret", "embedding": vec(0)}])

    async with await psycopg.AsyncConnection.connect(_runtime_url()) as conn:
        async with conn.cursor() as cur:
            # No set_config call at all -> tenant GUC is unset.
            await cur.execute(f"SELECT content FROM {SCHEMA}.graph_nodes")
            rows = await cur.fetchall()
    assert rows == []


@pytest.mark.asyncio
async def test_missing_tenant_context_fails_closed_on_write(store):
    async with await psycopg.AsyncConnection.connect(_runtime_url()) as conn:
        async with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                await cur.execute(
                    f"INSERT INTO {SCHEMA}.graph_nodes (namespace, content, embedding) VALUES ('ns', 'X', %s)",
                    (vec(0),),
                )


@pytest.mark.asyncio
async def test_connection_reuse_across_tenants_does_not_leak(store):
    """The scenario that makes pooled connections dangerous for RLS: the
    same physical connection serves tenant A, then gets reused for tenant
    B. If the tenant GUC were session-local instead of transaction-local,
    tenant B would inherit tenant A's context."""
    tenant_a, tenant_b = tid(), tid()

    async with await psycopg.AsyncConnection.connect(_runtime_url()) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT set_config('postgres_graph_rag.tenant_id', %s, true)", (str(tenant_a),))
            await cur.execute(
                f"INSERT INTO {SCHEMA}.graph_nodes (tenant_id, namespace, content, embedding) VALUES (%s, 'ns', 'SecretA', %s)",
                (str(tenant_a), vec(0)),
            )
            await conn.commit()

        # New transaction, same physical connection, different tenant.
        async with conn.cursor() as cur:
            await cur.execute("SELECT set_config('postgres_graph_rag.tenant_id', %s, true)", (str(tenant_b),))
            await cur.execute(f"SELECT content FROM {SCHEMA}.graph_nodes")
            rows = await cur.fetchall()
            await conn.commit()
        assert rows == [], "tenant B must not see tenant A's row on a reused connection"

        # Yet another transaction, tenant context never set this time.
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT content FROM {SCHEMA}.graph_nodes")
            rows_no_context = await cur.fetchall()
            await conn.rollback()
        assert rows_no_context == [], "transaction-local setting must not persist across transactions"


@pytest.mark.asyncio
async def test_store_tenant_connection_isolates_concurrent_tenants(store):
    """Same property as above, exercised through SecureGraphStore's actual
    pooled tenant_connection() rather than a hand-rolled connection."""
    tenant_a, tenant_b = tid(), tid()
    await store.resolve_and_upsert_nodes(tenant_a, "ns", [{"content": "OnlyA", "embedding": vec(0)}])
    await store.resolve_and_upsert_nodes(tenant_b, "ns", [{"content": "OnlyB", "embedding": vec(1)}])

    nodes_a = await store.vector_search_nodes(tenant_a, "ns", vec(0), top_k=10)
    nodes_b = await store.vector_search_nodes(tenant_b, "ns", vec(0), top_k=10)
    assert {n["content"] for n in nodes_a} == {"OnlyA"}
    assert {n["content"] for n in nodes_b} == {"OnlyB"}


@pytest.mark.asyncio
async def test_cross_tenant_traversal_seed_does_not_cross_over(store):
    """A node id that legitimately belongs to tenant A must not resolve
    under tenant B's traversal, even if passed in explicitly as a seed."""
    tenant_a, tenant_b = tid(), tid()
    ids_a = await store.resolve_and_upsert_nodes(tenant_a, "ns", [{"content": "OnlyA", "embedding": vec(0)}])

    result = await store.traverse_graph(tenant_b, list(ids_a.values()), namespace="ns", max_hops=2)
    assert result["nodes"] == []


# ----------------------------------------------------------------------
# Legacy data migration
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_data_migration_preserves_ids(admin_conn):
    """Seeds `public.graph_nodes`/`graph_edges` directly via raw SQL to
    simulate a pre-existing single-tenant deployment, since the
    `DatabaseManager` class that used to create this data has been removed
    (no customers depended on it, so it was deleted rather than kept
    deprecated). The schema seeded here matches exactly what
    `tenancy.py::_migrate_legacy_data()` expects to read from
    `public.graph_nodes`/`graph_edges` -- this test only replaces *how* that
    legacy-shaped data gets there, not what `migrate_legacy_data=True` does
    with it once it exists."""
    from postgres_graph_rag.tenancy import LEGACY_TENANT_ID

    async with admin_conn.cursor() as cur:
        await cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        await cur.execute("DROP TABLE IF EXISTS ingested_chunks CASCADE")
        await cur.execute("DROP TABLE IF EXISTS graph_edges CASCADE")
        await cur.execute("DROP TABLE IF EXISTS graph_nodes CASCADE")
        await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await cur.execute(
            f"""
            CREATE TABLE graph_nodes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                namespace TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({DIM}) NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await cur.execute(
            """
            CREATE TABLE graph_edges (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                namespace TEXT NOT NULL,
                source_node_id UUID NOT NULL REFERENCES graph_nodes(id),
                target_node_id UUID NOT NULL REFERENCES graph_nodes(id),
                relation TEXT NOT NULL,
                weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )

        await cur.execute(
            "INSERT INTO graph_nodes (namespace, content, embedding) VALUES (%s, %s, %s) RETURNING id",
            ("legacy-ns", "PreMigration", vec(0)),
        )
        pre_migration_id = str((await cur.fetchone())[0])

        await cur.execute(
            "INSERT INTO graph_nodes (namespace, content, embedding) VALUES (%s, %s, %s) RETURNING id",
            ("legacy-ns", "checkout-service", vec(1)),
        )
        checkout_id = str((await cur.fetchone())[0])
        await cur.execute(
            "INSERT INTO graph_nodes (namespace, content, embedding) VALUES (%s, %s, %s) RETURNING id",
            ("legacy-ns", "auth-service", vec(2)),
        )
        auth_id = str((await cur.fetchone())[0])
        node_ids = [checkout_id, auth_id]

        await cur.execute(
            "INSERT INTO graph_edges (namespace, source_node_id, target_node_id, relation, weight) "
            "VALUES (%s, %s, %s, %s, %s)",
            ("legacy-ns", checkout_id, auth_id, "depends_on", 3.0),
        )
        await admin_conn.commit()

    ids = [pre_migration_id]

    await migrate_schema(
        admin_url=POSTGRES_URL,
        runtime_role=RUNTIME_ROLE,
        runtime_password=RUNTIME_PASSWORD,
        embedding_dimension=DIM,
        migrate_legacy_data=True,
    )

    s = SecureGraphStore(_runtime_url(), vector_type="vector")
    nodes = await s.vector_search_nodes(LEGACY_TENANT_ID, "legacy-ns", vec(0), top_k=10)
    by_content = {n["content"]: n for n in nodes}
    assert "PreMigration" in by_content
    assert str(by_content["PreMigration"]["id"]) == ids[0]

    # A migrated edge has no document evidence (the legacy schema never
    # recorded provenance), but it must still be traversable -- regression
    # test for manual_weight being left at its default 0.0 during
    # migration, which made every migrated edge invisible to the
    # support_count > 0 OR manual_weight > 0 read-time filter even though
    # its `weight` column was correctly populated.
    graph = await s.traverse_graph(
        LEGACY_TENANT_ID, [node_ids[0]], namespace="legacy-ns", max_hops=1, directed=True,
    )
    await s.close()
    assert {n["content"] for n in graph["nodes"]} == {"checkout-service", "auth-service"}
    assert len(graph["edges"]) == 1
    edge = graph["edges"][0]
    assert edge["relation"] == "depends_on"
    assert edge["source_content"] == "checkout-service"
    assert edge["target_content"] == "auth-service"
    assert edge["weight"] == 3.0
    assert str(edge["source_node_id"]) == node_ids[0]
    assert str(edge["target_node_id"]) == node_ids[1]

    async with (await psycopg.AsyncConnection.connect(POSTGRES_URL, row_factory=dict_row)) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT manual_weight, support_count FROM {SCHEMA}.graph_edges "
                "WHERE tenant_id = %s AND relation = %s",
                (str(LEGACY_TENANT_ID), "depends_on"),
            )
            row = await cur.fetchone()
    assert row["manual_weight"] == 3.0
    assert row["support_count"] == 0


# ----------------------------------------------------------------------
# Lease-based extraction cache
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lease_cache_concurrent_claim_only_one_winner(store):
    tenant = tid()
    import asyncio as _asyncio

    results = await _asyncio.gather(
        store.claim_extraction(tenant, "h1", "openai", "m", "v1", "worker-A"),
        store.claim_extraction(tenant, "h1", "openai", "m", "v1", "worker-B"),
    )
    statuses = sorted(r["status"] for r in results)
    assert statuses == ["claimed", "in_progress"]


@pytest.mark.asyncio
async def test_lease_cache_reuses_completed_result(store):
    tenant = tid()
    claim = await store.claim_extraction(tenant, "h2", "openai", "m", "v1", "worker-A")
    assert claim["status"] == "claimed"
    await store.complete_extraction(tenant, "h2", "openai", "m", "v1", "worker-A", [{"subject": "A", "predicate": "p", "object": "B"}])

    cached = await store.claim_extraction(tenant, "h2", "openai", "m", "v1", "worker-B")
    assert cached["status"] == "done"
    assert cached["triplet_count"] == 1


@pytest.mark.asyncio
async def test_lease_cache_is_tenant_scoped(store):
    """The same content hash for two different tenants must not share a
    cached extraction result — otherwise tenant B could read tenant A's
    (potentially sensitive, model-extracted) triplets."""
    tenant_a, tenant_b = tid(), tid()
    claim_a = await store.claim_extraction(tenant_a, "shared-hash", "openai", "m", "v1", "worker-A")
    assert claim_a["status"] == "claimed"
    await store.complete_extraction(tenant_a, "shared-hash", "openai", "m", "v1", "worker-A", [{"subject": "Secret", "predicate": "p", "object": "X"}])

    claim_b = await store.claim_extraction(tenant_b, "shared-hash", "openai", "m", "v1", "worker-B")
    assert claim_b["status"] == "claimed", "tenant B must not see tenant A's cached extraction"


@pytest.mark.asyncio
async def test_lease_reclaim_after_expiry(store):
    """A lease claimed with a negative TTL is already expired by the time
    it's written, so a second claimant for the identical key must be able
    to reclaim it rather than getting stuck behind a dead worker forever."""
    tenant = tid()
    first = await store.claim_extraction(
        tenant, "h-expired", "openai", "m", "v1", "worker-a", lease_seconds=-1
    )
    assert first["status"] == "claimed"

    second = await store.claim_extraction(
        tenant, "h-expired", "openai", "m", "v1", "worker-b", lease_seconds=120
    )
    assert second["status"] == "claimed"


@pytest.mark.asyncio
async def test_lease_not_reclaimed_before_expiry(store):
    """Companion negative case: a live (non-expired) lease must not be
    reclaimable by another worker."""
    tenant = tid()
    first = await store.claim_extraction(
        tenant, "h-live", "openai", "m", "v1", "worker-a", lease_seconds=120
    )
    assert first["status"] == "claimed"

    second = await store.claim_extraction(
        tenant, "h-live", "openai", "m", "v1", "worker-b", lease_seconds=120
    )
    assert second["status"] == "in_progress"


# ----------------------------------------------------------------------
# Hybrid retrieval + provenance
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_search_fuses_lexical_and_semantic(store):
    tenant = tid()
    doc = await store.upsert_document(tenant, "ns", "doc-1", "h1", {})
    await store.replace_chunks(tenant, doc["id"], "ns", [
        {"content": "The quarterly revenue report shows strong growth.", "embedding": vec(0)},
        {"content": "A recipe for chocolate cake with frosting.", "embedding": vec(5)},
    ])

    results = await store.hybrid_search(tenant, "ns", "quarterly revenue", vec(0))
    assert results[0]["content"].startswith("The quarterly revenue")
    assert results[0]["lex_rank"] == 1


def test_lexical_query_terms_strips_stopwords_and_ors_content_words():
    from postgres_graph_rag.tenancy import _lexical_query_terms

    result = _lexical_query_terms(
        "Who is the on-call contact for the team that owns the service affected by CVE-2026-1188?"
    )
    terms = result.split(" OR ")
    assert "who" not in [t.lower() for t in terms]
    assert "is" not in [t.lower() for t in terms]
    assert "the" not in [t.lower() for t in terms]
    assert "CVE-2026-1188" in terms
    assert "team" in terms


def test_lexical_query_terms_falls_back_when_all_stopwords():
    from postgres_graph_rag.tenancy import _lexical_query_terms

    # A degenerate all-filler-words question shouldn't raise or return an
    # empty string that would break the SQL call.
    result = _lexical_query_terms("What is that?")
    assert result  # non-empty; exact fallback content isn't the contract


@pytest.mark.asyncio
async def test_hybrid_search_lexical_branch_ignores_question_stopwords(store):
    """Regression test for a real bug found via an adversarial e2e scenario:
    `websearch_to_tsquery` AND-conjoins every plain term with no stopword
    removal, so a natural-language question like "Who is the on-call
    contact for the team that owns the service affected by
    CVE-2026-1188?" required "who" AND "is" AND "the" AND "for" AND "by"
    (among others) to ALL appear verbatim in the same chunk — which a
    short factual chunk never satisfies. Confirmed directly against
    Postgres before fixing: the lexical branch returned zero rows even
    though the chunk below contains the named identifier verbatim."""
    tenant = tid()
    doc = await store.upsert_document(tenant, "ns", "doc-1", "h1", {})
    await store.replace_chunks(tenant, doc["id"], "ns", [
        {"content": "payment-lib-v3 has_vulnerability CVE-2026-1188. CVE-2026-1188 is_rated critical.", "embedding": vec(0)},
        {"content": "A recipe for chocolate cake with frosting.", "embedding": vec(5)},
    ])

    results = await store.hybrid_search(
        tenant, "ns",
        "Who is the on-call contact for the team that owns the service affected by CVE-2026-1188?",
        vec(0),
    )
    matched = [r for r in results if r["lex_rank"] is not None]
    assert matched, "lexical branch should match the chunk containing the named identifier"
    assert any("CVE-2026-1188" in r["content"] for r in matched)


@pytest.mark.asyncio
async def test_entity_mentions_provide_document_provenance(store):
    tenant = tid()
    doc = await store.upsert_document(tenant, "ns", "doc-1", "h1", {})
    chunk_ids = await store.replace_chunks(tenant, doc["id"], "ns", [
        {"content": "Marie Curie won the Nobel Prize.", "embedding": vec(0)},
    ])
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "Marie Curie", "embedding": vec(1)},
    ])
    await store.record_entity_mentions(tenant, chunk_ids[0], list(node_ids.values()))

    docs = await store.get_mentioning_documents(tenant, "ns", node_ids["Marie Curie"])
    assert len(docs) == 1
    assert docs[0]["source_id"] == "doc-1"


@pytest.mark.asyncio
async def test_delete_document_removes_mentions_but_keeps_shared_entity(store):
    """Two documents mention the same entity; deleting one must not affect
    the other's evidence or the entity itself."""
    tenant = tid()
    doc1 = await store.upsert_document(tenant, "ns", "doc-1", "h1", {})
    doc2 = await store.upsert_document(tenant, "ns", "doc-2", "h2", {})
    chunks1 = await store.replace_chunks(tenant, doc1["id"], "ns", [{"content": "Paris is beautiful.", "embedding": vec(0)}])
    chunks2 = await store.replace_chunks(tenant, doc2["id"], "ns", [{"content": "Paris hosted the Olympics.", "embedding": vec(1)}])

    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [{"content": "Paris", "embedding": vec(2)}])
    await store.record_entity_mentions(tenant, chunks1[0], list(node_ids.values()))
    await store.record_entity_mentions(tenant, chunks2[0], list(node_ids.values()))

    await store.delete_document(tenant, "ns", "doc-1")

    docs = await store.get_mentioning_documents(tenant, "ns", node_ids["Paris"])
    assert len(docs) == 1 and docs[0]["source_id"] == "doc-2"

    nodes = await store.vector_search_nodes(tenant, "ns", vec(2), top_k=10)
    assert any(n["content"] == "Paris" for n in nodes), "entity must survive one document's deletion"


@pytest.mark.asyncio
async def test_numbered_identifiers_are_never_fuzzy_merged(store):
    tenant = tid()
    first = await store.resolve_and_upsert_nodes(
        tenant,
        "ns",
        [{"content": "checkout-service-001", "embedding": vec(0)}],
    )
    second = await store.resolve_and_upsert_nodes(
        tenant,
        "ns",
        [{"content": "checkout-service-002", "embedding": vec(0)}],
    )

    assert first["checkout-service-001"] != second["checkout-service-002"]


# ----------------------------------------------------------------------
# Facade: PostgresGraphRAG.for_tenant()
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_for_tenant_facade_end_to_end_with_mocked_extractor(store):
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(
        postgres_url=POSTGRES_URL,
        openai_api_key="test",
        config=config,
        runtime_url=_runtime_url(),
    )
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Apple", predicate="released", object="M4")]
    )

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)

    tenant_a, tenant_b = tid(), tid()
    engine_a = rag.for_tenant(tenant_a)
    engine_b = rag.for_tenant(tenant_b)

    report = await engine_a.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")
    assert report == {
        "skipped": False, "chunks": 1, "triplets": 1,
        "extraction_status": "ready", "status_applied": True,
    }

    report_again = await engine_a.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")
    assert report_again["skipped"] is True

    result = await engine_a.retrieve("What did Apple release?", namespace="ns")
    assert any("Apple" in c.content for c in result.chunks)
    assert any(n.content == "Apple" for n in result.nodes)

    result_b = await engine_b.retrieve("What did Apple release?", namespace="ns")
    assert result_b.chunks == [] and result_b.nodes == []

    await rag.close()


# ----------------------------------------------------------------------
# v0.3: Administrative entity resolution corrections
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_entities_repoints_edges_and_mentions(store):
    tenant = tid()
    doc = await store.upsert_document(tenant, "ns", "doc-1", "h1", {})
    chunk_ids = await store.replace_chunks(tenant, doc["id"], "ns", [
        {"content": "Elon founded SpaceX. Elon Musk also leads Tesla.", "embedding": vec(0)},
    ])
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "Elon", "embedding": vec(0)},
        {"content": "Elon Musk", "embedding": vec(1)},
        {"content": "SpaceX", "embedding": vec(2)},
        {"content": "Tesla", "embedding": vec(3)},
    ])
    await store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids["Elon"], "target_id": node_ids["SpaceX"], "relation": "founded"},
        {"source_id": node_ids["Elon Musk"], "target_id": node_ids["Tesla"], "relation": "leads"},
    ])
    await store.record_entity_mentions(tenant, chunk_ids[0], list(node_ids.values()))

    result = await store.merge_entities(tenant, "ns", source_content="Elon", target_content="Elon Musk")
    assert result["merged"] is True

    remaining = await store.vector_search_nodes(tenant, "ns", vec(0), top_k=10)
    contents = {n["content"] for n in remaining}
    assert "Elon" not in contents
    assert "Elon Musk" in contents

    graph = await store.traverse_graph(tenant, [node_ids["Elon Musk"]], namespace="ns", max_hops=1, directed=True)
    targets = {e["target_content"] for e in graph["edges"]}
    assert targets == {"SpaceX", "Tesla"}

    docs = await store.get_mentioning_documents(tenant, "ns", node_ids["Elon Musk"])
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_merge_entities_accumulates_history_across_chained_merges(store):
    """A second merge into an already-merged-into node must not overwrite
    the first merge's `_merged_from` record — both must be preserved."""
    tenant = tid()
    await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "Elon", "embedding": vec(0)},
        {"content": "Elon Musk", "embedding": vec(1)},
        {"content": "Elon R. Musk", "embedding": vec(2)},
    ])

    await store.merge_entities(tenant, "ns", source_content="Elon", target_content="Elon Musk")
    await store.merge_entities(tenant, "ns", source_content="Elon R. Musk", target_content="Elon Musk")

    remaining = await store.vector_search_nodes(tenant, "ns", vec(1), top_k=10)
    target_node = next(n for n in remaining if n["content"] == "Elon Musk")
    assert target_node["metadata"]["_merged_from"] == ["Elon", "Elon R. Musk"]


@pytest.mark.asyncio
async def test_merge_entities_requires_both_to_exist(store):
    tenant = tid()
    await store.resolve_and_upsert_nodes(tenant, "ns", [{"content": "OnlyOne", "embedding": vec(0)}])
    with pytest.raises(ValueError):
        await store.merge_entities(tenant, "ns", source_content="OnlyOne", target_content="DoesNotExist")


# ----------------------------------------------------------------------
# v0.4: Community detection
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_connected_nodes_merge_into_one_community(store):
    """Regression test for a real bug: fully-synchronous label propagation
    oscillates forever on a 2-node/1-edge graph (each node only ever sees
    the *other's* label as a candidate and swaps every round). Fixed by
    switching to asynchronous (Gauss-Seidel) sequential updates."""
    from postgres_graph_rag.communities import CommunityEngine

    tenant = tid()
    engine = CommunityEngine(store)
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "Marie Curie", "embedding": vec(0)},
        {"content": "Pierre Curie", "embedding": vec(1)},
    ])
    await store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids["Marie Curie"], "target_id": node_ids["Pierre Curie"], "relation": "collaborated_with"},
    ])

    report = await engine.refresh_communities(tenant, "ns")
    assert report["skipped"] is False
    assert report["converged"] is True

    communities = await engine.list_communities(tenant, "ns")
    assert len(communities) == 1
    assert set(communities[0]["members"]) == {"Marie Curie", "Pierre Curie"}


@pytest.mark.asyncio
async def test_refresh_communities_reports_non_convergence(store):
    """A connected multi-node graph needs at least one full settling pass
    after the pass that actually changes labels before `any_changed` can be
    False, so capping `max_iterations=1` deterministically forces
    `converged=False` regardless of node-id ordering — verified empirically
    (this same 4-node ring reaches converged=True by iteration 2 when given
    room). Uses a distinct topology from the 2-node convergence-regression
    fixture above so this test isn't accidentally exercising the same
    oscillation fix."""
    from postgres_graph_rag.communities import CommunityEngine

    tenant = tid()
    engine = CommunityEngine(store)
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "R1", "embedding": vec(0)},
        {"content": "R2", "embedding": vec(1)},
        {"content": "R3", "embedding": vec(2)},
        {"content": "R4", "embedding": vec(3)},
    ])
    ring_edges = [("R1", "R2"), ("R2", "R3"), ("R3", "R4"), ("R4", "R1")]
    await store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids[a], "target_id": node_ids[b], "relation": "link"}
        for a, b in ring_edges
    ])

    result = await engine.refresh_communities(tenant, "ns", force=True, max_iterations=1)
    assert result["skipped"] is False
    assert result["converged"] is False
    assert result["iterations"] == 1


@pytest.mark.asyncio
async def test_dense_clusters_with_weak_bridge_stay_separated(store):
    from postgres_graph_rag.communities import CommunityEngine

    tenant = tid()
    engine = CommunityEngine(store)
    names = ["A1", "A2", "A3", "B1", "B2", "B3"]
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": n, "embedding": vec(i)} for i, n in enumerate(names)
    ])
    edges = [
        ("A1", "A2", 5.0), ("A2", "A3", 5.0), ("A1", "A3", 5.0),
        ("B1", "B2", 5.0), ("B2", "B3", 5.0), ("B1", "B3", 5.0),
        ("A1", "B1", 0.2),
    ]
    await store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids[a], "target_id": node_ids[b], "relation": "link", "weight": w}
        for a, b, w in edges
    ])

    await engine.refresh_communities(tenant, "ns")
    communities = await engine.list_communities(tenant, "ns")
    member_sets = [set(c["members"]) for c in communities]
    assert {"A1", "A2", "A3"} in member_sets
    assert {"B1", "B2", "B3"} in member_sets


@pytest.mark.asyncio
async def test_refresh_communities_skips_when_not_dirty(store):
    from postgres_graph_rag.communities import CommunityEngine

    tenant = tid()
    engine = CommunityEngine(store)
    await store.resolve_and_upsert_nodes(tenant, "ns", [{"content": "Solo", "embedding": vec(0)}])

    first = await engine.refresh_communities(tenant, "ns")
    assert first["skipped"] is False
    second = await engine.refresh_communities(tenant, "ns")
    assert second["skipped"] is True
    forced = await engine.refresh_communities(tenant, "ns", force=True)
    assert forced["skipped"] is False


@pytest.mark.asyncio
async def test_summarize_communities_reuses_cache_and_query_global(store):
    from unittest.mock import AsyncMock

    from postgres_graph_rag.communities import CommunityEngine

    tenant = tid()
    engine = CommunityEngine(store)
    doc = await store.upsert_document(tenant, "ns", "doc-1", "h1", {})
    chunk_ids = await store.replace_chunks(tenant, doc["id"], "ns", [
        {"content": "Marie Curie and Pierre Curie researched radioactivity together.", "embedding": vec(0)},
    ])
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "Marie Curie", "embedding": vec(0)},
        {"content": "Pierre Curie", "embedding": vec(1)},
    ])
    await store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids["Marie Curie"], "target_id": node_ids["Pierre Curie"], "relation": "collaborated_with"},
    ])
    await store.record_entity_mentions(tenant, chunk_ids[0], list(node_ids.values()))
    await engine.refresh_communities(tenant, "ns")

    extractor = AsyncMock()
    extractor.config = {"extraction_model": "gpt-5-nano-2025-08-07"}
    extractor.generate_text = AsyncMock(return_value="Marie and Pierre Curie researched radioactivity together.")

    summaries = await engine.summarize_communities(tenant, "ns", extractor)
    assert len(summaries) == 1 and summaries[0]["reused"] is False
    summaries_again = await engine.summarize_communities(tenant, "ns", extractor)
    assert summaries_again[0]["reused"] is True
    extractor.generate_text.assert_called_once()

    global_result = await engine.query_global(tenant, "ns", "Who researched radioactivity?")
    assert len(global_result) == 1


@pytest.mark.asyncio
async def test_community_detection_is_tenant_scoped(store):
    """Two tenants with identically-shaped graphs must get independent
    community runs — no cross-tenant leakage in run history either."""
    from postgres_graph_rag.communities import CommunityEngine

    tenant_a, tenant_b = tid(), tid()
    engine = CommunityEngine(store)
    for tenant in (tenant_a, tenant_b):
        ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
            {"content": "X", "embedding": vec(0)}, {"content": "Y", "embedding": vec(1)},
        ])
        await store.upsert_edges(tenant, "ns", [{"source_id": ids["X"], "target_id": ids["Y"], "relation": "r"}])

    await engine.refresh_communities(tenant_a, "ns")
    run_b = await engine.latest_run(tenant_b, "ns")
    assert run_b is None, "tenant B must not see tenant A's community run"


# ----------------------------------------------------------------------
# v0.4: Observability
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_bus_fans_out_to_multiple_sinks(store):
    from postgres_graph_rag.observability import Event, EventBus

    received_by_sink1 = []
    received_by_sink2 = []

    class Sink1:
        def emit(self, event):
            received_by_sink1.append(event.kind)

    class Sink2:
        async def emit(self, event):
            received_by_sink2.append(event.kind)

    bus = EventBus(sinks=[Sink1(), Sink2()])
    await bus.emit(Event(kind="test.kind", correlation_id="c1"))
    assert received_by_sink1 == ["test.kind"]
    assert received_by_sink2 == ["test.kind"]


@pytest.mark.asyncio
async def test_event_bus_sink_failure_does_not_propagate(store):
    from postgres_graph_rag.observability import Event, EventBus

    class BrokenSink:
        def emit(self, event):
            raise RuntimeError("sink is broken")

    bus = EventBus(sinks=[BrokenSink()])
    await bus.emit(Event(kind="test.kind", correlation_id="c1"))  # must not raise


@pytest.mark.asyncio
async def test_ingestion_and_retrieval_emit_usage_events(store):
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG
    from postgres_graph_rag.observability import EventBus, UsageAggregator

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(
        postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url(),
    )
    rag.extractor.extract_triplets = AsyncMock(return_value=[Triplet(subject="Apple", predicate="released", object="M4")])

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)

    usage = UsageAggregator()
    bus = EventBus(sinks=[usage])
    tenant = tid()
    engine = rag.for_tenant(tenant, event_bus=bus)

    await engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")
    await engine.retrieve("What did Apple release?", namespace="ns")

    snapshot = usage.snapshot(str(tenant), "ns")
    assert snapshot["counts"]["ingestion.completed"] == 1
    assert snapshot["counts"]["retrieval.completed"] == 1
    assert snapshot["counts"]["extraction.completed"] == 1

    await rag.close()


# ----------------------------------------------------------------------
# v0.5: MCP server
# ----------------------------------------------------------------------

mcp = pytest.importorskip("mcp", reason="mcp SDK not installed")


@pytest_asyncio.fixture
async def mcp_rag(store):
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(
        postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url(),
    )
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Apple", predicate="released", object="M4")]
    )

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)
    yield rag
    await rag.close()


async def _call(session, name, args):
    result = await session.call_tool(name, args)
    return result.structured_content["result"]


# ----------------------------------------------------------------------
# LEGACY_DELETION_PLAN.md R3: for_tenant() and the MCP lifespan must share
# one store-construction path (PostgresGraphRAG._get_or_create_store()),
# not two independent copies that can diverge on the error path.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_lifespan_and_for_tenant_share_the_same_store(mcp_rag):
    from mcp import ClientSession
    from mcp.client._memory import InMemoryTransport

    from postgres_graph_rag.mcp_server import build_server

    server = build_server(mcp_rag, stdio_tenant_id=tid(), enable_mutations=False)
    transport = InMemoryTransport(server)
    async with transport._connect() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            store_from_lifespan = mcp_rag._secure_store

    engine = mcp_rag.for_tenant(tid())
    assert engine._store is store_from_lifespan, (
        "for_tenant() must reuse the exact store instance the MCP lifespan "
        "already constructed, not build a second one"
    )


@pytest.mark.asyncio
async def test_for_tenant_and_mcp_lifespan_raise_the_same_error_without_runtime_url():
    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.mcp_server import build_server

    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", runtime_url=None)

    with pytest.raises(ValueError, match="runtime_url"):
        rag.for_tenant(tid())

    server = build_server(rag, stdio_tenant_id=tid(), enable_mutations=False)
    from mcp import ClientSession
    from mcp.client._memory import InMemoryTransport

    transport = InMemoryTransport(server)
    # The lifespan runs inside an anyio task group, which wraps the raised
    # ValueError in an exception group rather than letting it propagate
    # bare -- unwrap it (via `.exceptions`, duck-typed rather than naming
    # `BaseExceptionGroup` directly, since that name is only a builtin on
    # Python >= 3.11 and this project supports >= 3.10) to confirm it's the
    # same missing-runtime_url error for_tenant() raises, not just
    # "something failed."
    with pytest.raises(Exception) as exc_info:
        async with transport._connect() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
    inner = exc_info.value.exceptions[0] if hasattr(exc_info.value, "exceptions") else exc_info.value
    assert isinstance(inner, ValueError)
    assert "runtime_url" in str(inner)


@pytest.mark.asyncio
async def test_mcp_read_only_tools_available_without_mutations(mcp_rag):
    from mcp import ClientSession
    from mcp.client._memory import InMemoryTransport

    from postgres_graph_rag.mcp_server import build_server

    server = build_server(mcp_rag, stdio_tenant_id=tid(), enable_mutations=False)
    transport = InMemoryTransport(server)
    async with transport._connect() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            assert {"retrieve", "get_entity", "find_paths", "list_communities", "capabilities", "health"} <= tools
            assert "ingest_documents" not in tools
            assert "merge_entities" not in tools


@pytest.mark.asyncio
async def test_mcp_ingest_and_retrieve_round_trip(mcp_rag):
    from mcp import ClientSession
    from mcp.client._memory import InMemoryTransport

    from postgres_graph_rag.mcp_server import build_server

    tenant = tid()
    server = build_server(mcp_rag, stdio_tenant_id=tenant, enable_mutations=True)
    transport = InMemoryTransport(server)
    async with transport._connect() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            ingest = await _call(session, "ingest_documents", {
                "documents": [{"source_id": "doc-1", "text": "Apple released the M4 chip."}],
                "namespace": "ns",
            })
            assert ingest["results"][0]["skipped"] is False

            result = await _call(session, "retrieve", {"question": "What did Apple release?", "namespace": "ns"})
            assert any("Apple" in c["content"] for c in result["chunks"])

            entity = await _call(session, "get_entity", {"name": "Apple", "namespace": "ns"})
            assert entity["found"] is True


@pytest.mark.asyncio
async def test_mcp_stdio_tenants_are_isolated(mcp_rag):
    """Two servers built with different stdio_tenant_id values must not
    see each other's data, even though they share the same underlying
    SecureGraphStore/connection pool."""
    from mcp import ClientSession
    from mcp.client._memory import InMemoryTransport

    from postgres_graph_rag.mcp_server import build_server

    tenant_a, tenant_b = tid(), tid()
    server_a = build_server(mcp_rag, stdio_tenant_id=tenant_a, enable_mutations=True)
    transport_a = InMemoryTransport(server_a)
    async with transport_a._connect() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await _call(session, "ingest_documents", {
                "documents": [{"source_id": "doc-1", "text": "Apple released the M4 chip."}],
                "namespace": "ns",
            })

    server_b = build_server(mcp_rag, stdio_tenant_id=tenant_b, enable_mutations=False)
    transport_b = InMemoryTransport(server_b)
    async with transport_b._connect() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await _call(session, "retrieve", {"question": "What did Apple release?", "namespace": "ns"})
            assert result["chunks"] == []


@pytest.mark.asyncio
async def test_mcp_http_refuses_unauthenticated_public_bind(mcp_rag):
    from postgres_graph_rag.mcp_server import run_http

    with pytest.raises(ValueError):
        await run_http(mcp_rag)  # no resolver, no dev flag
    with pytest.raises(ValueError):
        await run_http(mcp_rag, host="0.0.0.0", allow_unauthenticated_dev=True)  # dev flag but public bind


# ----------------------------------------------------------------------
# Priority-3 fixes: retry-safety, source_id, metadata filtering, document
# update/delete lifecycle
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_extraction_leaves_chunk_durably_retrievable(store):
    """Regression test: chunks must be written before extraction is
    attempted, so a chunk whose extraction fails still has a row to find
    and retry later — not just retrievable via hybrid search text, but
    actually present in document_chunks at all."""
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    ingestion_config = {"max_extraction_retries": 1}
    rag = PostgresGraphRAG(
        postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url(),
        ingestion_config=ingestion_config,
    )
    rag.extractor.extract_triplets = AsyncMock(side_effect=RuntimeError("simulated outage"))

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)

    tenant = tid()
    engine = rag.for_tenant(tenant)
    report = await engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")
    assert report["skipped"] is False
    assert report["triplets"] == 0

    result = await engine.retrieve("What did Apple release?", namespace="ns")
    assert any("Apple" in c.content for c in result.chunks)
    assert result.nodes == []

    await rag.close()


@pytest.mark.asyncio
async def test_retry_failed_chunks_recovers_without_original_text(store):
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    ingestion_config = {"max_extraction_retries": 1}
    rag = PostgresGraphRAG(
        postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url(),
        ingestion_config=ingestion_config,
    )

    call_count = {"n": 0}

    async def flaky_extract(chunk):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated outage")
        return [Triplet(subject="Apple", predicate="released", object="M4")]

    rag.extractor.extract_triplets = AsyncMock(side_effect=flaky_extract)

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)

    tenant = tid()
    engine = rag.for_tenant(tenant)
    await engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")

    retry_report = await engine.retry_failed_chunks(namespace="ns")
    assert retry_report == {"retried": 1, "triplets": 1}

    result = await engine.retrieve("What did Apple release?", namespace="ns")
    assert any(n.content == "Apple" for n in result.nodes)

    # Nothing left to retry now.
    retry_again = await engine.retry_failed_chunks(namespace="ns")
    assert retry_again == {"retried": 0, "triplets": 0}

    await rag.close()


@pytest.mark.asyncio
async def test_embedding_failure_does_not_commit_hash_and_retry_recovers(store):
    """Regression test for the retry-hole bug: content_hash must never be
    persisted ahead of its chunks. If embedding fails partway through
    add_document(), the document's stored hash must stay at whatever it was
    before this call (nonexistent, for a first ingestion) so a retry with
    the *same* text is treated as new content rather than silently skipped
    with content_changed=False and no chunk rows to recover."""
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    call_count = {"n": 0}

    async def flaky_embed(text):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated embedding provider outage")
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=flaky_embed)
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Apple", predicate="released", object="M4")]
    )

    tenant = tid()
    engine = rag.for_tenant(tenant)

    with pytest.raises(RuntimeError, match="simulated embedding provider outage"):
        await engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")

    # No hash was ever committed for the failed attempt, and no chunk rows
    # exist to retry_failed_chunks() against.
    stored_hash = await store.get_document_content_hash(tenant, "ns", "doc-1")
    assert stored_hash is None
    retry_report = await engine.retry_failed_chunks(namespace="ns")
    assert retry_report == {"retried": 0, "triplets": 0}

    # Retrying add_document() with the identical text must actually run
    # (not skip), because the prior attempt never published a hash.
    report = await engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")
    assert report["skipped"] is False
    assert report["chunks"] == 1
    assert report["triplets"] == 1

    result = await engine.retrieve("What did Apple release?", namespace="ns")
    assert any("Apple" in c.content for c in result.chunks)

    await rag.close()


@pytest.mark.asyncio
async def test_update_failure_preserves_old_revision(store):
    """Failure-injection test: an *update* (not a first ingestion) whose
    embedding fails must leave the previous, already-published revision
    fully intact and retrievable — and a subsequent retry with the same new
    text must actually publish it, not skip."""
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG
    from postgres_graph_rag.tenancy import content_hash

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    call_count = {"n": 0}

    async def flaky_embed(text):
        call_count["n"] += 1
        if call_count["n"] == 3:  # 1st and 2nd calls publish revision A cleanly
            raise RuntimeError("simulated outage during revision B")
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=flaky_embed)
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Apple", predicate="released", object="M4")]
    )

    tenant = tid()
    engine = rag.for_tenant(tenant)
    report_a = await engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")
    assert report_a["skipped"] is False

    with pytest.raises(RuntimeError, match="simulated outage during revision B"):
        await engine.add_document("Apple released the M5 chip.", namespace="ns", source_id="doc-1")

    # Revision A's hash and chunk must still be exactly what they were.
    stored_hash = await store.get_document_content_hash(tenant, "ns", "doc-1")
    assert stored_hash == content_hash("Apple released the M4 chip.")
    result = await engine.retrieve("What did Apple release?", namespace="ns", top_k=10)
    contents = [c.content for c in result.chunks]
    assert "Apple released the M4 chip." in contents
    assert "Apple released the M5 chip." not in contents

    # Retrying with the same new text must now actually publish it.
    report_b = await engine.add_document("Apple released the M5 chip.", namespace="ns", source_id="doc-1")
    assert report_b["skipped"] is False
    result2 = await engine.retrieve("What did Apple release?", namespace="ns", top_k=10)
    contents2 = [c.content for c in result2.chunks]
    assert "Apple released the M5 chip." in contents2
    assert "Apple released the M4 chip." not in contents2

    await rag.close()


@pytest.mark.asyncio
async def test_chunk_replacement_failure_rolls_back_hash_too(store):
    """If replace_chunks() fails, upsert_document()'s hash write in the same
    transaction must roll back with it, rather than leaving a hash that
    points at chunks which were never (re)written."""
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG
    from postgres_graph_rag.tenancy import content_hash

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Apple", predicate="released", object="M4")]
    )

    tenant = tid()
    engine = rag.for_tenant(tenant)
    await engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")

    original_replace_chunks = engine._store.replace_chunks

    async def flaky_replace_chunks(*args, **kwargs):
        raise RuntimeError("simulated chunk-insert failure")

    engine._store.replace_chunks = flaky_replace_chunks
    try:
        with pytest.raises(RuntimeError, match="simulated chunk-insert failure"):
            await engine.add_document("Apple released the M5 chip.", namespace="ns", source_id="doc-1")
    finally:
        engine._store.replace_chunks = original_replace_chunks

    stored_hash = await store.get_document_content_hash(tenant, "ns", "doc-1")
    assert stored_hash == content_hash("Apple released the M4 chip.")

    result = await engine.retrieve("What did Apple release?", namespace="ns")
    assert any("M4" in c.content for c in result.chunks)

    await rag.close()


@pytest.mark.asyncio
async def test_concurrent_identical_ingestion_no_duplicate_chunks(store):
    """Two workers racing to ingest the *same* content for the same
    source_id must not both publish: exactly one performs the write (and
    extraction), the other observes content_changed=False and skips."""
    import asyncio
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Apple", predicate="released", object="M4")]
    )

    tenant = tid()
    engine = rag.for_tenant(tenant)
    reports = await asyncio.gather(
        engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1"),
        engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1"),
    )
    assert sorted(r["skipped"] for r in reports) == [False, True]

    result = await engine.retrieve("What did Apple release?", namespace="ns", top_k=10)
    contents = [c.content for c in result.chunks]
    assert contents.count("Apple released the M4 chip.") == 1

    await rag.close()


@pytest.mark.asyncio
async def test_concurrent_different_content_updates_yield_one_consistent_winner(store):
    """Two workers racing to publish *different* content for the same
    source_id must not corrupt each other: the final state (hash + chunks)
    must match exactly one of the two revisions, never a mix of both."""
    import asyncio
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG
    from postgres_graph_rag.tenancy import content_hash

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Apple", predicate="released", object="M4")]
    )

    tenant = tid()
    engine = rag.for_tenant(tenant)
    text_b = "Apple released the M5 chip."
    text_c = "Apple released the M6 chip."
    await asyncio.gather(
        engine.add_document(text_b, namespace="ns", source_id="doc-1"),
        engine.add_document(text_c, namespace="ns", source_id="doc-1"),
    )

    stored_hash = await store.get_document_content_hash(tenant, "ns", "doc-1")
    assert stored_hash in (content_hash(text_b), content_hash(text_c))

    result = await engine.retrieve("What did Apple release?", namespace="ns", top_k=10)
    contents = {c.content for c in result.chunks}
    assert contents == {text_b} or contents == {text_c}

    await rag.close()


@pytest.mark.asyncio
async def test_stale_extraction_does_not_wire_facts_to_superseded_chunks(store):
    """The stale-extraction race: revision B's chunks are durably stored and
    its (slow, out-of-transaction) extraction is still in flight when a
    concurrent revision C replaces B's chunks outright (replace_chunks
    deletes the old rows). When B's extraction finally completes, wiring it
    must not attach entity/edge mentions to chunk ids that no longer belong
    to the current revision, and must not leave an unsupported (zero-mention)
    edge behind either.

    Exercised directly against the engine's internal _wire_entities (rather
    than via real concurrent add_document() calls) so the race window is
    deterministic instead of timing-dependent — this is precisely the
    boundary filter_existing_chunk_ids()/prune_unsupported_edges() guard.
    """
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)

    tenant = tid()
    engine = rag.for_tenant(tenant)

    doc = await store.upsert_document(tenant, "ns", "doc-1", "hash-b", {})
    stale_chunk_ids = await store.replace_chunks(
        tenant, doc["id"], "ns", [{"content": "Revision B text.", "embedding": vec(0)}],
    )

    # A concurrent, newer revision C replaces the chunks outright before B's
    # extraction (below) gets a chance to wire its results.
    await store.upsert_document(tenant, "ns", "doc-1", "hash-c", {})
    await store.replace_chunks(
        tenant, doc["id"], "ns", [{"content": "Revision C text.", "embedding": vec(1)}],
    )

    # B's extraction "finally completes" and tries to wire its results
    # against the now-stale chunk ids.
    triplets = [{"subject": "Widget", "predicate": "madeBy", "object": "Acme"}]
    total = await engine._wire_entities(
        stale_chunk_ids, ["Revision B text."], [triplets], "ns", {}, "corr-1",
    )
    assert total == 0

    # Nothing was attached to the stale chunk, and no zero-support edge from
    # this attempt was left behind for later traversal to pick up.
    async with store.tenant_connection(tenant) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT count(*) AS n FROM {SCHEMA}.graph_edges WHERE tenant_id = %s AND support_count = 0",
                (str(tenant),),
            )
            row = await cur.fetchone()
    assert row["n"] == 0

    await rag.close()


@pytest.mark.asyncio
async def test_stale_extraction_does_not_overwrite_newer_revisions_status(store):
    """A companion race to the one above, at the status layer instead of the
    mentions layer: revision B's extraction is still running (status still
    'pending' for B) when revision C is published, which resets status to
    'pending' for C too. filter_existing_chunk_ids() already stops B's
    *facts* from attaching to C's chunks — but nothing stops B's *status
    verdict* from being written to the shared document row afterward unless
    set_extraction_status()'s expected_content_hash guard is used. Confirms
    that guard: B's late, stale "ready" write must be a no-op once C is the
    current revision, leaving C's own status (still 'pending', since this
    test never runs C's extraction) untouched.
    """
    from postgres_graph_rag.tenancy import content_hash

    hash_b, hash_c = content_hash("Revision B text."), content_hash("Revision C text.")
    doc = await store.upsert_document(tid_b := tid(), "ns", "doc-1", hash_b, {})
    tenant = tid_b
    await store.replace_chunks(
        tenant, doc["id"], "ns", [{"content": "Revision B text.", "embedding": vec(0)}],
    )

    # C is published while B's extraction is still "in flight" (from this
    # test's point of view, simply not yet having called set_extraction_status
    # for B) — the upsert_document CASE logic resets status to 'pending' for C.
    await store.upsert_document(tenant, "ns", "doc-1", hash_c, {})
    await store.replace_chunks(
        tenant, doc["id"], "ns", [{"content": "Revision C text.", "embedding": vec(1)}],
    )

    # B's extraction "finishes late" and tries to record its (stale) verdict.
    updated = await store.set_extraction_status(
        tenant, doc["id"], "ready", expected_content_hash=hash_b,
    )
    assert updated is False

    stored_hash = await store.get_document_content_hash(tenant, "ns", "doc-1")
    assert stored_hash == hash_c
    async with store.tenant_connection(tenant) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT extraction_status FROM {SCHEMA}.documents WHERE tenant_id=%s AND id=%s",
                (str(tenant), doc["id"]),
            )
            row = await cur.fetchone()
    assert row["extraction_status"] == "pending"


@pytest.mark.asyncio
async def test_set_extraction_status_rejects_invalid_status(store):
    doc = await store.upsert_document(tenant := tid(), "ns", "doc-1", "hash-a", {})
    with pytest.raises(ValueError, match="status must be one of"):
        await store.set_extraction_status(tenant, doc["id"], "done")


@pytest.mark.asyncio
async def test_retry_sweep_binds_status_to_hash_captured_at_fetch_time(store):
    """retry_failed_chunks() must bind its final status write to the
    document hash captured when it fetched retry candidates
    (document_content_hash from find_unfinished_chunks), not one re-read
    right before the write. Re-reading "whatever's current" at write time
    would trivially match a concurrently-published newer revision and
    overwrite *its* status with a verdict about the old revision's chunks —
    exactly the bug expected_content_hash exists to prevent.

    Revision C's publication is simulated as a side effect wrapped around
    _wire_entities, landing it deterministically between candidate capture
    (which happens before extraction runs) and the final status write
    (which happens after wiring) — precisely the window described above.
    """
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG
    from postgres_graph_rag.tenancy import content_hash

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Apple", predicate="released", object="M4")]
    )

    tenant = tid()
    engine = rag.for_tenant(tenant)

    hash_b = content_hash("Revision B text.")
    hash_c = content_hash("Revision C text.")

    doc = await store.upsert_document(tenant, "ns", "doc-1", hash_b, {})
    await store.replace_chunks(
        tenant, doc["id"], "ns", [{"content": "Revision B text.", "embedding": vec(0)}],
    )
    # B's chunk's extraction is left genuinely unfinished (no
    # chunk_extractions row at all yet), so find_unfinished_chunks() picks
    # it up as a retry candidate.

    original_wire_entities = engine._wire_entities

    async def wire_then_publish_c(*args, **kwargs):
        result = await original_wire_entities(*args, **kwargs)
        # A concurrent add_document() publishes revision C right after B's
        # extraction/wiring finished but before retry_failed_chunks() gets
        # to its own status-finalization loop.
        await store.upsert_document(tenant, "ns", "doc-1", hash_c, {})
        await store.replace_chunks(
            tenant, doc["id"], "ns", [{"content": "Revision C text.", "embedding": vec(1)}],
        )
        return result

    engine._wire_entities = wire_then_publish_c

    retry_report = await engine.retry_failed_chunks(namespace="ns")
    assert retry_report["retried"] == 1

    # C must still be the current revision, with its own (still 'pending',
    # since this test never runs C's own extraction) status untouched by
    # B's late-arriving, now-stale verdict.
    stored_hash = await store.get_document_content_hash(tenant, "ns", "doc-1")
    assert stored_hash == hash_c
    async with store.tenant_connection(tenant) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT extraction_status FROM {SCHEMA}.documents WHERE tenant_id=%s AND id=%s",
                (str(tenant), doc["id"]),
            )
            row = await cur.fetchone()
    assert row["extraction_status"] == "pending"

    await rag.close()


@pytest.mark.asyncio
async def test_traverse_graph_induced_edges_apply_same_filters_as_expansion(store):
    """Adversarial test for the induced-edge query in traverse_graph(): a
    zero-support edge between two nodes that were each reached via other,
    supported paths must not be silently reintroduced into the returned
    edge set just because both of its endpoints happen to be present.

    Graph: A --supported--> B --supported--> C, plus a zero-support
    A --> C edge that was never actually backed by any mention. Traversing
    from A must return all three nodes (B and C are reachable via the
    supported edges) but only the two supported edges — not the
    zero-support shortcut between A and C.
    """
    tenant = tid()
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "A", "embedding": vec(0)},
        {"content": "B", "embedding": vec(1)},
        {"content": "C", "embedding": vec(2)},
    ])
    doc = await store.upsert_document(tenant, "ns", "doc-1", "h1", {})
    chunk_ids = await store.replace_chunks(tenant, doc["id"], "ns", [
        {"content": "A relates_to B. B relates_to C.", "embedding": vec(0)},
    ])
    edge_ids = await store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids["A"], "target_id": node_ids["B"], "relation": "relates_to"},
        {"source_id": node_ids["B"], "target_id": node_ids["C"], "relation": "relates_to"},
    ], evidence_backed=True)
    await store.record_edge_mentions(tenant, chunk_ids[0], list(edge_ids.values()))

    # A zero-support edge: created but never backed by any mention (the
    # exact leftover shape prune_unsupported_edges() targets, constructed
    # directly here to test the read-path filter independently of it).
    await store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids["A"], "target_id": node_ids["C"], "relation": "relates_to"},
    ], evidence_backed=True)

    result = await store.traverse_graph(tenant, [node_ids["A"]], namespace="ns", max_hops=2, directed=True)
    returned_node_contents = {n["content"] for n in result["nodes"]}
    assert returned_node_contents == {"A", "B", "C"}

    returned_edges = {(e["source_content"], e["target_content"]) for e in result["edges"]}
    assert returned_edges == {("A", "B"), ("B", "C")}
    assert ("A", "C") not in returned_edges


@pytest.mark.asyncio
async def test_add_record_failure_rolls_back_entity_node_too(store):
    """If chunk publication fails after the entity node was resolved, the
    node must roll back with it (both happen in one transaction now) — no
    orphan entity should become visible without the document/mention that
    was supposed to back it."""
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)

    tenant = tid()
    engine = rag.for_tenant(tenant)

    async def flaky_replace_chunks(*args, **kwargs):
        raise RuntimeError("simulated failure publishing record's chunk")

    engine._store.replace_chunks = flaky_replace_chunks

    with pytest.raises(RuntimeError, match="simulated failure publishing record's chunk"):
        await engine.add_record(
            namespace="ns", source_id="rec-1", entity_type="company",
            record={"name": "Acme Corp", "founded": 1990},
        )

    async with store.tenant_connection(tenant) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT count(*) AS n FROM {SCHEMA}.graph_nodes WHERE tenant_id = %s",
                (str(tenant),),
            )
            row = await cur.fetchone()
    assert row["n"] == 0

    await rag.close()


@pytest.mark.asyncio
async def test_partial_wiring_failure_leaves_no_unsupported_edge(store):
    """Inject a failure after the edge is created (upsert_edges) but before
    its evidence mention is recorded: the edge must not be left permanently
    traversable with zero support (traversal's min_weight defaults to 0.0,
    so a zero-weight edge would otherwise pass through undetected)."""
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Apple", predicate="released", object="M4")]
    )

    tenant = tid()
    engine = rag.for_tenant(tenant)

    async def flaky_record_edge_mentions(*args, **kwargs):
        raise RuntimeError("simulated failure recording edge mention")

    engine._store.record_edge_mentions = flaky_record_edge_mentions

    with pytest.raises(RuntimeError, match="simulated failure recording edge mention"):
        await engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")

    async with store.tenant_connection(tenant) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT count(*) AS n FROM {SCHEMA}.graph_edges WHERE tenant_id = %s AND support_count = 0",
                (str(tenant),),
            )
            row = await cur.fetchone()
    assert row["n"] == 0

    await rag.close()


@pytest.mark.asyncio
async def test_retrieval_returns_real_source_id_not_none(store):
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())
    rag.extractor.extract_triplets = AsyncMock(return_value=[Triplet(subject="Apple", predicate="released", object="M4")])

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)

    engine = rag.for_tenant(tid())
    await engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-42")
    result = await engine.retrieve("What did Apple release?", namespace="ns")
    assert result.chunks[0].source_id == "doc-42"
    assert "doc-42" in result.to_context_string()

    await rag.close()


@pytest.mark.asyncio
async def test_hybrid_search_metadata_filter(store):
    tenant = tid()
    doc_a = await store.upsert_document(tenant, "ns", "doc-a", "ha", {})
    doc_b = await store.upsert_document(tenant, "ns", "doc-b", "hb", {})
    await store.replace_chunks(tenant, doc_a["id"], "ns", [
        {"content": "The onboarding policy requires manager approval.", "embedding": vec(0), "metadata": {"source": "handbook"}},
    ])
    await store.replace_chunks(tenant, doc_b["id"], "ns", [
        {"content": "The onboarding policy also covers equipment requests.", "embedding": vec(0), "metadata": {"source": "wiki"}},
    ])

    unfiltered = await store.hybrid_search(tenant, "ns", "onboarding policy", vec(0))
    assert len(unfiltered) == 2

    filtered = await store.hybrid_search(tenant, "ns", "onboarding policy", vec(0), metadata_filter={"source": "handbook"})
    assert len(filtered) == 1
    assert filtered[0]["metadata"]["source"] == "handbook"


@pytest.mark.asyncio
async def test_traverse_graph_metadata_filter_excludes_nonmatching_nodes(store):
    tenant = tid()
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "PolicyA", "embedding": vec(1), "metadata": {"confidential": True}},
        {"content": "PolicyB", "embedding": vec(2), "metadata": {"confidential": False}},
    ])
    await store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids["PolicyA"], "target_id": node_ids["PolicyB"], "relation": "related_to"},
    ])

    unfiltered = await store.traverse_graph(tenant, list(node_ids.values()), namespace="ns", max_hops=1)
    assert {n["content"] for n in unfiltered["nodes"]} == {"PolicyA", "PolicyB"}

    filtered = await store.traverse_graph(
        tenant, list(node_ids.values()), namespace="ns", max_hops=1, metadata_filter={"confidential": False},
    )
    assert {n["content"] for n in filtered["nodes"]} == {"PolicyB"}


@pytest.mark.asyncio
async def test_document_update_replaces_chunks_and_mentions(store):
    """A re-ingestion with *changed* content must fully replace the old
    chunks/mentions, not accumulate alongside them."""
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Apple", predicate="released", object="M4")]
    )

    tenant = tid()
    engine = rag.for_tenant(tenant)
    await engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")

    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Apple", predicate="released", object="M5")]
    )
    report = await engine.add_document("Apple released the M5 chip.", namespace="ns", source_id="doc-1")
    assert report["skipped"] is False

    result = await engine.retrieve("What did Apple release?", namespace="ns", top_k=10)
    contents = [c.content for c in result.chunks]
    assert "Apple released the M5 chip." in contents
    assert "Apple released the M4 chip." not in contents


@pytest.mark.asyncio
async def test_delete_document_removes_chunks_from_retrieval(store):
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())
    rag.extractor.extract_triplets = AsyncMock(return_value=[])

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)

    tenant = tid()
    engine = rag.for_tenant(tenant)
    await engine.add_document("The vault code is 12345.", namespace="ns", source_id="doc-1")

    result_before = await engine.retrieve("What is the vault code?", namespace="ns")
    assert any("12345" in c.content for c in result_before.chunks)

    await engine.delete_document(namespace="ns", source_id="doc-1")

    result_after = await engine.retrieve("What is the vault code?", namespace="ns")
    assert result_after.chunks == []

    await rag.close()


@pytest.mark.asyncio
async def test_delete_document_removes_unsupported_extracted_edges(store):
    """Edge traversal must follow active evidence, not stale historical
    extraction counters."""
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(
        postgres_url=POSTGRES_URL,
        openai_api_key="test",
        config=config,
        runtime_url=_runtime_url(),
    )
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Service A", predicate="depends_on", object="Service B")]
    )
    rag.extractor.get_embedding = AsyncMock(
        side_effect=lambda text: [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM
    )
    tenant = tid()
    engine = rag.for_tenant(tenant)
    await engine.add_document("Service A depends on Service B.", namespace="ns", source_id="doc-1")

    nodes = await rag._secure_store.vector_search_nodes(tenant, "ns", [0.1] * DIM, top_k=10)
    service_a = next(n for n in nodes if n["content"] == "Service A")
    before = await rag._secure_store.traverse_graph(tenant, [str(service_a["id"])], "ns", max_hops=1)
    assert any(edge["relation"] == "depends_on" for edge in before["edges"])

    await engine.delete_document("ns", "doc-1")
    after = await rag._secure_store.traverse_graph(tenant, [str(service_a["id"])], "ns", max_hops=1)
    assert not any(edge["relation"] == "depends_on" for edge in after["edges"])
    await rag.close()


# ----------------------------------------------------------------------
# Migration versioning + advisory lock
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migrate_schema_records_version_idempotently(store):
    from postgres_graph_rag.tenancy import SCHEMA_VERSION

    async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT version FROM {SCHEMA}.schema_migrations")
            rows = await cur.fetchall()
    assert [r[0] for r in rows] == [SCHEMA_VERSION]

    # Re-running (the `store` fixture already did once for setup; do it
    # again explicitly) must not create a duplicate row or error.
    await migrate_schema(
        admin_url=POSTGRES_URL, runtime_role=RUNTIME_ROLE, runtime_password=RUNTIME_PASSWORD,
        embedding_dimension=DIM, migrate_legacy_data=False,
    )
    async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT count(*) FROM {SCHEMA}.schema_migrations")
            count = (await cur.fetchone())[0]
    assert count == 1


@pytest.mark.asyncio
async def test_migrate_schema_refuses_to_run_concurrently(store):
    from postgres_graph_rag.tenancy import _MIGRATION_LOCK_KEY

    holder = await psycopg.AsyncConnection.connect(POSTGRES_URL)
    async with holder.cursor() as cur:
        await cur.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_KEY,))
    try:
        with pytest.raises(RuntimeError):
            await migrate_schema(
                admin_url=POSTGRES_URL, runtime_role=RUNTIME_ROLE, runtime_password=RUNTIME_PASSWORD,
                embedding_dimension=DIM, migrate_legacy_data=False,
            )
    finally:
        async with holder.cursor() as cur:
            await cur.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_KEY,))
        await holder.close()


@pytest.mark.asyncio
async def test_migrate_schema_rejects_embedding_dimension_mismatch(store):
    from postgres_graph_rag.tenancy import SchemaCompatibilityError

    with pytest.raises(SchemaCompatibilityError, match="embedding type is incompatible"):
        await migrate_schema(
            admin_url=POSTGRES_URL,
            runtime_role=RUNTIME_ROLE,
            runtime_password=RUNTIME_PASSWORD,
            embedding_dimension=DIM + 1,
            migrate_legacy_data=False,
        )

    status = await store.schema_status()
    assert status["ready"] is True
    assert status["embedding_dimension"] == DIM


def test_migration_ddl_checksum_changes_if_ddl_shape_changes():
    """Sanity check that MIGRATION_DDL_CHECKSUM is actually derived from
    _DDL_SHAPE_STATEMENTS (not just a hardcoded value that happens to sit
    next to it) — no DB needed."""
    import hashlib

    from postgres_graph_rag.tenancy import _DDL_SHAPE_STATEMENTS, MIGRATION_DDL_CHECKSUM

    assert MIGRATION_DDL_CHECKSUM == hashlib.sha256(
        "\n".join(_DDL_SHAPE_STATEMENTS).encode("utf-8")
    ).hexdigest()

    mutated = hashlib.sha256(
        "\n".join([*_DDL_SHAPE_STATEMENTS, "ALTER TABLE foo ADD COLUMN bar TEXT"]).encode("utf-8")
    ).hexdigest()
    assert mutated != MIGRATION_DDL_CHECKSUM


# ----------------------------------------------------------------------
# Priority 1: structured/deterministic ingestion
# ----------------------------------------------------------------------


def _mock_rag():
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)
    return rag


@pytest.mark.asyncio
async def test_add_record_never_calls_the_llm_extractor(store):
    from unittest.mock import AsyncMock

    rag = _mock_rag()
    rag.extractor.extract_triplets = AsyncMock(side_effect=AssertionError("must not be called"))
    engine = rag.for_tenant(tid())

    report = await engine.add_record(
        namespace="company", source_id="employee:123", entity_type="employee",
        record={"name": "Alice", "department": "Payments", "role": "Manager"},
    )
    assert report["skipped"] is False
    assert report["entity_id"] is not None

    await rag.close()


@pytest.mark.asyncio
async def test_add_record_is_idempotent_on_unchanged_record(store):
    rag = _mock_rag()
    engine = rag.for_tenant(tid())

    record = {"name": "Alice", "department": "Payments", "role": "Manager"}
    first = await engine.add_record(namespace="company", source_id="employee:123", entity_type="employee", record=record)
    second = await engine.add_record(namespace="company", source_id="employee:123", entity_type="employee", record=record)
    assert first["skipped"] is False
    assert second["skipped"] is True

    changed = await engine.add_record(
        namespace="company", source_id="employee:123", entity_type="employee",
        record={**record, "role": "Director"},
    )
    assert changed["skipped"] is False

    await rag.close()


@pytest.mark.asyncio
async def test_add_triplets_resolves_to_same_entity_as_add_record(store):
    rag = _mock_rag()
    engine = rag.for_tenant(tid())

    await engine.add_record(
        namespace="company", source_id="employee:123", entity_type="employee",
        record={"name": "Alice", "department": "Payments"},
    )
    triplet_report = await engine.add_triplets(
        [{"subject": "Alice", "predicate": "works_in", "object": "Payments", "metadata": {"source_id": "employee:123"}}],
        namespace="company",
    )
    assert triplet_report == {"edges": 1, "entities": 2}

    result = await engine.retrieve("Who works in Payments?", namespace="company")
    assert any(n.content == "Alice" for n in result.nodes)
    assert any(e.relation == "works_in" and e.source_content == "Alice" for e in result.edges)

    await rag.close()


@pytest.mark.asyncio
async def test_add_record_is_tenant_isolated(store):
    rag = _mock_rag()
    tenant_a, tenant_b = tid(), tid()
    engine_a = rag.for_tenant(tenant_a)
    engine_b = rag.for_tenant(tenant_b)

    await engine_a.add_record(
        namespace="company", source_id="employee:123", entity_type="employee",
        record={"name": "Alice", "department": "Payments"},
    )
    result_b = await engine_b.retrieve("Who works in Payments?", namespace="company")
    assert result_b.nodes == []

    await rag.close()


@pytest.mark.asyncio
async def test_add_triplets_empty_list_is_a_noop(store):
    rag = _mock_rag()
    engine = rag.for_tenant(tid())
    report = await engine.add_triplets([], namespace="company")
    assert report == {"edges": 0, "entities": 0}
    await rag.close()


# ----------------------------------------------------------------------
# Token telemetry
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extractor_last_usage_is_concurrency_safe():
    """Regression test: last_usage must be a contextvar, not a plain
    instance attribute, or concurrent extraction calls sharing one
    LLMExtractor clobber each other's usage numbers. Verified directly on
    LLMExtractor rather than through the full ingestion pipeline, since the
    bug is about the extractor's own state management."""
    import asyncio as _asyncio
    from unittest.mock import MagicMock

    from postgres_graph_rag.extractor import ExtractionResult, LLMExtractor, Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    extractor = LLMExtractor(config=OPENAI_DEFAULT_CONFIG, openai_api_key="test")

    def make_completion(n):
        usage = MagicMock(prompt_tokens=n * 10, completion_tokens=n, total_tokens=n * 10 + n)
        parsed = ExtractionResult(triplets=[Triplet(subject=f"S{n}", predicate="p", object=f"O{n}")])
        return MagicMock(usage=usage, choices=[MagicMock(message=MagicMock(parsed=parsed))])

    call_n = {"i": 0}

    async def fake_parse(*args, **kwargs):
        call_n["i"] += 1
        n = call_n["i"]
        await _asyncio.sleep(0.05 if n == 1 else 0.01)  # first call finishes LAST
        return make_completion(n)

    extractor.openai_client = MagicMock()
    extractor.openai_client.beta.chat.completions.parse = fake_parse

    results = {}

    async def worker(n):
        await extractor.extract_triplets(f"text {n}")
        results[n] = extractor.last_usage["total_tokens"]

    await _asyncio.gather(worker(1), worker(2), worker(3))
    assert results == {1: 11, 2: 22, 3: 33}


@pytest.mark.asyncio
async def test_ingestion_and_retrieval_report_real_token_counts(store):
    from unittest.mock import MagicMock

    from postgres_graph_rag.extractor import ExtractionResult, Triplet
    from postgres_graph_rag.observability import EventBus, UsageAggregator

    rag = _mock_rag()

    async def fake_parse(*args, **kwargs):
        usage = MagicMock(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        parsed = ExtractionResult(triplets=[Triplet(subject="Apple", predicate="released", object="M4")])
        return MagicMock(usage=usage, choices=[MagicMock(message=MagicMock(parsed=parsed))])

    rag.extractor.openai_client = MagicMock()
    rag.extractor.openai_client.beta.chat.completions.parse = fake_parse

    async def fake_embed_create(input, model):
        usage = MagicMock(prompt_tokens=len(input) * 5, total_tokens=len(input) * 5)
        data = [MagicMock(embedding=[0.1] * DIM) for _ in input]
        return MagicMock(usage=usage, data=data)

    rag.extractor.openai_client.embeddings.create = fake_embed_create

    usage = UsageAggregator()
    bus = EventBus(sinks=[usage])
    tenant = tid()
    engine = rag.for_tenant(tenant, event_bus=bus)

    await engine.add_document("Apple released the M4 chip.", namespace="ns", source_id="doc-1")
    await engine.retrieve("What did Apple release?", namespace="ns")

    snap = usage.snapshot(str(tenant), "ns")
    assert snap["tokens"] > 0
    assert snap["counts"]["embedding.completed"] >= 1
    assert snap["counts"]["extraction.completed"] == 1

    await rag.close()


@pytest.mark.asyncio
async def test_google_embedding_reports_no_usage_without_crashing(store):
    """Google's EmbedContentResponse has no usage/token field at all (a
    real SDK limitation, verified against the installed types) — this must
    result in last_usage=None, not a crash or a fabricated 0."""
    from unittest.mock import MagicMock

    from postgres_graph_rag.extractor import LLMExtractor
    from postgres_graph_rag.models import GOOGLE_DEFAULT_CONFIG

    extractor = LLMExtractor(config=GOOGLE_DEFAULT_CONFIG, google_api_key="test")
    extractor.google_client = MagicMock()

    async def fake_embed_content(model, contents):
        return MagicMock(embeddings=[MagicMock(values=[0.1] * DIM) for _ in contents])

    extractor.google_client.models.embed_content = fake_embed_content

    await extractor.get_embedding(["hello"])
    assert extractor.last_usage is None


# ----------------------------------------------------------------------
# Range/comparison metadata filter DSL (against a real database)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_search_date_range_filter_matches_roadmap_example(store):
    """The literal roadmap example this DSL exists to satisfy: 'only
    traverse relationships from documents updated in the last 90 days.'
    Plain containment couldn't express this at all; this proves the range
    DSL actually can, against a real database."""
    tenant = tid()
    doc_recent = await store.upsert_document(tenant, "ns", "doc-recent", "h1", {})
    doc_old = await store.upsert_document(tenant, "ns", "doc-old", "h2", {})
    await store.replace_chunks(tenant, doc_recent["id"], "ns", [
        {"content": "The onboarding policy requires manager approval.", "embedding": vec(0),
         "metadata": {"updated_at": "2026-07-01T00:00:00Z"}},
    ])
    await store.replace_chunks(tenant, doc_old["id"], "ns", [
        {"content": "The onboarding policy used to require director approval.", "embedding": vec(0),
         "metadata": {"updated_at": "2020-01-01T00:00:00Z"}},
    ])

    unfiltered = await store.hybrid_search(tenant, "ns", "onboarding policy", vec(0))
    assert len(unfiltered) == 2

    recent_only = await store.hybrid_search(
        tenant, "ns", "onboarding policy", vec(0),
        metadata_filter=[{"field": "updated_at", "op": "gte", "value": "2026-05-16T00:00:00Z"}],
    )
    assert len(recent_only) == 1
    assert "requires manager approval" in recent_only[0]["content"]


@pytest.mark.asyncio
async def test_traverse_graph_date_range_filter(store):
    tenant = tid()
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "PolicyDoc2026", "embedding": vec(1), "metadata": {"updated_at": "2026-07-01T00:00:00Z"}},
        {"content": "PolicyDoc2020", "embedding": vec(2), "metadata": {"updated_at": "2020-01-01T00:00:00Z"}},
    ])
    await store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids["PolicyDoc2026"], "target_id": node_ids["PolicyDoc2020"], "relation": "supersedes"},
    ])

    filtered = await store.traverse_graph(
        tenant, list(node_ids.values()), namespace="ns", max_hops=1,
        metadata_filter=[{"field": "updated_at", "op": "gte", "value": "2026-05-16T00:00:00Z"}],
    )
    assert {n["content"] for n in filtered["nodes"]} == {"PolicyDoc2026"}


@pytest.mark.asyncio
async def test_metadata_filter_numeric_and_in_ops_against_real_db(store):
    tenant = tid()
    doc_a = await store.upsert_document(tenant, "ns", "doc-a", "ha", {})
    doc_b = await store.upsert_document(tenant, "ns", "doc-b", "hb", {})
    doc_c = await store.upsert_document(tenant, "ns", "doc-c", "hc", {})
    await store.replace_chunks(tenant, doc_a["id"], "ns", [
        {"content": "Report alpha.", "embedding": vec(0), "metadata": {"priority": 5, "status": "active"}},
    ])
    await store.replace_chunks(tenant, doc_b["id"], "ns", [
        {"content": "Report beta.", "embedding": vec(0), "metadata": {"priority": 1, "status": "archived"}},
    ])
    await store.replace_chunks(tenant, doc_c["id"], "ns", [
        {"content": "Report gamma.", "embedding": vec(0), "metadata": {"priority": 3, "status": "pending"}},
    ])

    high_priority = await store.hybrid_search(
        tenant, "ns", "report", vec(0), metadata_filter=[{"field": "priority", "op": "gt", "value": 2}],
    )
    assert {r["content"] for r in high_priority} == {"Report alpha.", "Report gamma."}

    active_or_pending = await store.hybrid_search(
        tenant, "ns", "report", vec(0),
        metadata_filter=[{"field": "status", "op": "in", "value": ["active", "pending"]}],
    )
    assert {r["content"] for r in active_or_pending} == {"Report alpha.", "Report gamma."}


# ----------------------------------------------------------------------
# Reasoning-path reconstruction (find_path / explain_connection)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_path_reconstructs_exact_chain_and_ignores_decoy_branch(store):
    """The literal roadmap ask: given Node A and Node C, show the actual
    path — not just that both are somewhere in a traversal."""
    tenant = tid()
    names = ["JohnySrouji", "HardwareTeam", "M4Chip", "ARM", "M3Chip"]
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": n, "embedding": vec(i)} for i, n in enumerate(names)
    ])
    await store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids["JohnySrouji"], "target_id": node_ids["HardwareTeam"], "relation": "leads"},
        {"source_id": node_ids["HardwareTeam"], "target_id": node_ids["M4Chip"], "relation": "designed"},
        {"source_id": node_ids["M4Chip"], "target_id": node_ids["ARM"], "relation": "uses"},
        {"source_id": node_ids["HardwareTeam"], "target_id": node_ids["M3Chip"], "relation": "designed"},  # decoy
    ])

    result = await store.find_path(tenant, "ns", node_ids["JohnySrouji"], node_ids["ARM"], max_hops=3)
    assert result is not None
    assert result["hop_distance"] == 3
    assert [step["node"] for step in result["path"]] == ["JohnySrouji", "HardwareTeam", "M4Chip", "ARM"]
    assert [step["relation"] for step in result["path"]] == [None, "leads", "designed", "uses"]


@pytest.mark.asyncio
async def test_find_path_returns_none_when_unreachable(store):
    tenant = tid()
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "A", "embedding": vec(0)}, {"content": "B", "embedding": vec(1)},
    ])
    await store.upsert_edges(tenant, "ns", [{"source_id": node_ids["A"], "target_id": node_ids["B"], "relation": "r"}])

    # Directed reverse: B does not reach A.
    result = await store.find_path(tenant, "ns", node_ids["B"], node_ids["A"], max_hops=3, directed=True)
    assert result is None

    # Same node.
    same = await store.find_path(tenant, "ns", node_ids["A"], node_ids["A"], max_hops=3)
    assert same is None

    # Insufficient hop budget.
    tight = await store.find_path(tenant, "ns", node_ids["A"], node_ids["B"], max_hops=0)
    assert tight is None


@pytest.mark.asyncio
async def test_find_path_is_tenant_isolated(store):
    tenant_a, tenant_b = tid(), tid()
    ids_a = await store.resolve_and_upsert_nodes(tenant_a, "ns", [
        {"content": "A", "embedding": vec(0)}, {"content": "B", "embedding": vec(1)},
    ])
    await store.upsert_edges(tenant_a, "ns", [{"source_id": ids_a["A"], "target_id": ids_a["B"], "relation": "r"}])

    # Tenant A's node ids mean nothing under tenant B.
    result = await store.find_path(tenant_b, "ns", ids_a["A"], ids_a["B"], max_hops=3)
    assert result is None


@pytest.mark.asyncio
async def test_explain_connection_end_to_end_through_tenant_engine(store):
    rag = _mock_rag()

    async def fake_embed(text):
        vecs = {
            "JohnySrouji": vec(0), "HardwareTeam": vec(1), "M4Chip": vec(2), "ARM": vec(3),
        }
        if isinstance(text, list):
            return [vecs.get(t, vec(7)) for t in text]
        return vecs.get(text, vec(7))

    from unittest.mock import AsyncMock
    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)

    tenant = tid()
    engine = rag.for_tenant(tenant)
    node_ids = await rag._secure_store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": n, "embedding": v} for n, v in
        {"JohnySrouji": vec(0), "HardwareTeam": vec(1), "M4Chip": vec(2), "ARM": vec(3)}.items()
    ])
    await rag._secure_store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids["JohnySrouji"], "target_id": node_ids["HardwareTeam"], "relation": "leads"},
        {"source_id": node_ids["HardwareTeam"], "target_id": node_ids["M4Chip"], "relation": "designed"},
        {"source_id": node_ids["M4Chip"], "target_id": node_ids["ARM"], "relation": "uses"},
    ])

    explained = await engine.explain_connection("JohnySrouji", "ARM", namespace="ns", max_hops=3)
    assert explained is not None
    assert [s.node for s in explained.steps] == ["JohnySrouji", "HardwareTeam", "M4Chip", "ARM"]
    assert str(explained) == "JohnySrouji --leads--> HardwareTeam --designed--> M4Chip --uses--> ARM"

    too_short = await engine.explain_connection("JohnySrouji", "ARM", namespace="ns", max_hops=1)
    assert too_short is None

    await rag.close()


@pytest.mark.asyncio
async def test_mcp_explain_connection_tool(mcp_rag):
    from mcp import ClientSession
    from mcp.client._memory import InMemoryTransport

    from postgres_graph_rag.mcp_server import build_server

    async def fake_embed(text):
        vecs = {"A": vec(0), "B": vec(1), "C": vec(2)}
        if isinstance(text, list):
            return [vecs.get(t, vec(7)) for t in text]
        return vecs.get(text, vec(7))

    from unittest.mock import AsyncMock
    mcp_rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)

    tenant = tid()
    mcp_rag.for_tenant(tenant)  # side effect: initializes mcp_rag._secure_store
    node_ids = await mcp_rag._secure_store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": n, "embedding": v} for n, v in {"A": vec(0), "B": vec(1), "C": vec(2)}.items()
    ])
    await mcp_rag._secure_store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids["A"], "target_id": node_ids["B"], "relation": "r1"},
        {"source_id": node_ids["B"], "target_id": node_ids["C"], "relation": "r2"},
    ])

    server = build_server(mcp_rag, stdio_tenant_id=tenant, enable_mutations=False)
    transport = InMemoryTransport(server)
    async with transport._connect() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            assert "explain_connection" in tools

            result = await _call(session, "explain_connection", {"source_entity": "A", "target_entity": "C", "namespace": "ns"})
            assert result["found"] is True
            assert result["summary"] == "A --r1--> B --r2--> C"


# ----------------------------------------------------------------------
# Phase A: relationship-level evidence
# ----------------------------------------------------------------------


async def _edge_evidence_fixture(store, tenant, relations):
    doc = await store.upsert_document(tenant, "ns", "evidence-doc", "evidence-hash", {})
    chunk_ids = await store.replace_chunks(tenant, doc["id"], "ns", [
        {"content": "Relationship evidence chunk.", "embedding": vec(0)},
    ])
    node_ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": name, "embedding": vec(i + 1)}
        for i, name in enumerate({n for rel in relations for n in (rel[0], rel[2])})
    ])
    edge_ids = await store.upsert_edges(tenant, "ns", [
        {"source_id": node_ids[s], "target_id": node_ids[o], "relation": p}
        for s, p, o in relations
    ])
    await store.record_edge_mentions(tenant, chunk_ids[0], list(edge_ids.values()))
    return doc, chunk_ids[0], node_ids, edge_ids


@pytest.mark.asyncio
async def test_edge_evidence_returns_the_asserting_document_and_chunk(store):
    tenant = tid()
    _, chunk_id, _, edge_ids = await _edge_evidence_fixture(
        store, tenant, [("Alice", "leads", "Payments")]
    )

    evidence = await store.get_edge_evidence(tenant, "ns", next(iter(edge_ids.values())))
    assert len(evidence) == 1
    assert evidence[0]["source_id"] == "evidence-doc"
    assert str(evidence[0]["chunk_id"]) == chunk_id

    batched = await store.get_edges_evidence(
        tenant, "ns", list(edge_ids.values())
    )
    assert [(row["source_id"], row["chunk_id"]) for row in batched] == [
        ("evidence-doc", chunk_id)
    ]


@pytest.mark.asyncio
async def test_conflicting_relationships_keep_separate_edges_and_evidence(store):
    tenant = tid()
    doc_a = await store.upsert_document(tenant, "ns", "doc-a", "hash-a", {})
    doc_b = await store.upsert_document(tenant, "ns", "doc-b", "hash-b", {})
    chunks_a = await store.replace_chunks(tenant, doc_a["id"], "ns", [{"content": "Alice leads Payments.", "embedding": vec(0)}])
    chunks_b = await store.replace_chunks(tenant, doc_b["id"], "ns", [{"content": "Alice left Payments.", "embedding": vec(1)}])
    ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "Alice", "embedding": vec(2)},
        {"content": "Payments", "embedding": vec(3)},
    ])
    edges = await store.upsert_edges(tenant, "ns", [
        {"source_id": ids["Alice"], "target_id": ids["Payments"], "relation": "leads"},
        {"source_id": ids["Alice"], "target_id": ids["Payments"], "relation": "left"},
    ])
    await store.record_edge_mentions(tenant, chunks_a[0], [edges[(ids["Alice"], ids["Payments"], "leads")]])
    await store.record_edge_mentions(tenant, chunks_b[0], [edges[(ids["Alice"], ids["Payments"], "left")]])

    leads = await store.get_edge_evidence(tenant, "ns", edges[(ids["Alice"], ids["Payments"], "leads")])
    left = await store.get_edge_evidence(tenant, "ns", edges[(ids["Alice"], ids["Payments"], "left")])
    assert [r["source_id"] for r in leads] == ["doc-a"]
    assert [r["source_id"] for r in left] == ["doc-b"]


@pytest.mark.asyncio
async def test_deleting_one_document_removes_only_its_edge_evidence(store):
    tenant = tid()
    doc_a = await store.upsert_document(tenant, "ns", "doc-a", "hash-a", {})
    doc_b = await store.upsert_document(tenant, "ns", "doc-b", "hash-b", {})
    chunk_a = await store.replace_chunks(tenant, doc_a["id"], "ns", [{"content": "A supports B.", "embedding": vec(0)}])
    chunk_b = await store.replace_chunks(tenant, doc_b["id"], "ns", [{"content": "A supports B again.", "embedding": vec(1)}])
    ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": "A", "embedding": vec(2)}, {"content": "B", "embedding": vec(3)}
    ])
    edge_ids = await store.upsert_edges(tenant, "ns", [{"source_id": ids["A"], "target_id": ids["B"], "relation": "supports"}])
    edge_id = edge_ids[(ids["A"], ids["B"], "supports")]
    await store.record_edge_mentions(tenant, chunk_a[0], [edge_id])
    await store.record_edge_mentions(tenant, chunk_b[0], [edge_id])

    await store.delete_document(tenant, "ns", "doc-a")
    evidence = await store.get_edge_evidence(tenant, "ns", edge_id)
    assert [r["source_id"] for r in evidence] == ["doc-b"]


# ----------------------------------------------------------------------
# Phase B: bounded top-N paths
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_paths_returns_ranked_paths_with_edge_evidence(store):
    tenant = tid()
    doc = await store.upsert_document(tenant, "ns", "path-doc", "path-hash", {})
    chunk = await store.replace_chunks(tenant, doc["id"], "ns", [{"content": "Path evidence.", "embedding": vec(0)}])
    names = ["A", "B", "C", "T"]
    ids = await store.resolve_and_upsert_nodes(tenant, "ns", [{"content": n, "embedding": vec(i + 1)} for i, n in enumerate(names)])
    edge_rows = [
        {"source_id": ids["A"], "target_id": ids["B"], "relation": "r1", "weight": 3},
        {"source_id": ids["B"], "target_id": ids["T"], "relation": "r2", "weight": 3},
        {"source_id": ids["A"], "target_id": ids["C"], "relation": "r3", "weight": 1},
        {"source_id": ids["C"], "target_id": ids["T"], "relation": "r4", "weight": 1},
    ]
    edge_ids = await store.upsert_edges(tenant, "ns", edge_rows)
    await store.record_edge_mentions(tenant, chunk[0], list(edge_ids.values()))

    paths = await store.find_paths(tenant, "ns", [ids["A"]], [ids["T"]], max_hops=2, top_k=2)
    assert len(paths) == 2
    assert paths[0]["hop_distance"] == 2
    assert [step["node"] for step in paths[0]["path"]] == ["A", "B", "T"]
    assert all("evidence" in step and step["evidence"] for step in paths[0]["path"][1:])


@pytest.mark.asyncio
async def test_find_paths_supports_multiple_sources_targets_and_top_k(store):
    tenant = tid()
    ids = await store.resolve_and_upsert_nodes(tenant, "ns", [
        {"content": n, "embedding": vec(i)} for i, n in enumerate(["A", "B", "C", "T1", "T2"])
    ])
    await store.upsert_edges(tenant, "ns", [
        {"source_id": ids["A"], "target_id": ids["T1"], "relation": "direct", "weight": 3},
        {"source_id": ids["B"], "target_id": ids["T2"], "relation": "direct", "weight": 2},
        {"source_id": ids["A"], "target_id": ids["C"], "relation": "via", "weight": 1},
        {"source_id": ids["C"], "target_id": ids["T2"], "relation": "via", "weight": 1},
    ])

    paths = await store.find_paths(
        tenant, "ns", [ids["A"], ids["B"]], [ids["T1"], ids["T2"]], max_hops=2, top_k=2
    )
    assert len(paths) == 2
    assert {p["target_id"] for p in paths} == {ids["T1"], ids["T2"]}


@pytest.mark.asyncio
async def test_answer_verified_mode_end_to_end_against_real_retrieval(store):
    """Release 2 PR 5 integration check: grounding_mode="verified" against
    a document actually ingested and retrieved through the real secure
    store (test_tenant_engine.py's 25 tests cover the engine logic itself
    against a mocked store; this confirms the same code path also works
    with genuine TenantRetrievedChunk objects coming back from a live
    hybrid_search/traverse_graph round trip, not just hand-built fixtures).
    """
    from unittest.mock import AsyncMock

    from postgres_graph_rag import PostgresGraphRAG
    from postgres_graph_rag.extractor import Triplet
    from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG

    config = dict(OPENAI_DEFAULT_CONFIG)
    config["dimension"] = DIM
    rag = PostgresGraphRAG(postgres_url=POSTGRES_URL, openai_api_key="test", config=config, runtime_url=_runtime_url())

    async def fake_embed(text):
        return [[0.1] * DIM for _ in text] if isinstance(text, list) else [0.1] * DIM

    rag.extractor.get_embedding = AsyncMock(side_effect=fake_embed)
    rag.extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="Acme", predicate="depends_on", object="Widgets Inc")]
    )
    rag.extractor.generate_text = AsyncMock(
        return_value="Acme depends on Widgets Inc for supply. [doc-1#0]"
    )

    tenant = tid()
    engine = rag.for_tenant(tenant)
    await engine.add_document("Acme depends on Widgets Inc for supply.", namespace="ns", source_id="doc-1")

    result = await engine.answer("What does Acme depend on?", namespace="ns", grounding_mode="verified")

    assert result.grounding_mode == "verified"
    assert result.grounding_status == "verified"
    assert result.grounded is True
    assert len(result.claims) == 1
    assert result.verifications[0].verdict == "supported"
    assert "Acme depends on Widgets Inc" in result.answer
    assert result.citations[0].source_id == "doc-1"

    await rag.close()
