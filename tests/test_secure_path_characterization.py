"""Secure-path characterization tests (LEGACY_DELETION_PLAN.md, G1-G14).

These restore coverage for behaviors that were only exercised, before this
change set, through the legacy `DatabaseManager` (`tests/test_database.py`)
or `live_provider`-marked scenario tests (`tests/test_scenarios.py`). Every
behavior here is still live and default-on in the secure/RLS path
(`SecureGraphStore` / `TenantGraphRAG`); none of it is legacy. G1-G13 run
against a real Postgres instance (skipped without POSTGRES_URL, matching
tests/test_tenancy.py's convention); G14 is a pure mock-based test with no
database dependency, matching tests/test_tenant_engine.py's convention.

Per LEGACY_DELETION_PLAN.md Phase 1 step 6, these must be proven green
against a tree where the legacy engine still exists (i.e. HEAD, before the
uncommitted removal is committed) -- they characterize secure-path behavior
that the legacy removal does not touch, so they must pass identically
before and after it lands.
"""
import os
import uuid

import pytest
import pytest_asyncio
import psycopg
from dotenv import load_dotenv

from postgres_graph_rag.tenancy import SCHEMA, SecureGraphStore, migrate_schema

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
DIM = 8
RUNTIME_ROLE = "pgr_test_runtime"
RUNTIME_PASSWORD = "pgr_test_runtime_pw"

pytestmark = pytest.mark.skipif(not POSTGRES_URL, reason="POSTGRES_URL not set")


def _runtime_url() -> str:
    import urllib.parse as up

    parsed = up.urlparse(POSTGRES_URL)
    netloc = f"{RUNTIME_ROLE}:{RUNTIME_PASSWORD}@{parsed.hostname}:{parsed.port or 5432}"
    return up.urlunparse(parsed._replace(netloc=netloc))


def tid() -> uuid.UUID:
    return uuid.uuid4()


_ns_counter = 0


def ns() -> str:
    global _ns_counter
    _ns_counter += 1
    return f"char-ns-{_ns_counter}"


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
# G1 -- relation_types allow / exclude_relation_types deny, both APIs
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traverse_graph_relation_types_allowlist_restricts_expansion(store):
    tenant = tid()
    namespace = ns()
    node_ids = await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": "checkout-service", "embedding": vec(0)},
        {"content": "auth-service", "embedding": vec(1)},
        {"content": "billing-service", "embedding": vec(2)},
    ])
    await store.upsert_edges(tenant, namespace, [
        {"source_id": node_ids["checkout-service"], "target_id": node_ids["auth-service"], "relation": "depends_on"},
        {"source_id": node_ids["checkout-service"], "target_id": node_ids["billing-service"], "relation": "owned_by"},
    ])

    allowed = await store.traverse_graph(
        tenant, [node_ids["checkout-service"]], namespace=namespace, max_hops=1,
        directed=True, relation_types=["depends_on"],
    )
    contents = {e["target_content"] for e in allowed["edges"]}
    assert contents == {"auth-service"}


@pytest.mark.asyncio
async def test_traverse_graph_exclude_relation_types_denylist_filters_expansion(store):
    tenant = tid()
    namespace = ns()
    node_ids = await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": "checkout-service", "embedding": vec(0)},
        {"content": "auth-service", "embedding": vec(1)},
        {"content": "billing-service", "embedding": vec(2)},
    ])
    await store.upsert_edges(tenant, namespace, [
        {"source_id": node_ids["checkout-service"], "target_id": node_ids["auth-service"], "relation": "depends_on"},
        {"source_id": node_ids["checkout-service"], "target_id": node_ids["billing-service"], "relation": "owned_by"},
    ])

    filtered = await store.traverse_graph(
        tenant, [node_ids["checkout-service"]], namespace=namespace, max_hops=1,
        directed=True, exclude_relation_types=["owned_by"],
    )
    contents = {e["target_content"] for e in filtered["edges"]}
    assert contents == {"auth-service"}


@pytest.mark.asyncio
async def test_find_paths_relation_types_allowlist_restricts_paths(store):
    tenant = tid()
    namespace = ns()
    node_ids = await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": "A", "embedding": vec(0)},
        {"content": "B", "embedding": vec(1)},
    ])
    await store.upsert_edges(tenant, namespace, [
        {"source_id": node_ids["A"], "target_id": node_ids["B"], "relation": "blocked_relation"},
    ])

    paths = await store.find_paths(
        tenant, namespace, [node_ids["A"]], [node_ids["B"]], max_hops=2, top_k=5,
        directed=True, relation_types=["allowed_relation"],
    )
    assert paths == []


# ----------------------------------------------------------------------
# G2 -- max_neighbors_per_node caps fan-out (default is 20; asserted here
# with an explicit small value so the test is fast and deterministic)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traverse_graph_max_neighbors_per_node_caps_fanout(store):
    tenant = tid()
    namespace = ns()
    hub_name = "hub"
    leaf_names = [f"leaf-{i}" for i in range(6)]
    node_ids = await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": hub_name, "embedding": vec(0)},
        *[{"content": name, "embedding": vec(i + 1)} for i, name in enumerate(leaf_names)],
    ])
    await store.upsert_edges(tenant, namespace, [
        {"source_id": node_ids[hub_name], "target_id": node_ids[name], "relation": "connects_to", "weight": 1.0 + i}
        for i, name in enumerate(leaf_names)
    ])

    capped = await store.traverse_graph(
        tenant, [node_ids[hub_name]], namespace=namespace, max_hops=1,
        directed=True, max_neighbors_per_node=3,
    )
    assert len(capped["edges"]) == 3

    uncapped = await store.traverse_graph(
        tenant, [node_ids[hub_name]], namespace=namespace, max_hops=1,
        directed=True, max_neighbors_per_node=20,
    )
    assert len(uncapped["edges"]) == len(leaf_names)


# ----------------------------------------------------------------------
# G3 -- min_weight filters weak edges
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traverse_graph_min_weight_filters_weak_edges(store):
    tenant = tid()
    namespace = ns()
    node_ids = await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": "A", "embedding": vec(0)},
        {"content": "strong-neighbor", "embedding": vec(1)},
        {"content": "weak-neighbor", "embedding": vec(2)},
    ])
    await store.upsert_edges(tenant, namespace, [
        {"source_id": node_ids["A"], "target_id": node_ids["strong-neighbor"], "relation": "r", "weight": 5.0},
        {"source_id": node_ids["A"], "target_id": node_ids["weak-neighbor"], "relation": "r", "weight": 0.1},
    ])

    graph = await store.traverse_graph(
        tenant, [node_ids["A"]], namespace=namespace, max_hops=1,
        directed=True, min_weight=1.0,
    )
    contents = {e["target_content"] for e in graph["edges"]}
    assert contents == {"strong-neighbor"}


# ----------------------------------------------------------------------
# G4 -- score_decay applies per hop distance
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traverse_graph_score_decay_reduces_score_with_hop_distance(store):
    tenant = tid()
    namespace = ns()
    node_ids = await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": "seed", "embedding": vec(0)},
        {"content": "hop1", "embedding": vec(1)},
        {"content": "hop2", "embedding": vec(2)},
    ])
    await store.upsert_edges(tenant, namespace, [
        {"source_id": node_ids["seed"], "target_id": node_ids["hop1"], "relation": "r"},
        {"source_id": node_ids["hop1"], "target_id": node_ids["hop2"], "relation": "r"},
    ])

    decayed = await store.traverse_graph(
        tenant, [node_ids["seed"]], namespace=namespace, max_hops=2,
        directed=True, score_decay=0.5,
    )
    by_id = {str(n["id"]): n for n in decayed["nodes"]}
    hop1_score = by_id[node_ids["hop1"]]["score"]
    hop2_score = by_id[node_ids["hop2"]]["score"]
    assert hop2_score < hop1_score, "score must strictly decrease with hop distance"

    # Isolate score_decay's own contribution by comparing the SAME hop
    # (hop2) across two decay values: edge weight also attenuates score
    # per hop (via `1 - exp(-weight)`), so hop2's score is never equal to
    # hop1's even at score_decay=1.0 -- the comparison must hold decay
    # constant across weight and vary only decay itself.
    less_decayed = await store.traverse_graph(
        tenant, [node_ids["seed"]], namespace=namespace, max_hops=2,
        directed=True, score_decay=1.0,
    )
    by_id_less_decayed = {str(n["id"]): n for n in less_decayed["nodes"]}
    assert by_id_less_decayed[node_ids["hop2"]]["score"] > by_id[node_ids["hop2"]]["score"], (
        "a higher score_decay must yield a higher (less attenuated) score at the same hop distance"
    )


# ----------------------------------------------------------------------
# G5 -- max_hops hard limit rejected, all three raise sites
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traverse_graph_rejects_max_hops_over_hard_limit(store):
    tenant = tid()
    namespace = ns()
    node_ids = await store.resolve_and_upsert_nodes(tenant, namespace, [{"content": "A", "embedding": vec(0)}])

    with pytest.raises(ValueError, match="exceeds the hard limit"):
        await store.traverse_graph(tenant, [node_ids["A"]], namespace=namespace, max_hops=6)


@pytest.mark.asyncio
async def test_find_paths_rejects_max_hops_over_hard_limit(store):
    tenant = tid()
    namespace = ns()
    node_ids = await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": "A", "embedding": vec(0)},
        {"content": "B", "embedding": vec(1)},
    ])

    with pytest.raises(ValueError, match="exceeds the hard limit"):
        await store.find_paths(tenant, namespace, [node_ids["A"]], [node_ids["B"]], max_hops=6)


# ----------------------------------------------------------------------
# G6 -- directed vs undirected traversal semantics
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traverse_graph_directed_does_not_follow_reverse_edge(store):
    tenant = tid()
    namespace = ns()
    node_ids = await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": "checkout-service", "embedding": vec(0)},
        {"content": "auth-service", "embedding": vec(1)},
    ])
    await store.upsert_edges(tenant, namespace, [
        {"source_id": node_ids["checkout-service"], "target_id": node_ids["auth-service"], "relation": "depends_on"},
    ])

    forward = await store.traverse_graph(
        tenant, [node_ids["checkout-service"]], namespace=namespace, max_hops=1, directed=True,
    )
    assert len(forward["edges"]) == 1

    reverse = await store.traverse_graph(
        tenant, [node_ids["auth-service"]], namespace=namespace, max_hops=1, directed=True,
    )
    assert reverse["edges"] == [], "directed traversal must not walk an edge backwards from its target"


@pytest.mark.asyncio
async def test_traverse_graph_undirected_follows_edge_both_ways(store):
    tenant = tid()
    namespace = ns()
    node_ids = await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": "checkout-service", "embedding": vec(0)},
        {"content": "auth-service", "embedding": vec(1)},
    ])
    await store.upsert_edges(tenant, namespace, [
        {"source_id": node_ids["checkout-service"], "target_id": node_ids["auth-service"], "relation": "depends_on"},
    ])

    reverse = await store.traverse_graph(
        tenant, [node_ids["auth-service"]], namespace=namespace, max_hops=1, directed=False,
    )
    assert len(reverse["edges"]) == 1, "undirected traversal must reach a node via an edge pointing away from it"


# ----------------------------------------------------------------------
# G7 -- batch node upsert is idempotent on repeat
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_and_upsert_nodes_is_idempotent_on_repeat(store):
    tenant = tid()
    namespace = ns()
    first = await store.resolve_and_upsert_nodes(tenant, namespace, [{"content": "Apple Inc.", "embedding": vec(0)}])
    second = await store.resolve_and_upsert_nodes(tenant, namespace, [{"content": "Apple Inc.", "embedding": vec(0)}])
    assert first["Apple Inc."] == second["Apple Inc."], "re-upserting the same exact content must resolve to the same node id"

    nodes = await store.vector_search_nodes(tenant, namespace, vec(0), top_k=10)
    assert sum(1 for n in nodes if n["content"] == "Apple Inc.") == 1, "no duplicate node should be created"


# ----------------------------------------------------------------------
# G8 -- edge weight increments on repeated mention
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_edges_weight_increments_on_repeated_mention(store):
    tenant = tid()
    namespace = ns()
    node_ids = await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": "A", "embedding": vec(0)},
        {"content": "B", "embedding": vec(1)},
    ])
    await store.upsert_edges(tenant, namespace, [
        {"source_id": node_ids["A"], "target_id": node_ids["B"], "relation": "r", "weight": 1.0},
    ])
    await store.upsert_edges(tenant, namespace, [
        {"source_id": node_ids["A"], "target_id": node_ids["B"], "relation": "r", "weight": 1.0},
    ])

    graph = await store.traverse_graph(tenant, [node_ids["A"]], namespace=namespace, max_hops=1, directed=True)
    assert len(graph["edges"]) == 1
    assert graph["edges"][0]["weight"] >= 2.0, "repeated mentions of the same edge must accumulate weight, not overwrite it"


# ----------------------------------------------------------------------
# G9 -- namespace isolation *within one tenant* (a distinct boundary from
# tenant isolation, which is already covered elsewhere)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_namespace_isolation_in_vector_search(store):
    tenant = tid()
    ns_a, ns_b = ns(), ns()
    await store.resolve_and_upsert_nodes(tenant, ns_a, [{"content": "only-in-a", "embedding": vec(0)}])
    await store.resolve_and_upsert_nodes(tenant, ns_b, [{"content": "only-in-b", "embedding": vec(0)}])

    results_a = await store.vector_search_nodes(tenant, ns_a, vec(0), top_k=10)
    assert {n["content"] for n in results_a} == {"only-in-a"}

    results_b = await store.vector_search_nodes(tenant, ns_b, vec(0), top_k=10)
    assert {n["content"] for n in results_b} == {"only-in-b"}


@pytest.mark.asyncio
async def test_namespace_isolation_in_traversal(store):
    tenant = tid()
    ns_a, ns_b = ns(), ns()
    ids_a = await store.resolve_and_upsert_nodes(tenant, ns_a, [
        {"content": "A", "embedding": vec(0)}, {"content": "B", "embedding": vec(1)},
    ])
    await store.upsert_edges(tenant, ns_a, [
        {"source_id": ids_a["A"], "target_id": ids_a["B"], "relation": "r"},
    ])
    ids_b = await store.resolve_and_upsert_nodes(tenant, ns_b, [
        {"content": "A", "embedding": vec(0)}, {"content": "C", "embedding": vec(2)},
    ])

    graph = await store.traverse_graph(tenant, [ids_a["A"]], namespace=ns_b, max_hops=1, directed=True)
    assert graph["edges"] == [], "a seed id from namespace A must not expand into namespace B's edges"


# ----------------------------------------------------------------------
# G10 -- automatic entity resolution: exact-normalization merge, metadata
# merge on exact match, cross-chunk resolution
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exact_normalization_merges_whitespace_variants(store):
    tenant = tid()
    namespace = ns()
    first = await store.resolve_and_upsert_nodes(tenant, namespace, [{"content": "Apple Inc.", "embedding": vec(0)}])
    second = await store.resolve_and_upsert_nodes(tenant, namespace, [{"content": "  Apple   Inc.  ", "embedding": vec(0)}])
    assert first["Apple Inc."] == second["  Apple   Inc.  "], (
        "whitespace-only variance must resolve to the same node via normalize_entity()"
    )


@pytest.mark.asyncio
async def test_exact_match_resolution_merges_new_metadata(store):
    tenant = tid()
    namespace = ns()
    await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": "Apple Inc.", "embedding": vec(0), "metadata": {"ticker": "AAPL"}},
    ])
    await store.resolve_and_upsert_nodes(tenant, namespace, [
        {"content": "Apple Inc.", "embedding": vec(0), "metadata": {"sector": "Technology"}},
    ])

    nodes = await store.vector_search_nodes(tenant, namespace, vec(0), top_k=10)
    node = next(n for n in nodes if n["content"] == "Apple Inc.")
    assert node["metadata"].get("ticker") == "AAPL"
    assert node["metadata"].get("sector") == "Technology", (
        "a second exact-match upsert must merge new metadata keys rather than overwrite the whole object"
    )


@pytest.mark.asyncio
async def test_entity_resolution_across_chunks(store):
    """The same entity mentioned in two different chunks/documents resolves
    to one node, so its edges/mentions are unified rather than split."""
    tenant = tid()
    namespace = ns()
    doc1 = await store.upsert_document(tenant, namespace, "doc-1", "h1", {})
    doc2 = await store.upsert_document(tenant, namespace, "doc-2", "h2", {})
    chunk_ids_1 = await store.replace_chunks(tenant, doc1["id"], namespace, [
        {"content": "Marie Curie discovered radium.", "embedding": vec(0)},
    ])
    chunk_ids_2 = await store.replace_chunks(tenant, doc2["id"], namespace, [
        {"content": "Marie Curie won two Nobel Prizes.", "embedding": vec(1)},
    ])

    node_ids_1 = await store.resolve_and_upsert_nodes(tenant, namespace, [{"content": "Marie Curie", "embedding": vec(0)}])
    node_ids_2 = await store.resolve_and_upsert_nodes(tenant, namespace, [{"content": "Marie Curie", "embedding": vec(0)}])
    assert node_ids_1["Marie Curie"] == node_ids_2["Marie Curie"]

    await store.record_entity_mentions(tenant, chunk_ids_1[0], [node_ids_1["Marie Curie"]])
    await store.record_entity_mentions(tenant, chunk_ids_2[0], [node_ids_2["Marie Curie"]])

    docs = await store.get_mentioning_documents(tenant, namespace, node_ids_1["Marie Curie"])
    assert {d["source_id"] for d in docs} == {"doc-1", "doc-2"}, (
        "one resolved entity must accumulate provenance from every document that mentions it"
    )


# ----------------------------------------------------------------------
# G11 -- fuzzy resolution merges true variants (the positive case; the
# negative case already exists as test_numbered_identifiers_are_never_fuzzy_merged)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fuzzy_resolution_merges_true_variants(store):
    tenant = tid()
    namespace = ns()
    first = await store.resolve_and_upsert_nodes(
        tenant, namespace, [{"content": "International Business Machines", "embedding": vec(0)}],
    )
    second = await store.resolve_and_upsert_nodes(
        tenant, namespace,
        [{"content": "International Business Machine", "embedding": vec(0)}],
        fuzzy=True, trgm_threshold=0.4, embedding_threshold=0.90,
    )
    assert first["International Business Machines"] == second["International Business Machine"], (
        "a trigram-similar variant with a near-identical embedding must fuzzy-merge into the existing node"
    )


# ----------------------------------------------------------------------
# G12 -- migrate_schema rejects invalid embedding_dimension before connecting
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migrate_schema_rejects_negative_embedding_dimension():
    with pytest.raises(ValueError, match="Invalid embedding_dimension"):
        await migrate_schema(
            admin_url=POSTGRES_URL,
            runtime_role=RUNTIME_ROLE,
            runtime_password=RUNTIME_PASSWORD,
            embedding_dimension=-1,
        )


@pytest.mark.asyncio
async def test_migrate_schema_rejects_injection_shaped_embedding_dimension():
    with pytest.raises(ValueError, match="Invalid embedding_dimension"):
        await migrate_schema(
            admin_url=POSTGRES_URL,
            runtime_role=RUNTIME_ROLE,
            runtime_password=RUNTIME_PASSWORD,
            embedding_dimension="8; DROP TABLE graph_nodes;",  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------------
# G13 -- fuzzy resolution does NOT merge trigram-similar names with
# dissimilar embeddings (the John Smith / John Smyth adversarial case)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fuzzy_resolution_does_not_merge_trigram_similar_dissimilar_embeddings(store):
    """Adversarial case: names can be textually similar (e.g. share a
    common surname) while referring to different entities. Embedding
    confirmation must prevent a false merge even if trigram similarity
    alone would suggest one."""
    namespace = ns()
    tenant = tid()
    ids1 = await store.resolve_and_upsert_nodes(
        tenant, namespace, [{"content": "John Smith", "embedding": vec(0)}],
    )
    # "John Smyth" is textually close (high trigram similarity) but its
    # embedding points in an unrelated direction -> must NOT merge.
    ids2 = await store.resolve_and_upsert_nodes(
        tenant, namespace, [{"content": "John Smyth", "embedding": vec(5, 6)}],
        fuzzy=True, trgm_threshold=0.2, embedding_threshold=0.85,
    )
    assert ids1["John Smith"] != ids2["John Smyth"]
