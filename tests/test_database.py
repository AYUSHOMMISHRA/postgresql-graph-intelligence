"""DB-level tests that exercise DatabaseManager directly against a real
Postgres+pgvector instance (no LLM key required, since we supply synthetic
embeddings ourselves). Requires POSTGRES_URL; skipped otherwise.

Safety: these tests run inside a dedicated `pgr_test` Postgres *schema*
(via `search_path`), not the default `public` schema. This means they never
touch `public.graph_nodes`/`graph_edges` — the tables an application
pointed at the same POSTGRES_URL would actually be using. The dedicated
schema is dropped and recreated per-test, so tests stay fully isolated from
each other without ever running a destructive DROP against shared/app data.
"""
import os
import uuid
from urllib.parse import quote

import psycopg
import pytest
import pytest_asyncio
from dotenv import load_dotenv
from postgres_graph_rag.database import DatabaseManager, normalize_entity, content_hash

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
DIM = 8
TEST_SCHEMA = "pgr_test"

pytestmark = pytest.mark.skipif(not POSTGRES_URL, reason="POSTGRES_URL not set")


def _scoped_url(base_url: str, schema: str) -> str:
    """Returns `base_url` with a `search_path` connection option that scopes
    every unqualified table name (graph_nodes, graph_edges, ...) to `schema`
    instead of `public`, without requiring any change to DatabaseManager."""
    options = quote(f"-c search_path={schema},public")
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}options={options}"


def vec(*nonzero_positions: int, dim: int = DIM) -> list:
    """Builds a simple one-hot-ish embedding so cosine similarity between
    vectors we construct is predictable and easy to reason about in tests."""
    v = [0.0] * dim
    for p in nonzero_positions:
        v[p] = 1.0
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v] if norm else v


def near(base_positions, dim=DIM, noise=0.05):
    v = vec(*base_positions, dim=dim)
    return [x + (noise if i not in base_positions else 0) for i, x in enumerate(v)]


@pytest_asyncio.fixture
async def db():
    # Recreate the dedicated test schema from scratch. This only ever
    # touches `pgr_test`, never the `public` schema an application would
    # actually be reading/writing through the same POSTGRES_URL.
    admin_conn = await psycopg.AsyncConnection.connect(POSTGRES_URL)
    async with admin_conn.cursor() as cur:
        await cur.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
        await cur.execute(f"CREATE SCHEMA {TEST_SCHEMA}")
        await admin_conn.commit()
    await admin_conn.close()

    manager = DatabaseManager(_scoped_url(POSTGRES_URL, TEST_SCHEMA))
    await manager.setup_database(embedding_dimension=DIM)
    yield manager
    await manager.close()


def ns() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_setup_rejects_invalid_dimension(db):
    with pytest.raises(ValueError):
        await db.setup_database(embedding_dimension=-1)
    with pytest.raises(ValueError):
        await db.setup_database(embedding_dimension="8; DROP TABLE graph_nodes;")


@pytest.mark.asyncio
async def test_bulk_upsert_single_round_trip_and_idempotent(db):
    namespace = ns()
    nodes = [
        {"content": f"Entity{i}", "embedding": vec(i % DIM), "metadata": {"i": i}}
        for i in range(50)
    ]
    ids_first = await db.upsert_nodes_batch(nodes, namespace=namespace)
    assert len(set(ids_first)) == 50

    # Re-upserting the same content should return the *same* ids (upsert,
    # not duplicate insert).
    ids_second = await db.upsert_nodes_batch(nodes, namespace=namespace)
    assert ids_first == ids_second


@pytest.mark.asyncio
async def test_edge_weight_increments_on_repeated_mention(db):
    namespace = ns()
    ids = await db.upsert_nodes_batch(
        [
            {"content": "A", "embedding": vec(0)},
            {"content": "B", "embedding": vec(1)},
        ],
        namespace=namespace,
    )
    a, b = ids
    await db.upsert_edge(a, b, "relates_to", namespace=namespace, weight=1.0)
    await db.upsert_edge(a, b, "relates_to", namespace=namespace, weight=1.0)
    await db.upsert_edge(a, b, "relates_to", namespace=namespace, weight=1.0)

    result = await db.traverse_graph([a], namespace=namespace, max_hops=1)
    edge = result["edges"][0]
    assert edge["weight"] == 3.0


@pytest.mark.asyncio
async def test_directed_traversal_respects_edge_direction(db):
    namespace = ns()
    ids = await db.upsert_nodes_batch(
        [
            {"content": "Service", "embedding": vec(0)},
            {"content": "Database", "embedding": vec(1)},
        ],
        namespace=namespace,
    )
    service, database = ids
    await db.upsert_edge(service, database, "depends_on", namespace=namespace)

    # Directed: traversing *from* Database must NOT reach Service.
    forward = await db.traverse_graph(
        [service], namespace=namespace, max_hops=1, directed=True
    )
    assert {n["content"] for n in forward["nodes"]} == {"Service", "Database"}

    backward = await db.traverse_graph(
        [database], namespace=namespace, max_hops=1, directed=True
    )
    assert {n["content"] for n in backward["nodes"]} == {"Database"}

    # Undirected (default): traversing from either side reaches both.
    undirected = await db.traverse_graph(
        [database], namespace=namespace, max_hops=1, directed=False
    )
    assert {n["content"] for n in undirected["nodes"]} == {"Service", "Database"}


@pytest.mark.asyncio
async def test_relation_type_allow_and_deny_lists(db):
    namespace = ns()
    ids = await db.upsert_nodes_batch(
        [
            {"content": "A", "embedding": vec(0)},
            {"content": "B", "embedding": vec(1)},
            {"content": "C", "embedding": vec(2)},
        ],
        namespace=namespace,
    )
    a, b, c = ids
    await db.upsert_edge(a, b, "uses", namespace=namespace)
    await db.upsert_edge(a, c, "mentions", namespace=namespace)

    allowed = await db.traverse_graph(
        [a], namespace=namespace, max_hops=1, relation_types=["uses"]
    )
    assert {n["content"] for n in allowed["nodes"]} == {"A", "B"}

    denied = await db.traverse_graph(
        [a], namespace=namespace, max_hops=1, exclude_relation_types=["uses"]
    )
    assert {n["content"] for n in denied["nodes"]} == {"A", "C"}


@pytest.mark.asyncio
async def test_max_hops_hard_limit_rejected(db):
    with pytest.raises(ValueError):
        await db.traverse_graph(["00000000-0000-0000-0000-000000000000"], max_hops=6)


@pytest.mark.asyncio
async def test_max_neighbors_per_node_caps_fanout(db):
    namespace = ns()
    ids = await db.upsert_nodes_batch(
        [{"content": f"N{i}", "embedding": vec(i % DIM)} for i in range(11)],
        namespace=namespace,
    )
    hub, leaves = ids[0], ids[1:]
    for i, leaf in enumerate(leaves):
        # Distinct weights so the "highest-weight neighbors" cap is
        # deterministic to assert against.
        await db.upsert_edge(hub, leaf, "rel", namespace=namespace, weight=float(i + 1))

    result = await db.traverse_graph(
        [hub], namespace=namespace, max_hops=1, max_neighbors_per_node=3
    )
    reached = {n["content"] for n in result["nodes"]} - {"N0"}
    # Only the 3 highest-weight edges (weights 8, 9, 10 -> N8, N9, N10) should
    # be followed, not all 10.
    assert reached == {"N8", "N9", "N10"}


@pytest.mark.asyncio
async def test_min_weight_filters_weak_edges(db):
    namespace = ns()
    ids = await db.upsert_nodes_batch(
        [
            {"content": "A", "embedding": vec(0)},
            {"content": "Weak", "embedding": vec(1)},
        ],
        namespace=namespace,
    )
    a, weak = ids
    await db.upsert_edge(a, weak, "maybe_related", namespace=namespace, weight=0.1)

    result = await db.traverse_graph(
        [a], namespace=namespace, max_hops=1, min_weight=0.5
    )
    assert {n["content"] for n in result["nodes"]} == {"A"}


@pytest.mark.asyncio
async def test_hop_distance_and_score_decay(db):
    namespace = ns()
    ids = await db.upsert_nodes_batch(
        [
            {"content": "Seed", "embedding": vec(0)},
            {"content": "OneHop", "embedding": vec(1)},
            {"content": "TwoHop", "embedding": vec(2)},
        ],
        namespace=namespace,
    )
    seed, one_hop, two_hop = ids
    await db.upsert_edge(seed, one_hop, "rel", namespace=namespace, weight=1.0)
    await db.upsert_edge(one_hop, two_hop, "rel", namespace=namespace, weight=1.0)

    result = await db.traverse_graph(
        [seed],
        namespace=namespace,
        max_hops=2,
        seed_scores={seed: 1.0},
        score_decay=0.5,
    )
    by_content = {n["content"]: n for n in result["nodes"]}
    assert by_content["Seed"]["hop_distance"] == 0
    assert by_content["OneHop"]["hop_distance"] == 1
    assert by_content["TwoHop"]["hop_distance"] == 2
    # Score should strictly decrease with distance from the seed.
    assert by_content["Seed"]["score"] > by_content["OneHop"]["score"] > by_content["TwoHop"]["score"]


@pytest.mark.asyncio
async def test_namespace_isolation_in_traversal_and_vector_search(db):
    ns_a, ns_b = ns(), ns()
    ids_a = await db.upsert_nodes_batch(
        [{"content": "SecretA", "embedding": vec(0)}], namespace=ns_a
    )
    await db.upsert_nodes_batch(
        [{"content": "SecretB", "embedding": vec(0)}], namespace=ns_b
    )

    # Same embedding, different namespace: vector_search must not cross over.
    results_a = await db.vector_search(vec(0), namespace=ns_a, top_k=10)
    assert {r["content"] for r in results_a} == {"SecretA"}

    # A seed id from namespace A must not resolve/expand under namespace B.
    cross = await db.traverse_graph(ids_a, namespace=ns_b, max_hops=2)
    assert cross["nodes"] == []


@pytest.mark.asyncio
async def test_entity_resolution_exact_normalization_merges(db):
    namespace = ns()
    ids1 = await db.resolve_and_upsert_nodes_batch(
        [{"content": "  Apple   Inc.  ", "embedding": vec(0)}], namespace=namespace
    )
    ids2 = await db.resolve_and_upsert_nodes_batch(
        [{"content": "Apple Inc.", "embedding": vec(0)}], namespace=namespace
    )
    assert list(ids1.values()) == list(ids2.values())


@pytest.mark.asyncio
async def test_exact_match_resolution_merges_new_metadata(db):
    """Regression test: resolving to an existing node via exact match must
    still merge in newly-supplied metadata, not silently drop it (this is
    exactly what test_scenarios.py::test_metadata_integrity_via_jsonb_merge
    caught against a live database: re-ingesting known content with new
    metadata resolved to the existing node and then did nothing with it)."""
    namespace = ns()
    await db.resolve_and_upsert_nodes_batch(
        [{"content": "London", "embedding": vec(0), "metadata": {"source": "book_1"}}],
        namespace=namespace,
    )
    await db.resolve_and_upsert_nodes_batch(
        [{"content": "London", "embedding": vec(0), "metadata": {"quality": "high"}}],
        namespace=namespace,
    )

    result = await db.vector_search(vec(0), namespace=namespace, top_k=1)
    metadata = result[0]["metadata"]
    assert metadata.get("source") == "book_1"
    assert metadata.get("quality") == "high"


@pytest.mark.asyncio
async def test_entity_resolution_fuzzy_merges_true_variants(db):
    namespace = ns()
    ids1 = await db.resolve_and_upsert_nodes_batch(
        [{"content": "Elon Musk", "embedding": vec(0, 1)}], namespace=namespace
    )
    # "Elon" is a substring/near-duplicate mention with a near-identical
    # embedding (same semantic entity) -> should resolve to the same node.
    ids2 = await db.resolve_and_upsert_nodes_batch(
        [{"content": "Elon", "embedding": near([0, 1])}],
        namespace=namespace,
        fuzzy=True,
        trgm_threshold=0.2,
        embedding_threshold=0.85,
    )
    assert ids1["Elon Musk"] == ids2["Elon"]


@pytest.mark.asyncio
async def test_entity_resolution_does_not_merge_unrelated_entities(db):
    """Adversarial case: names can be textually similar (e.g. share a
    common surname/company suffix) while referring to different entities.
    Embedding confirmation must prevent a false merge even if the trigram
    similarity alone would suggest one."""
    namespace = ns()
    await db.resolve_and_upsert_nodes_batch(
        [{"content": "John Smith", "embedding": vec(0)}], namespace=namespace
    )
    # "John Smyth" is textually close (high trigram similarity) but
    # embeddings point in an unrelated direction -> must NOT merge.
    ids2 = await db.resolve_and_upsert_nodes_batch(
        [{"content": "John Smyth", "embedding": vec(5, 6)}],
        namespace=namespace,
        fuzzy=True,
        trgm_threshold=0.2,
        embedding_threshold=0.85,
    )
    ids1 = await db.resolve_and_upsert_nodes_batch(
        [{"content": "John Smith", "embedding": vec(0)}], namespace=namespace
    )
    assert ids1["John Smith"] != ids2["John Smyth"]


@pytest.mark.asyncio
async def test_chunk_hash_idempotency_tracking(db):
    namespace = ns()
    chunks = ["Alpha ingests fine.", "Beta ingests fine too."]

    new_chunks = await db.filter_new_chunks(chunks, namespace=namespace)
    assert new_chunks == chunks

    await db.mark_chunks_ingested(chunks, namespace=namespace, triplet_counts=[1, 2])

    new_chunks_again = await db.filter_new_chunks(chunks, namespace=namespace)
    assert new_chunks_again == []

    # A genuinely new chunk alongside previously-seen ones is still detected.
    mixed = chunks + ["Gamma is brand new."]
    assert await db.filter_new_chunks(mixed, namespace=namespace) == [
        "Gamma is brand new."
    ]


def test_normalize_entity_collapses_whitespace():
    assert normalize_entity("  Apple   Inc.  ") == "Apple Inc."


def test_content_hash_stable_and_sensitive_to_change():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")
