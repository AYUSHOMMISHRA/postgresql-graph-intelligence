import uuid
import warnings

import pytest
from unittest.mock import AsyncMock, MagicMock
from postgres_graph_rag import PostgresGraphRAG
from postgres_graph_rag.extractor import Triplet
from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG


@pytest.fixture
def mock_db_class(mocker):
    return mocker.patch("postgres_graph_rag.core.DatabaseManager")


@pytest.fixture
def mock_extractor_class(mocker):
    return mocker.patch("postgres_graph_rag.core.LLMExtractor")


def _wire_async_connection(mock_db):
    """Configures `mock_db.pool.connection()` to behave like psycopg's
    async context manager (`async with pool.connection() as conn: ...`)."""
    mock_db._init_pool = AsyncMock()
    mock_conn = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    mock_db.pool.connection.return_value = cm
    mock_conn.commit = AsyncMock()
    return mock_conn


@pytest.fixture
def rag(mock_db_class, mock_extractor_class):
    # Synchronous initialization
    return PostgresGraphRAG(
        postgres_url="postgresql://user:pass@localhost:5432/db",
        openai_api_key="test_key",
        config=OPENAI_DEFAULT_CONFIG,
    )


@pytest.mark.asyncio
async def test_ingest(rag, mock_db_class, mock_extractor_class):
    mock_db = mock_db_class.return_value
    mock_extractor = mock_extractor_class.return_value
    mock_extractor.config = OPENAI_DEFAULT_CONFIG
    _wire_async_connection(mock_db)

    mock_db.filter_new_chunks = AsyncMock(side_effect=lambda chunks, namespace: chunks)
    mock_db.mark_chunks_ingested = AsyncMock()

    mock_extractor.extract_triplets = AsyncMock()
    mock_extractor.extract_triplets.return_value = [
        Triplet(subject="Apple", predicate="released", object="M4")
    ]

    mock_extractor.get_embedding = AsyncMock()
    mock_extractor.get_embedding.return_value = [[0.1] * 1536, [0.1] * 1536]

    mock_db.resolve_and_upsert_nodes_batch = AsyncMock(
        return_value={"Apple": "uuid1", "M4": "uuid2"}
    )
    mock_db.upsert_edges_batch = AsyncMock()

    await rag.add_texts("Apple released the M4.", namespace="test-ns")

    mock_extractor.extract_triplets.assert_called_once()
    mock_db.resolve_and_upsert_nodes_batch.assert_called_once()
    call_kwargs = mock_db.resolve_and_upsert_nodes_batch.call_args
    assert call_kwargs.kwargs["namespace"] == "test-ns"

    mock_db.upsert_edges_batch.assert_called_once()
    edges_arg = mock_db.upsert_edges_batch.call_args.args[0]
    assert edges_arg == [
        {
            "source_id": "uuid1",
            "target_id": "uuid2",
            "relation": "released",
            "metadata": None,
        }
    ]


@pytest.mark.asyncio
async def test_ingest_skips_already_ingested_chunks(
    rag, mock_db_class, mock_extractor_class
):
    """Re-ingesting identical content should not re-trigger LLM extraction."""
    mock_db = mock_db_class.return_value
    mock_extractor = mock_extractor_class.return_value
    mock_extractor.config = OPENAI_DEFAULT_CONFIG
    _wire_async_connection(mock_db)

    mock_db.filter_new_chunks = AsyncMock(return_value=[])
    mock_db.mark_chunks_ingested = AsyncMock()
    mock_extractor.extract_triplets = AsyncMock()

    await rag.add_texts("Already seen text.", namespace="test-ns")

    mock_extractor.extract_triplets.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_does_not_skip_when_metadata_provided(
    rag, mock_db_class, mock_extractor_class
):
    """Regression test: a call attaching metadata to already-seen content
    must not be skipped by the idempotency optimization, since there is no
    way to attach that metadata to previously-resolved entities after the
    fact (found via a real end-to-end test failure:
    test_scenarios.py::test_metadata_integrity_via_jsonb_merge)."""
    mock_db = mock_db_class.return_value
    mock_extractor = mock_extractor_class.return_value
    mock_extractor.config = OPENAI_DEFAULT_CONFIG
    _wire_async_connection(mock_db)

    mock_db.filter_new_chunks = AsyncMock(return_value=[])
    mock_db.mark_chunks_ingested = AsyncMock()
    mock_extractor.extract_triplets = AsyncMock(
        return_value=[Triplet(subject="London", predicate="is_a", object="city")]
    )
    mock_extractor.get_embedding = AsyncMock(return_value=[[0.1] * 1536, [0.1] * 1536])
    mock_db.resolve_and_upsert_nodes_batch = AsyncMock(
        return_value={"London": "uuid1", "city": "uuid2"}
    )
    mock_db.upsert_edges_batch = AsyncMock()

    await rag.add_texts(
        "London is a city.", namespace="test-ns", metadata={"quality": "high"}
    )

    mock_db.filter_new_chunks.assert_not_called()
    mock_extractor.extract_triplets.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_skips_failed_chunk_without_aborting_batch(
    rag, mock_db_class, mock_extractor_class
):
    """A chunk whose extraction fails after retries should be skipped, not
    crash the whole ingestion call."""
    mock_db = mock_db_class.return_value
    mock_extractor = mock_extractor_class.return_value
    mock_extractor.config = OPENAI_DEFAULT_CONFIG
    _wire_async_connection(mock_db)

    rag.ingestion_config["max_extraction_retries"] = 1
    rag.chunker = lambda text: ["good chunk", "bad chunk"]

    mock_db.filter_new_chunks = AsyncMock(side_effect=lambda chunks, namespace: chunks)
    mock_db.mark_chunks_ingested = AsyncMock()

    async def extract_side_effect(chunk):
        if chunk == "bad chunk":
            raise RuntimeError("provider timeout")
        return [Triplet(subject="A", predicate="rel", object="B")]

    mock_extractor.extract_triplets = AsyncMock(side_effect=extract_side_effect)
    mock_extractor.get_embedding = AsyncMock(return_value=[[0.1] * 1536, [0.1] * 1536])
    mock_db.resolve_and_upsert_nodes_batch = AsyncMock(
        return_value={"A": "uuid1", "B": "uuid2"}
    )
    mock_db.upsert_edges_batch = AsyncMock()

    await rag.add_texts("irrelevant", namespace="test-ns")

    mock_db.upsert_edges_batch.assert_called_once()
    marked_chunks = mock_db.mark_chunks_ingested.call_args.args[0]
    assert marked_chunks == ["good chunk"]


@pytest.mark.asyncio
async def test_query(rag, mock_db_class, mock_extractor_class):
    mock_db = mock_db_class.return_value
    mock_extractor = mock_extractor_class.return_value
    _wire_async_connection(mock_db)

    mock_extractor.get_embedding = AsyncMock()
    mock_extractor.get_embedding.return_value = [0.1] * 1536

    mock_db.vector_search = AsyncMock()
    mock_db.vector_search.return_value = [
        {"id": "uuid1", "content": "Apple", "metadata": {}, "distance": 0.1}
    ]

    mock_db.traverse_graph = AsyncMock()
    mock_db.traverse_graph.return_value = {
        "nodes": [
            {"id": "uuid1", "content": "Apple", "metadata": {}, "hop_distance": 0, "score": 0.9},
            {"id": "uuid2", "content": "M4", "metadata": {}, "hop_distance": 1, "score": 0.6},
        ],
        "edges": [
            {
                "source_node_id": "uuid1",
                "target_node_id": "uuid2",
                "relation": "released",
                "metadata": {},
                "source_content": "Apple",
                "target_content": "M4",
                "weight": 1.0,
            }
        ],
    }

    context = await rag.query("What did Apple release?", namespace="test-ns")

    mock_db.vector_search.assert_called_once_with(
        [0.1] * 1536, namespace="test-ns", top_k=5, connection=mock_db.pool.connection.return_value.__aenter__.return_value
    )
    assert "Apple" in context
    assert "M4" in context


@pytest.mark.asyncio
async def test_query_structured_sorts_by_score_and_passes_filters(
    rag, mock_db_class, mock_extractor_class
):
    mock_db = mock_db_class.return_value
    mock_extractor = mock_extractor_class.return_value
    _wire_async_connection(mock_db)

    mock_extractor.get_embedding = AsyncMock(return_value=[0.1] * 1536)
    mock_db.vector_search = AsyncMock(
        return_value=[{"id": "uuid1", "content": "A", "metadata": {}, "distance": 0.2}]
    )
    mock_db.traverse_graph = AsyncMock(
        return_value={
            "nodes": [
                {"id": "uuid1", "content": "A", "metadata": {}, "hop_distance": 0, "score": 0.8},
                {"id": "uuid2", "content": "B", "metadata": {}, "hop_distance": 1, "score": 0.95},
            ],
            "edges": [],
        }
    )

    result = await rag.query_structured(
        "q", namespace="ns", directed=True, relation_types=["depends_on"]
    )

    assert [n.content for n in result.nodes] == ["B", "A"]
    mock_db.traverse_graph.assert_called_once()
    kwargs = mock_db.traverse_graph.call_args.kwargs
    assert kwargs["directed"] is True
    assert kwargs["relation_types"] == ["depends_on"]


# ----------------------------------------------------------------------
# Legacy path deprecation warnings
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_emits_deprecation_warning(rag, mock_db_class, mock_extractor_class):
    mock_db = mock_db_class.return_value
    mock_db.setup_database = AsyncMock()

    with pytest.warns(DeprecationWarning, match="setup\\(\\) uses the legacy single-tenant path"):
        await rag.setup()


@pytest.mark.asyncio
async def test_add_texts_emits_deprecation_warning(rag, mock_db_class, mock_extractor_class):
    mock_db = mock_db_class.return_value
    mock_extractor = mock_extractor_class.return_value
    mock_extractor.config = OPENAI_DEFAULT_CONFIG
    _wire_async_connection(mock_db)
    mock_db.filter_new_chunks = AsyncMock(side_effect=lambda chunks, namespace: chunks)
    mock_db.mark_chunks_ingested = AsyncMock()
    mock_extractor.extract_triplets = AsyncMock(return_value=[])
    mock_extractor.get_embedding = AsyncMock(return_value=[[0.1] * 1536])

    with pytest.warns(DeprecationWarning, match="add_texts\\(\\) uses the legacy single-tenant path"):
        await rag.add_texts("Apple released the M4.", namespace="test-ns")


@pytest.mark.asyncio
async def test_query_structured_emits_deprecation_warning(rag, mock_db_class, mock_extractor_class):
    mock_db = mock_db_class.return_value
    mock_extractor = mock_extractor_class.return_value
    _wire_async_connection(mock_db)
    mock_extractor.get_embedding = AsyncMock(return_value=[0.1] * 1536)
    mock_db.vector_search = AsyncMock(return_value=[])
    mock_db.traverse_graph = AsyncMock(return_value={"nodes": [], "edges": []})

    with pytest.warns(DeprecationWarning, match="query_structured\\(\\) uses the legacy single-tenant path"):
        await rag.query_structured("q", namespace="ns")


@pytest.mark.asyncio
async def test_query_emits_deprecation_warning(rag, mock_db_class, mock_extractor_class):
    mock_db = mock_db_class.return_value
    mock_extractor = mock_extractor_class.return_value
    _wire_async_connection(mock_db)
    mock_extractor.get_embedding = AsyncMock(return_value=[0.1] * 1536)
    mock_db.vector_search = AsyncMock(return_value=[])
    mock_db.traverse_graph = AsyncMock(return_value={"nodes": [], "edges": []})

    # query() delegates to query_structured(), which also warns -- both are
    # real, distinct call sites (different lines), so both fire; confirm at
    # least query()'s own warning is present rather than asserting an exact
    # count that would be brittle to that implementation detail.
    with pytest.warns(DeprecationWarning, match="query\\(\\) uses the legacy single-tenant path"):
        await rag.query("q", namespace="ns")


@pytest.mark.asyncio
async def test_for_tenant_does_not_emit_legacy_deprecation_warning(rag, mock_db_class, mock_extractor_class):
    """for_tenant() is the recommended, secure path -- it must never trigger
    the legacy warning, even though it's a method on the same class."""
    mock_extractor = mock_extractor_class.return_value
    mock_extractor.config = OPENAI_DEFAULT_CONFIG
    rag._runtime_url = "postgresql://user:pass@localhost:5432/db"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rag.for_tenant(uuid.uuid4())
    legacy_warnings = [w for w in caught if "legacy single-tenant path" in str(w.message)]
    assert legacy_warnings == []
