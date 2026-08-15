import os
import pytest
from postgres_graph_rag import PostgresGraphRAG
from postgres_graph_rag.models import (
    OPENAI_DEFAULT_CONFIG,
    GOOGLE_DEFAULT_CONFIG,
)
from dotenv import load_dotenv
from timing_utils import time_it, stats, time_func

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


@pytest.mark.skipif(not POSTGRES_URL, reason="POSTGRES_URL not set")
class TestIntegration:

    @pytest.fixture(autouse=True)
    def report_timing(self):
        yield
        stats.report()

    def _instrument_rag(self, rag: PostgresGraphRAG):
        # Instrument internal components
        rag.extractor.extract_triplets = time_func("LLM: Extract Triplets")(
            rag.extractor.extract_triplets
        )
        rag.extractor.get_embedding = time_func("LLM: Get Embedding")(
            rag.extractor.get_embedding
        )

        rag.db.upsert_node = time_func("DB: Upsert Node")(rag.db.upsert_node)
        rag.db.upsert_nodes_batch = time_func("DB: Upsert Nodes Batch")(
            rag.db.upsert_nodes_batch
        )
        rag.db.upsert_edge = time_func("DB: Upsert Edge")(rag.db.upsert_edge)
        rag.db.upsert_edges_batch = time_func("DB: Upsert Edges Batch")(
            rag.db.upsert_edges_batch
        )
        rag.db.vector_search = time_func("DB: Vector Search")(
            rag.db.vector_search
        )
        rag.db.traverse_graph = time_func("DB: Traverse Graph")(
            rag.db.traverse_graph
        )
        rag.db.setup_database = time_func("DB: Setup Schema")(
            rag.db.setup_database
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
    async def test_openai_e2e(self):
        # Frictionless instantiation
        rag = PostgresGraphRAG(
            postgres_url=POSTGRES_URL,
            openai_api_key=OPENAI_API_KEY,
            config=OPENAI_DEFAULT_CONFIG,
        )
        self._instrument_rag(rag)
        try:
            async with time_it("E2E: OpenAI Flow"):
                await self._run_e2e_flow(rag, "OpenAI")
        finally:
            await rag.close()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not GOOGLE_API_KEY, reason="GOOGLE_API_KEY not set")
    async def test_google_e2e(self):
        # Frictionless instantiation
        rag = PostgresGraphRAG(
            postgres_url=POSTGRES_URL,
            google_api_key=GOOGLE_API_KEY,
            config=GOOGLE_DEFAULT_CONFIG,
        )
        self._instrument_rag(rag)
        try:
            async with time_it("E2E: Google Flow"):
                await self._run_e2e_flow(rag, "Google")
        finally:
            await rag.close()

    async def _run_e2e_flow(self, rag, provider_name):
        await rag.db._init_pool()
        async with rag.db.pool.connection() as conn:
            async with conn.cursor() as cur:
                # ingested_chunks must be dropped alongside graph_edges/
                # graph_nodes: it's never touched by rag.setup() (only
                # created if missing) and idempotent ingestion is keyed on
                # content hash, not on graph_nodes/graph_edges existing —
                # leaving it in place makes this test poison every
                # subsequent run against the same database (this exact
                # text+namespace gets silently skipped as "already
                # ingested" instead of re-extracted), which is a real,
                # deterministic failure this test hit, not test flakiness.
                await cur.execute("DROP TABLE IF EXISTS ingested_chunks CASCADE")
                await cur.execute("DROP TABLE IF EXISTS graph_edges CASCADE")
                await cur.execute("DROP TABLE IF EXISTS graph_nodes CASCADE")
                await conn.commit()

        await rag.setup()

        namespace = f"test-{provider_name.lower()}"
        test_text = f"The {provider_name} team developed a new RAG system. This system uses Postgres."

        # Ingest with namespace
        async with time_it(f"Action: Ingest ({provider_name})"):
            await rag.add_texts(test_text, namespace=namespace)

        # Query with same namespace
        async with time_it(f"Action: Query ({provider_name})"):
            context = await rag.query(
                f"What did the {provider_name} team develop?",
                namespace=namespace,
                hops=2,
            )

        assert provider_name in context
        assert "Postgres" in context or "PostgreSQL" in context

        # Verify other namespace is empty
        await rag.query("Any team?", namespace="other-empty-ns")
