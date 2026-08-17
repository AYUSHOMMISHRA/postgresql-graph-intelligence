import pytest

from postgres_graph_rag import OfflineExtractor
from postgres_graph_rag.extractor import Triplet


@pytest.mark.asyncio
async def test_offline_provider_is_deterministic_and_uses_fixture_triplets():
    extractor = OfflineExtractor(
        {"checkout depends on auth": [Triplet(subject="checkout", predicate="depends_on", object="auth")]}
    )
    first = await extractor.get_embedding("checkout depends on auth")
    second = await extractor.get_embedding("checkout depends on auth")
    assert first == second
    assert len(first) == 1536
    assert await extractor.extract_triplets("checkout depends on auth") == [
        Triplet(subject="checkout", predicate="depends_on", object="auth")
    ]


@pytest.mark.asyncio
async def test_offline_provider_does_not_invent_relationships():
    extractor = OfflineExtractor()
    assert await extractor.extract_triplets("plain incident prose without a relation") == []
