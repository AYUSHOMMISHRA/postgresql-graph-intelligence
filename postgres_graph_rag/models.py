from typing import Literal, TypedDict, Optional, List

from .filters import MetadataFilter


class ProviderConfig(TypedDict):
    extraction_model: str
    embedding_model: str
    dimension: int


# Default configurations for the current era
OPENAI_DEFAULT_CONFIG: ProviderConfig = {
    "extraction_model": "gpt-5-nano-2025-08-07",
    "embedding_model": "text-embedding-3-small",
    "dimension": 1536,
}

GOOGLE_DEFAULT_CONFIG: ProviderConfig = {
    "extraction_model": "gemini-3.1-flash-lite",
    "embedding_model": "gemini-embedding-001",
    "dimension": 3072,
}


class RetrievalConfig(TypedDict, total=False):
    """Tunable knobs for the retrieval/traversal pipeline.

    All fields are optional; unset fields fall back to the defaults in
    ``DEFAULT_RETRIEVAL_CONFIG``.
    """

    top_k: int
    hops: int
    directed: bool
    relation_types: Optional[List[str]]
    exclude_relation_types: Optional[List[str]]
    min_weight: float
    score_decay: float
    max_context_nodes: int
    max_context_edges: int
    max_context_tokens: int
    graph_seed_chunks: int
    max_neighbors_per_node: int
    metadata_filter: MetadataFilter
    mode: Literal["vector", "hybrid", "hybrid_graph"]


DEFAULT_RETRIEVAL_CONFIG: RetrievalConfig = {
    "top_k": 5,
    "hops": 2,
    "directed": False,
    "relation_types": None,
    "exclude_relation_types": None,
    "min_weight": 0.0,
    "score_decay": 0.7,
    "max_context_nodes": 25,
    "max_context_edges": 50,
    "max_context_tokens": 4000,
    # Expanding every retrieved chunk makes unrelated top-k candidates into
    # equally strong roots. Start graph traversal from the best evidence.
    "graph_seed_chunks": 1,
    "max_neighbors_per_node": 20,
    "metadata_filter": None,
    "mode": "hybrid_graph",
}


class IngestionConfig(TypedDict, total=False):
    """Tunable knobs for the ingestion pipeline."""

    max_concurrent_extractions: int
    max_extraction_retries: int
    retry_base_delay: float
    skip_duplicate_chunks: bool
    fuzzy_entity_resolution: bool
    fuzzy_trgm_threshold: float
    fuzzy_embedding_threshold: float


DEFAULT_INGESTION_CONFIG: IngestionConfig = {
    "max_concurrent_extractions": 5,
    "max_extraction_retries": 3,
    "retry_base_delay": 1.0,
    "skip_duplicate_chunks": True,
    "fuzzy_entity_resolution": True,
    "fuzzy_trgm_threshold": 0.4,
    "fuzzy_embedding_threshold": 0.90,
}
