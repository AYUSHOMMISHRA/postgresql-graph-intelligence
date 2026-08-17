from typing import Literal, TypedDict, Optional, List

from .filters import MetadataFilter


class _RequiredProviderConfig(TypedDict):
    extraction_model: str
    embedding_model: str
    dimension: int


class ProviderConfig(_RequiredProviderConfig, total=False):
    # Explicit dispatch is required for OpenAI-compatible gateways: a
    # LiteLLM model alias need not contain "gpt" even though it uses the
    # OpenAI wire protocol.
    api_family: Literal["openai", "google", "offline"]


# Default configurations for the current era
OPENAI_DEFAULT_CONFIG: ProviderConfig = {
    "extraction_model": "gpt-5.6-luna",
    "embedding_model": "text-embedding-3-small",
    "dimension": 1536,
    "api_family": "openai",
}

GOOGLE_DEFAULT_CONFIG: ProviderConfig = {
    "extraction_model": "gemini-3.1-flash-lite",
    "embedding_model": "gemini-embedding-001",
    "dimension": 3072,
    "api_family": "google",
}


def build_litellm_config(
    *, extraction_model: str, embedding_model: str, dimension: int,
) -> ProviderConfig:
    """Build an OpenAI-compatible LiteLLM provider configuration.

    Model aliases and embedding dimensions are deployment-specific in a
    LiteLLM gateway, so this project intentionally has no guessed defaults.
    """
    if not extraction_model.strip():
        raise ValueError("LiteLLM extraction model must not be empty")
    if not embedding_model.strip():
        raise ValueError("LiteLLM embedding model must not be empty")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError("LiteLLM embedding dimension must be a positive integer")
    return {
        "extraction_model": extraction_model.strip(),
        "embedding_model": embedding_model.strip(),
        "dimension": dimension,
        "api_family": "openai",
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
