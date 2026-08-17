import hashlib
import math
import re
from typing import List, Any

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
