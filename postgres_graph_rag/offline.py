"""Deterministic provider for local demos and evaluation.

The offline provider deliberately has no network or model dependency. It uses
signed feature hashing for embeddings and a caller-supplied extraction map so
the demo measures retrieval and graph behavior rather than provider variance.
"""

import hashlib
import math
import re
from typing import Iterable, List, Mapping, Optional, overload

from .extractor import Triplet
from .models import ProviderConfig


OFFLINE_CONFIG: ProviderConfig = {
    "extraction_model": "offline-fixture",
    "embedding_model": "offline-hash-1536",
    "dimension": 1536,
    "api_family": "offline",
}


class OfflineExtractor:
    """Drop-in extractor/embedding provider for repeatable local runs."""

    def __init__(
        self,
        triplets_by_text: Optional[Mapping[str, Iterable[Triplet]]] = None,
        config: Optional[ProviderConfig] = None,
    ) -> None:
        # {**OFFLINE_CONFIG} rather than dict(OFFLINE_CONFIG): the dict()
        # constructor loses ProviderConfig's TypedDict-specific field types
        # (e.g. "dimension" narrows to plain `object`, not `int`), which
        # then surfaced as real arithmetic-on-`object` errors below.
        self.config: ProviderConfig = config or {**OFFLINE_CONFIG}
        self._triplets = {
            text: list(triplets) for text, triplets in (triplets_by_text or {}).items()
        }
        self._last_usage = None

    @property
    def last_usage(self):
        return self._last_usage

    async def extract_triplets(self, text: str) -> List[Triplet]:
        return list(self._triplets.get(text, self._fallback_extract(text)))

    @overload
    async def get_embedding(self, text: str) -> List[float]: ...
    @overload
    async def get_embedding(self, text: List[str]) -> List[List[float]]: ...

    async def get_embedding(self, text: str | List[str]) -> List[float] | List[List[float]]:
        if isinstance(text, list):
            return [self._embed_one(value) for value in text]
        return self._embed_one(text)

    async def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        # Deterministic summary fallback for community/demo flows.
        compact = " ".join(prompt.split())
        return compact[: max(80, max_tokens * 4)]

    def _embed_one(self, text: str) -> List[float]:
        dimension = self.config["dimension"]
        vector = [0.0] * dimension
        tokens = re.findall(r"[\w-]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    @staticmethod
    def _fallback_extract(text: str) -> List[Triplet]:
        # Conservative fixture fallback; unknown prose produces no graph edge
        # rather than inventing a relationship.
        pattern = re.compile(
            r"([A-Z][\w-]*(?:\s+[A-Z][\w-]*)*)\s+"
            r"(depends_on|owned_by|caused|deployed|uses|runs_on|has_runbook)\s+"
            r"([A-Z][\w-]*(?:\s+[A-Z][\w-]*)*)"
        )
        return [Triplet(subject=s.strip(), predicate=p, object=o.strip()) for s, p, o in pattern.findall(text)]
