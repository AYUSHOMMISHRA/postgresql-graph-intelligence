import logging
from typing import List, Any, Optional, Callable
from .extractor import LLMExtractor
from .models import (
    ProviderConfig,
    OPENAI_DEFAULT_CONFIG,
    GOOGLE_DEFAULT_CONFIG,
    RetrievalConfig,
    DEFAULT_RETRIEVAL_CONFIG,
    IngestionConfig,
    DEFAULT_INGESTION_CONFIG,
)

logger = logging.getLogger("postgres_graph_rag")


def simple_chunker(
    text: str, size: int = 1000, overlap: int = 100
) -> List[str]:
    """Default simple character-based chunking."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


class PostgresGraphRAG:
    def __init__(
        self,
        postgres_url: str,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        config: Optional[ProviderConfig] = None,
        chunker: Optional[Callable[[str], List[str]]] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        ingestion_config: Optional[IngestionConfig] = None,
        runtime_url: Optional[str] = None,
        extractor: Optional[Any] = None,
    ):
        """
        Initializes the PostgresGraphRAG instance.
        This is a standard synchronous initialization.

        `runtime_url` is only needed for the multi-tenant, RLS-secured path
        (`for_tenant()` / `setup_secure()`): it must be the connection
        string for the restricted runtime role created by `setup_secure()`,
        never a superuser/owner URL — RLS does nothing for a role that can
        bypass it.

        `postgres_url` is accepted for constructor-signature compatibility
        with existing callers but is otherwise unused: it backed the legacy
        single-tenant engine (removed), and `setup_secure()`/`for_tenant()`
        take their own `admin_url`/`runtime_url` instead.
        """
        self._runtime_url = runtime_url
        self._secure_store = None  # lazily created by for_tenant()

        if config is None:
            config = (
                GOOGLE_DEFAULT_CONFIG
                if google_api_key
                else OPENAI_DEFAULT_CONFIG
            )

        # Dependency injection keeps the production provider path unchanged
        # while allowing deterministic offline demos/tests without network or
        # paid model calls.
        self.extractor = extractor or LLMExtractor(
            config=config,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
        )
        self.chunker = chunker or simple_chunker
        self.retrieval_config: RetrievalConfig = {
            **DEFAULT_RETRIEVAL_CONFIG,
            **(retrieval_config or {}),
        }
        self.ingestion_config: IngestionConfig = {
            **DEFAULT_INGESTION_CONFIG,
            **(ingestion_config or {}),
        }

    async def __aenter__(self):
        """Supports async context manager usage."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes resources when exiting the context."""
        await self.close()

    async def close(self):
        """Manually closes the database connection pool(s)."""
        if self._secure_store is not None:
            await self._secure_store.close()

    async def setup_secure(
        self,
        admin_url: str,
        runtime_role: str,
        runtime_password: str,
        migrate_legacy_data: bool = False,
    ):
        """Creates/upgrades the tenant-aware, RLS-secured schema (v0.3).

        Must be called with `admin_url` — a privileged (e.g. superuser)
        connection string, separate from `runtime_url` passed to the
        constructor — since this creates the restricted runtime role and
        installs RLS policies, which the runtime role itself must not be
        able to do. Safe to re-run.
        """
        from .tenancy import migrate_schema

        await migrate_schema(
            admin_url=admin_url,
            runtime_role=runtime_role,
            runtime_password=runtime_password,
            embedding_dimension=self.extractor.config["dimension"],
            migrate_legacy_data=migrate_legacy_data,
        )

    def _get_or_create_store(self):
        """Lazily builds (or returns the already-built) `SecureGraphStore`
        for this instance's `runtime_url`.

        This is the single construction path for the secure store: both
        `for_tenant()` below and `mcp_server.py`'s server lifespan call it,
        instead of each independently re-implementing lazy init by reaching
        into `self._secure_store`/`self._runtime_url` directly. Before this
        was extracted, the two copies diverged on the error path -- this
        one's missing-`runtime_url` check is the only one; the MCP lifespan
        used to skip it entirely and would construct a store around `None`.

        Requires `runtime_url` to have been passed to the constructor.
        """
        if not self._runtime_url:
            raise ValueError(
                "PostgresGraphRAG requires runtime_url to be set to the "
                "restricted RLS runtime role's connection string before a "
                "SecureGraphStore can be constructed (for_tenant() or the "
                "MCP server both need this)."
            )
        if self._secure_store is None:
            from .tenancy import SecureGraphStore, _vector_column_type as _vt

            vector_type = _vt(self.extractor.config["dimension"])
            self._secure_store = SecureGraphStore(self._runtime_url, vector_type=vector_type)
        return self._secure_store

    def for_tenant(self, tenant_id, event_bus=None):
        """Returns a `TenantGraphRAG` bound to `tenant_id` for the lifetime
        of the returned object — tenant_id is never a per-call argument
        again, so a single query can't accidentally target the wrong
        tenant. Requires `runtime_url` to have been passed to the
        constructor and `setup_secure()` to have been run at least once.

        `event_bus` (an `observability.EventBus`) is optional; omit it to
        get the default logging-only sink.
        """
        from .tenant_engine import TenantGraphRAG

        return TenantGraphRAG(
            tenant_id=tenant_id,
            store=self._get_or_create_store(),
            extractor=self.extractor,
            chunker=self.chunker,
            ingestion_config=self.ingestion_config,
            retrieval_config=self.retrieval_config,
            event_bus=event_bus,
        )
