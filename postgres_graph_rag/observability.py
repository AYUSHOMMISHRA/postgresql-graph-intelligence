"""Observability hooks (v0.4): typed events, pluggable sinks, per-tenant
usage aggregation.

Deliberately not vendor-specific: `EventBus` fans events out to whatever
sinks the application registers (a logger, an in-memory aggregator, an
OpenTelemetry adapter, a custom one) rather than this library owning a
hard dependency on a particular observability backend.

Privacy by construction, not by convention: `Event.attributes` is meant for
counts, durations, ids, and other metadata — call sites in this codebase
never put raw document/chunk text, prompts, embeddings, API keys, or
database URLs into an event. There's no runtime redaction filter (nothing
enforces this at the type level either), so a custom call site that
violates this is possible; the convention is documented here because it's
the actual contract this module's built-in call sites follow.
"""

import inspect
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, Union, runtime_checkable

logger = logging.getLogger("postgres_graph_rag.events")

# Well-known event kinds. Not an exhaustive enum on purpose — sinks should
# treat unrecognized kinds as forward-compatible rather than erroring, so
# this list documents what exists today without locking the wire format.
INGESTION_STARTED = "ingestion.started"
INGESTION_COMPLETED = "ingestion.completed"
CHUNKING_COMPLETED = "chunking.completed"
CACHE_CLAIMED = "cache.claimed"
CACHE_HIT = "cache.hit"
CACHE_IN_PROGRESS = "cache.in_progress"
EXTRACTION_COMPLETED = "extraction.completed"
EXTRACTION_FAILED = "extraction.failed"
EMBEDDING_COMPLETED = "embedding.completed"
ENTITY_RESOLUTION_COMPLETED = "entity_resolution.completed"
DB_WRITE_COMPLETED = "db_write.completed"
RETRIEVAL_HYBRID_COMPLETED = "retrieval.hybrid_completed"
RETRIEVAL_TRAVERSAL_COMPLETED = "retrieval.traversal_completed"
RETRIEVAL_COMPLETED = "retrieval.completed"
ANSWER_COMPLETED = "answer.completed"
COMMUNITY_REFRESH_COMPLETED = "community.refresh_completed"
COMMUNITY_SUMMARIZE_COMPLETED = "community.summarize_completed"
RETRY_ATTEMPTED = "retry.attempted"
FAILURE = "failure"


@dataclass
class Event:
    kind: str
    correlation_id: str
    tenant_id: Optional[str] = None
    namespace: Optional[str] = None
    duration_ms: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


@runtime_checkable
class AsyncEventSink(Protocol):
    async def emit(self, event: Event) -> None: ...


class LoggingSink:
    """Default sink: structured log lines, no external dependency."""

    def __init__(self, level: int = logging.DEBUG):
        self.level = level

    def emit(self, event: Event) -> None:
        logger.log(
            self.level,
            "%s tenant=%s namespace=%s duration_ms=%s %s",
            event.kind, event.tenant_id, event.namespace, event.duration_ms, event.attributes,
        )


class UsageAggregator:
    """In-memory per-tenant/namespace counters and (when present in an
    event's attributes) token/cost usage — the "per-tenant and
    per-namespace usage aggregation" the plan calls for, without requiring
    an external metrics backend. Intended for lightweight self-hosting;
    swap in an OpenTelemetry/Prometheus sink for production-grade metrics.
    """

    def __init__(self):
        self._counts: Dict[tuple, Dict[str, int]] = {}
        self._tokens: Dict[tuple, int] = {}

    def emit(self, event: Event) -> None:
        key = (event.tenant_id, event.namespace)
        bucket = self._counts.setdefault(key, {})
        bucket[event.kind] = bucket.get(event.kind, 0) + 1
        tokens = event.attributes.get("tokens")
        if isinstance(tokens, int):
            self._tokens[key] = self._tokens.get(key, 0) + tokens

    def snapshot(self, tenant_id: Optional[str] = None, namespace: Optional[str] = None) -> Dict[str, Any]:
        key = (tenant_id, namespace)
        return {"counts": dict(self._counts.get(key, {})), "tokens": self._tokens.get(key, 0)}


def make_otel_sink(meter_name: str = "postgres_graph_rag"):
    """Returns an OpenTelemetry-backed sink (a counter-per-event-kind plus
    a duration histogram), or None if `opentelemetry` isn't installed.
    Optional by construction — this library never requires OTel."""
    try:
        from opentelemetry import metrics
    except ImportError:
        return None

    meter = metrics.get_meter(meter_name)
    counter = meter.create_counter("postgres_graph_rag.events", description="Event counts by kind")
    histogram = meter.create_histogram("postgres_graph_rag.duration_ms", description="Event durations")

    class _OtelSink:
        def emit(self, event: Event) -> None:
            attrs = {"kind": event.kind, "tenant_id": event.tenant_id or "", "namespace": event.namespace or ""}
            counter.add(1, attrs)
            if event.duration_ms is not None:
                histogram.record(event.duration_ms, attrs)

    return _OtelSink()


class EventBus:
    """Fans events out to registered sinks (sync or async). Never raises
    from `emit()` on a sink's own error — observability must not be able to
    break the operation it's observing."""

    def __init__(self, sinks: Optional[List[Union[EventSink, AsyncEventSink]]] = None):
        self.sinks: List[Union[EventSink, AsyncEventSink]] = sinks or [LoggingSink()]

    async def emit(self, event: Event) -> None:
        for sink in self.sinks:
            try:
                result = sink.emit(event)
                # inspect.isawaitable() (unlike a bare hasattr(result,
                # "__await__") check) is recognized by mypy as narrowing
                # `result` to Awaitable[Any], since sink.emit() can return
                # either None (a sync EventSink) or a coroutine (an
                # AsyncEventSink).
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001
                logger.exception("Event sink %r raised while handling %s", sink, event.kind)

    @asynccontextmanager
    async def timed(
        self, kind: str, correlation_id: str, tenant_id: Optional[str] = None,
        namespace: Optional[str] = None, **attributes: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Emits one event for the wrapped block, with duration_ms filled
        in automatically. `attributes` seeds the event; the yielded dict
        can be mutated inside the block to add more (e.g. a result count
        only known after the work completes)."""
        start = time.perf_counter()
        attrs = dict(attributes)
        try:
            yield attrs
        finally:
            await self.emit(
                Event(
                    kind=kind, correlation_id=correlation_id, tenant_id=tenant_id,
                    namespace=namespace, duration_ms=(time.perf_counter() - start) * 1000,
                    attributes=attrs,
                )
            )


def new_correlation_id() -> str:
    return str(uuid.uuid4())


NULL_EVENT_BUS = EventBus(sinks=[])
