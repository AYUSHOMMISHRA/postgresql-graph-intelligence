"""Tenant-scoped engine facade (v0.3) on top of `tenancy.SecureGraphStore`.

`TenantGraphRAG` is what `PostgresGraphRAG.for_tenant(tenant_id)` returns —
it never takes `tenant_id` as a per-call argument again after construction,
so a caller can't accidentally pass the wrong tenant to a single query while
correctly scoping everything else (the plan's rule #1: never expose
tenant_id as a freely overridable argument after for_tenant()).
"""

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import psycopg

from .extractor import LLMExtractor
from .models import IngestionConfig, RetrievalConfig
from .observability import EventBus, NULL_EVENT_BUS, new_correlation_id
from . import observability as obs
from .tenancy import DEFAULT_LEASE_SECONDS, SecureGraphStore, content_hash

logger = logging.getLogger("postgres_graph_rag")

PROMPT_VERSION = "v1"
MAX_DOCUMENT_CHARS = 2_000_000
MAX_NAMESPACE_LENGTH = 255
MAX_SOURCE_ID_LENGTH = 512
LEASE_WAIT_SECONDS = 15.0


@dataclass
class TenantRetrievedChunk:
    id: str
    document_id: str
    source_id: Optional[str]
    ordinal: int
    content: str
    rrf_score: float
    lexical_rank: Optional[int]
    semantic_rank: Optional[int]


@dataclass
class TenantRetrievedNode:
    id: str
    content: str
    metadata: Dict[str, Any]
    hop_distance: int
    score: float


@dataclass
class TenantRetrievedEdge:
    source_id: str
    target_id: str
    source_content: str
    target_content: str
    relation: str
    weight: float
    score: float = 0.0
    id: str = ""


@dataclass
class RetrievalTrace:
    mode: str
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    traversal_ms: float = 0.0
    search_candidates: int = 0
    seed_scores: Dict[str, float] = field(default_factory=dict)
    context_tokens: int = 0
    context_truncated: bool = False


@dataclass
class Citation:
    source_id: str
    chunk_id: str
    ordinal: int
    excerpt: str


@dataclass
class AnswerResult:
    answer: str
    citations: List[Citation]
    retrieval: "TenantRetrievalResult"
    grounded: bool
    usage: Dict[str, int]
    latency_ms: float
    # Why `grounded` is False (None when it's True). There are several
    # distinct abstention paths that all produce the identical
    # "Insufficient evidence..." answer text, and conflating them made a
    # real incident hard to diagnose: a genuine anti-hallucination catch
    # (a named identifier from the question absent from all evidence) looks
    # identical from the outside to the model simply failing to cite
    # correctly, unless this is surfaced. One of:
    #   "no_evidence_retrieved"    — retrieve() returned zero chunks
    #   "missing_query_anchor"     — a question identifier isn't in any
    #                                retrieved evidence (LLM was never called)
    #   "empty_model_response"     — the model returned no visible text at
    #                                all; for reasoning-tier models this
    #                                usually means max_answer_tokens was
    #                                consumed entirely by invisible
    #                                reasoning tokens (raise the budget),
    #                                not a citation-formatting problem
    #   "invalid_or_missing_citation" — model's response had no citation, or
    #                                one that doesn't match retrieved evidence
    #   "model_abstained"          — model itself returned the abstention text
    abstain_reason: Optional[str] = None


@dataclass
class PathStep:
    node: str
    relation: Optional[str]  # the edge crossed to *arrive* at `node`; None for the first step


@dataclass
class ExplainedPath:
    """The reconstructed reasoning path between two named entities — "why
    did the system connect Node A to Node C" as an actual ordered sequence
    of hops, not just a bag of scored neighbors."""

    steps: List[PathStep]
    score: float
    hop_distance: int

    def __str__(self) -> str:
        parts = [self.steps[0].node]
        for step in self.steps[1:]:
            parts.append(f"--{step.relation}-->")
            parts.append(step.node)
        return " ".join(parts)


@dataclass
class TenantRetrievalResult:
    """Structured, citable evidence: which chunks (and therefore documents)
    were retrieved, plus the graph neighborhood seeded from the entities
    those chunks actually mention."""

    chunks: List[TenantRetrievedChunk] = field(default_factory=list)
    nodes: List[TenantRetrievedNode] = field(default_factory=list)
    edges: List[TenantRetrievedEdge] = field(default_factory=list)
    trace: Optional[RetrievalTrace] = None

    def to_context_string(self) -> str:
        if not self.chunks and not self.nodes:
            return "No relevant context found."
        parts = ["Relevant Passages:"]
        for c in self.chunks:
            cite = f"[{c.source_id or c.document_id}#{c.ordinal}]"
            parts.append(f"- {cite} {c.content}")
        if self.nodes:
            parts.append("\nRelated Entities:")
            for n in self.nodes:
                parts.append(f"- {n.content} (score: {n.score:.2f}, hops: {n.hop_distance})")
        if self.edges:
            parts.append("\nRelationships:")
            for e in self.edges:
                parts.append(f"- {e.source_content} --[{e.relation}]--> {e.target_content}")
        return "\n".join(parts)


def _estimate_tokens(text: str) -> int:
    """Deterministic dependency-free context estimate (~4 chars/token)."""
    return max(1, (len(text) + 3) // 4)


def _citation_marker(chunk: TenantRetrievedChunk) -> str:
    return f"[{chunk.source_id or chunk.document_id}#{chunk.ordinal}]"


def _identifier_anchors(question: str) -> List[str]:
    """Identifiers with separators are high-precision premise anchors.

    If a query names ``database-042`` but no retrieved evidence contains it,
    the system must not answer a nearby fact about ``checkout-service-042``.
    """
    return list(dict.fromkeys(
        match.lower()
        for match in re.findall(r"\b[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)+\b", question)
    ))


def _normalize_identifier(text: str) -> str:
    """Collapses hyphen/underscore/dot separators to spaces before an
    identifier containment check, so matching isn't defeated by superficial
    formatting differences — e.g. a question naming ``checkout-service``
    while retrieved prose describes it as ``checkout service``.

    This is a real failure mode, not a hypothetical: a live end-to-end run
    abstained with "Insufficient evidence" purely because of this
    formatting mismatch (the identifier *was* in the evidence, just not
    byte-for-byte), not because the evidence was actually missing. Without
    this normalization, `_identifier_anchors`' exact-substring check is
    stricter than the anti-hallucination guarantee it's meant to enforce.
    """
    return re.sub(r"[-_.]+", " ", text.lower())


def _graph_seed_chunks(
    question: str,
    chunks: List[TenantRetrievedChunk],
    limit: int,
) -> List[TenantRetrievedChunk]:
    """Prefer retrieved evidence containing exact query identifiers.

    PostgreSQL FTS can rank fluent ownership prose above an exact hyphenated
    identifier because punctuation is tokenized. The graph should start from
    the retrieved chunk that actually contains the named system/deploy ID.
    """
    anchors = _identifier_anchors(question)
    if not anchors:
        return chunks[:limit]
    ranked = sorted(
        enumerate(chunks),
        key=lambda item: (
            -sum(_normalize_identifier(anchor) in _normalize_identifier(item[1].content) for anchor in anchors),
            item[0],
        ),
    )
    return [chunk for _, chunk in ranked[:limit]]


class TenantGraphRAG:
    def __init__(
        self,
        tenant_id: uuid.UUID,
        store: SecureGraphStore,
        extractor: LLMExtractor,
        chunker: Callable[[str], List[str]],
        ingestion_config: IngestionConfig,
        retrieval_config: RetrievalConfig,
        event_bus: Optional[EventBus] = None,
    ):
        self.tenant_id = tenant_id
        self._store = store
        self._extractor = extractor
        self._chunker = chunker
        self._ingestion_config = ingestion_config
        self._retrieval_config = retrieval_config
        self._event_bus = event_bus or NULL_EVENT_BUS

    async def _extract_with_retry(self, chunk: str):
        max_retries = self._ingestion_config["max_extraction_retries"]
        base_delay = self._ingestion_config["retry_base_delay"]
        last_exc = None
        for attempt in range(max_retries):
            try:
                return await self._extractor.extract_triplets(chunk)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2**attempt))
        logger.warning("Extraction failed for chunk after %d attempts: %s", max_retries, last_exc)
        return None

    async def _embed(self, texts, correlation_id: str, namespace: str):
        """Wraps `LLMExtractor.get_embedding` and emits an
        `EMBEDDING_COMPLETED` event with token usage when the provider
        reports it (`last_usage` is `None` for providers/endpoints that
        don't — e.g. Google's embedding API — in which case `tokens` is
        just omitted rather than faked as 0)."""
        tid_str = str(self.tenant_id)
        result = await self._extractor.get_embedding(texts)
        usage = self._extractor.last_usage
        attrs = {"count": len(texts) if isinstance(texts, list) else 1}
        if usage:
            attrs["tokens"] = usage["total_tokens"]
        await self._event_bus.emit(obs.Event(obs.EMBEDDING_COMPLETED, correlation_id, tid_str, namespace, attributes=attrs))
        return result

    async def add_document_detailed(
        self,
        text: str,
        namespace: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the legacy ingestion report plus operator-facing status."""
        started = asyncio.get_running_loop().time()
        report = await self.add_document(text, namespace, source_id, metadata)
        return {
            **report,
            "status": "skipped" if report.get("skipped") else "active",
            "duration_ms": round((asyncio.get_running_loop().time() - started) * 1000, 2),
        }

    async def add_document(
        self,
        text: str,
        namespace: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingests one document under `source_id`. Re-ingesting the same
        `source_id` with identical text is a cheap no-op (document-level
        content hash comparison, not just per-chunk); with changed text, all
        of that document's chunks are atomically replaced and re-extracted.

        Returns a small report dict: {"skipped": bool, "chunks": int,
        "triplets": int}.
        """
        if not isinstance(text, str) or len(text) > MAX_DOCUMENT_CHARS:
            raise ValueError(
                f"text must be a string no longer than {MAX_DOCUMENT_CHARS} characters"
            )
        if not namespace or len(namespace) > MAX_NAMESPACE_LENGTH:
            raise ValueError(f"namespace must be 1-{MAX_NAMESPACE_LENGTH} characters")
        if not source_id or len(source_id) > MAX_SOURCE_ID_LENGTH:
            raise ValueError(f"source_id must be 1-{MAX_SOURCE_ID_LENGTH} characters")

        correlation_id = new_correlation_id()
        tid_str, ns_str = str(self.tenant_id), namespace

        await self._event_bus.emit(obs.Event(
            obs.INGESTION_STARTED, correlation_id, tid_str, ns_str, attributes={"source_id": source_id}
        ))
        async with self._event_bus.timed(
            obs.INGESTION_COMPLETED, correlation_id, tid_str, ns_str, source_id=source_id
        ) as attrs:
            report = await self._add_document_inner(text, namespace, source_id, metadata, correlation_id)
            attrs.update(report)
            return report

    async def _add_document_inner(
        self, text: str, namespace: str, source_id: str,
        metadata: Optional[Dict[str, Any]], correlation_id: str,
    ) -> Dict[str, Any]:
        tid_str = str(self.tenant_id)
        doc_hash = content_hash(text)

        # Read-only peek: does *not* commit the new hash. Committing it here
        # (as a prior version of this method did, via upsert_document before
        # chunking) created a permanent-retry hole — if chunking/embedding
        # failed afterward, the hash was already updated, so a retry with the
        # same text would see content_changed=False and skip forever, with
        # no chunk rows for retry_failed_chunks() to recover. The new hash is
        # only persisted once chunking and embedding have already succeeded,
        # in the same transaction as the chunk rows themselves (below).
        existing_hash = await self._store.get_document_content_hash(self.tenant_id, namespace, source_id)
        if existing_hash == doc_hash:
            return {"skipped": True, "chunks": 0, "triplets": 0}

        chunks = self._chunker(text)
        if len(chunks) > 5000:
            raise ValueError("document produces more than 5000 chunks")
        if chunks:
            await self._event_bus.emit(obs.Event(
                obs.CHUNKING_COMPLETED, correlation_id, tid_str, namespace, attributes={"chunk_count": len(chunks)}
            ))

        # `metadata` is stored on each chunk row specifically so
        # retry_failed_chunks() can recover it later without the caller
        # re-supplying it.
        chunk_embeddings = await self._embed(chunks, correlation_id, namespace) if chunks else []
        chunk_payload = [
            {"content": c, "embedding": emb, "metadata": metadata or {}}
            for c, emb in zip(chunks, chunk_embeddings)
        ]

        # Publish the new hash and its chunks atomically: one transaction, so
        # readers (and retries) only ever see the complete old document or
        # the complete new one, never a hash pointing at chunks that don't
        # exist yet (or don't exist anymore). The advisory lock (released
        # automatically at commit/rollback) serializes concurrent publishers
        # of this exact document so upsert_document()'s content_changed
        # check reads a fresh, not stale, snapshot — see its docstring.
        async with self._store.tenant_connection(self.tenant_id) as conn:
            await self._store.lock_document_for_publication(self.tenant_id, namespace, source_id, connection=conn)
            doc = await self._store.upsert_document(
                self.tenant_id, namespace, source_id, doc_hash, metadata, connection=conn,
            )
            if not doc["content_changed"]:
                # A concurrent call already published this exact content
                # (e.g. a racing retry) between our peek and this write.
                return {"skipped": True, "chunks": 0, "triplets": 0}
            chunk_ids = await self._store.replace_chunks(
                self.tenant_id, doc["id"], namespace, chunk_payload, connection=conn,
            )

        if not chunks:
            # No extraction to run: the document is graph-ready as soon as
            # it's published, trivially (there's nothing to extract from).
            # This particular call can't actually be superseded (there's no
            # extraction delay between publish and this status write), but
            # status_applied is still reported honestly rather than assumed.
            applied = await self._store.set_extraction_status(
                self.tenant_id, doc["id"], "ready", expected_content_hash=doc_hash,
            )
            return {
                "skipped": False, "chunks": 0, "triplets": 0,
                "extraction_status": "ready", "status_applied": applied,
            }

        # From here on, hash/chunks are already durably published (the
        # atomic step above). Extraction is a separate, eventually-consistent
        # stage: it involves LLM calls that can take seconds to minutes, so
        # it deliberately runs outside any database transaction — a document
        # is "text-searchable" the moment the block above commits, and
        # becomes "graph-ready" only once this finishes. `extraction_status`
        # is how a caller tells the two apart instead of them looking
        # identical from outside.
        chunk_results = await self._run_extraction(chunks, namespace, correlation_id)
        total_triplets = await self._wire_entities(chunk_ids, chunks, chunk_results, namespace, metadata, correlation_id)

        failed_count = sum(1 for r in chunk_results if r is None)
        if failed_count == 0:
            status, error = "ready", None
        elif failed_count == len(chunk_results):
            status, error = "failed", f"extraction failed for all {failed_count} chunks"
        else:
            status, error = "partial", f"extraction failed for {failed_count}/{len(chunk_results)} chunks"
        # expected_content_hash guards against exactly this scenario: this
        # extraction has been running against revision B's chunks; a
        # concurrent revision C got published (and reset extraction_status
        # to 'pending' for C) while we were still working; we finish late.
        # filter_existing_chunk_ids() (inside _wire_entities, above) already
        # stopped B's facts from attaching to C's chunks — this stops B's
        # verdict from overwriting C's status row too. `status_applied`
        # reports which happened: `status` here is always what *this call's*
        # own extraction concluded, not necessarily the current document's
        # actual status — when status_applied is False, the caller's own
        # work was superseded and the document's real current status is
        # whatever the newer revision's own processing has (or hasn't yet)
        # determined, not this value.
        applied = await self._store.set_extraction_status(
            self.tenant_id, doc["id"], status, error=error, increment_attempts=True,
            expected_content_hash=doc_hash,
        )

        return {
            "skipped": False, "chunks": len(chunks), "triplets": total_triplets,
            "extraction_status": status, "status_applied": applied,
        }

    async def _run_extraction(
        self, chunks: List[str], namespace: str, correlation_id: str
    ) -> List[Optional[List[Dict[str, Any]]]]:
        """Runs (bounded-concurrency, lease-cached, retried) extraction over
        `chunks`, returning one triplet-list-or-None per chunk in order.
        Shared by `add_document()` (fresh ingestion) and
        `retry_failed_chunks()` (re-attempting chunks whose extraction
        never completed) — the extraction logic itself doesn't know or care
        which case it's in."""
        tid_str = str(self.tenant_id)
        model = self._extractor.config["extraction_model"]
        provider = "openai" if "gpt" in model else "google"
        lease_owner = f"{uuid.uuid4()}"
        semaphore = asyncio.Semaphore(self._ingestion_config["max_concurrent_extractions"])

        async def _extract_chunk(chunk: str):
            chash = content_hash(chunk)
            claim = await self._store.claim_extraction(
                self.tenant_id, chash, provider, model, PROMPT_VERSION, lease_owner,
                lease_seconds=DEFAULT_LEASE_SECONDS,
            )
            waited = 0.0
            while claim["status"] == "in_progress" and waited < LEASE_WAIT_SECONDS:
                await asyncio.sleep(0.25)
                waited += 0.25
                claim = await self._store.claim_extraction(
                    self.tenant_id, chash, provider, model, PROMPT_VERSION, lease_owner,
                    lease_seconds=DEFAULT_LEASE_SECONDS,
                )
            if claim["status"] == "done":
                await self._event_bus.emit(obs.Event(obs.CACHE_HIT, correlation_id, tid_str, namespace))
                return claim["result"]
            if claim["status"] == "in_progress":
                # Another worker still owns the lease after the bounded wait;
                # leave this chunk retryable rather than duplicating a paid
                # provider call.
                await self._event_bus.emit(obs.Event(obs.CACHE_IN_PROGRESS, correlation_id, tid_str, namespace))
                logger.info("Extraction lease remained in progress after %.1fs; deferring chunk.", LEASE_WAIT_SECONDS)
                return None
            await self._event_bus.emit(obs.Event(obs.CACHE_CLAIMED, correlation_id, tid_str, namespace))

            async with semaphore:
                triplets = await self._extract_with_retry(chunk)
            usage = self._extractor.last_usage  # read on this task, right after its own call
            if triplets is None:
                await self._store.fail_extraction(self.tenant_id, chash, provider, model, PROMPT_VERSION, lease_owner)
                await self._event_bus.emit(obs.Event(obs.EXTRACTION_FAILED, correlation_id, tid_str, namespace))
                return None
            result = [t.model_dump() for t in triplets]
            await self._store.complete_extraction(self.tenant_id, chash, provider, model, PROMPT_VERSION, lease_owner, result)
            extraction_attrs = {"triplet_count": len(result)}
            if usage:
                extraction_attrs["tokens"] = usage["total_tokens"]
            await self._event_bus.emit(obs.Event(
                obs.EXTRACTION_COMPLETED, correlation_id, tid_str, namespace,
                attributes=extraction_attrs,
            ))
            return result

        return await asyncio.gather(*[_extract_chunk(c) for c in chunks])

    async def _wire_entities(
        self,
        chunk_ids: List[str],
        chunks: List[str],
        chunk_results: List[Optional[List[Dict[str, Any]]]],
        namespace: str,
        metadata: Optional[Dict[str, Any]],
        correlation_id: str,
    ) -> int:
        """Resolves entities/edges from successfully-extracted chunks and
        records provenance. `chunks` (raw text) is accepted alongside
        `chunk_ids` only for symmetry with the extraction step; it isn't
        used here directly (kept as a parameter so this stays easy to call
        with either a fresh document's chunks or a retry batch's).

        Guards against the stale-extraction race: extraction is a
        long-running, out-of-transaction LLM call, so a *concurrent*
        re-ingestion of the same document can call `replace_chunks()`
        (deleting these exact chunk rows) while this call is still in
        flight. Chunk ids that no longer exist by the time we're ready to
        wire them are dropped rather than wired to a superseded revision.
        """
        if chunk_ids:
            still_current = set(await self._store.filter_existing_chunk_ids(self.tenant_id, chunk_ids))
            if len(still_current) != len(chunk_ids):
                kept = [
                    (cid, chunk, results)
                    for cid, chunk, results in zip(chunk_ids, chunks, chunk_results)
                    if cid in still_current
                ]
                logger.info(
                    "%d of %d chunks were superseded by a concurrent update before "
                    "extraction finished; skipping their entity/edge wiring.",
                    len(chunk_ids) - len(kept), len(chunk_ids),
                )
                chunk_ids = [k[0] for k in kept]
                chunks = [k[1] for k in kept]
                chunk_results = [k[2] for k in kept]

        unique_entities = set()
        chunk_entity_names: List[List[str]] = []
        chunk_edge_keys: List[List[tuple]] = []
        all_edges_raw = []
        total_triplets = 0
        for triplets in chunk_results:
            names_here = []
            edge_keys_here = []
            if triplets:
                for t in triplets:
                    unique_entities.add(t["subject"])
                    unique_entities.add(t["object"])
                    names_here.append(t["subject"])
                    names_here.append(t["object"])
                    all_edges_raw.append(t)
                    edge_keys_here.append((t["subject"], t["predicate"], t["object"]))
                    total_triplets += 1
            chunk_entity_names.append(names_here)
            chunk_edge_keys.append(edge_keys_here)

        if not unique_entities:
            return 0

        entity_list = list(unique_entities)
        entity_embeddings = await self._embed(entity_list, correlation_id, namespace)
        entities_payload = [
            {"content": e, "embedding": emb, "metadata": metadata} for e, emb in zip(entity_list, entity_embeddings)
        ]
        content_to_id = await self._store.resolve_and_upsert_nodes(
            self.tenant_id,
            namespace,
            entities_payload,
            fuzzy=self._ingestion_config["fuzzy_entity_resolution"],
            trgm_threshold=self._ingestion_config["fuzzy_trgm_threshold"],
            embedding_threshold=self._ingestion_config["fuzzy_embedding_threshold"],
        )

        edges_data = [
            {
                "source_id": content_to_id[t["subject"]],
                "target_id": content_to_id[t["object"]],
                "relation": t["predicate"],
                "metadata": metadata,
            }
            for t in all_edges_raw
            if t["subject"] != t["object"]
        ]
        edge_ids = await self._store.upsert_edges(
            self.tenant_id, namespace, edges_data, evidence_backed=True
        )

        try:
            for chunk_id, names, triplet_keys in zip(chunk_ids, chunk_entity_names, chunk_edge_keys):
                node_ids = [content_to_id[n] for n in names if n in content_to_id]
                edge_ids_here = [
                    edge_ids[(content_to_id[s], content_to_id[o], p)]
                    for s, p, o in triplet_keys
                    if s != o and (content_to_id[s], content_to_id[o], p) in edge_ids
                ]
                try:
                    if node_ids:
                        await self._store.record_entity_mentions(self.tenant_id, chunk_id, node_ids)
                    if edge_ids_here:
                        await self._store.record_edge_mentions(self.tenant_id, chunk_id, edge_ids_here)
                except psycopg.errors.ForeignKeyViolation:
                    # This exact chunk was deleted (a concurrent
                    # replace_chunks) in the narrow window between the
                    # filter_existing_chunk_ids check above and this insert.
                    # Same stale-revision case, just caught by the database
                    # instead of our own pre-check — skip this chunk's
                    # mentions rather than fail the whole batch.
                    logger.info(
                        "Chunk %s was superseded mid-wiring (foreign key violation "
                        "recording its mentions); skipping.", chunk_id,
                    )
        finally:
            # An edge is created (upsert_edges, above) before any mention
            # backs it. If every chunk that would have supplied its first
            # mention was dropped (stale-chunk skip, a caught FK violation,
            # or *any other* error aborting this loop early — hence
            # `finally`, not just the happy path), the edge is left with
            # zero support — and, since traversal's min_weight defaults to
            # 0.0, it would otherwise be silently traversable with no
            # supporting evidence at all. Prune any such leftovers now,
            # whether or not the loop above finished cleanly.
            if edge_ids:
                await self._store.prune_unsupported_edges(self.tenant_id, list(edge_ids.values()))

        return total_triplets

    async def retry_failed_chunks(self, namespace: str, source_id: Optional[str] = None) -> Dict[str, Any]:
        """Re-attempts extraction for chunks that were already durably
        stored (via a prior `add_document()` call) but whose extraction
        never completed successfully — either it hasn't been attempted yet,
        or a prior attempt failed and `fail_extraction()` reset it to
        retryable. Unlike calling `add_document()` again, this does not
        require the caller to still have the original document text: the
        chunk content and its ingestion-time metadata are read back from
        `document_chunks` itself.

        `source_id` narrows retry to one document; omit it to sweep the
        whole namespace.
        """
        model = self._extractor.config["extraction_model"]
        provider = "openai" if "gpt" in model else "google"
        candidates = await self._store.find_unfinished_chunks(
            self.tenant_id,
            namespace,
            source_id,
            provider=provider,
            model=model,
            prompt_version=PROMPT_VERSION,
        )
        if not candidates:
            return {"retried": 0, "triplets": 0}

        correlation_id = new_correlation_id()
        chunk_ids = [c["id"] for c in candidates]
        texts = [c["content"] for c in candidates]
        # All candidates share the ingestion-time metadata stored per
        # chunk; retry re-applies each chunk's own metadata individually
        # rather than requiring one value for the whole batch.
        chunk_results = await self._run_extraction(texts, namespace, correlation_id)

        total = 0
        for chunk_id, text, result, candidate in zip(chunk_ids, texts, chunk_results, candidates):
            total += await self._wire_entities([chunk_id], [text], [result], namespace, candidate.get("metadata"), correlation_id)

        # This sweep may span one document (source_id given) or the whole
        # namespace; recompute extraction_status per affected document from
        # what's *actually* still unfinished afterward, rather than assuming
        # success from this batch alone (a chunk can fail again).
        #
        # The CAS guard binds to the hash *captured when these candidates
        # were fetched* (document_content_hash, from find_unfinished_chunks),
        # not one re-read right before the write. Re-reading "whatever's
        # current" at write time defeats the guard entirely: if a concurrent
        # add_document() published a newer revision while this retry sweep
        # was running, "current" would simply *be* that newer hash, so the
        # CAS would trivially match and overwrite the newer revision's
        # status with a verdict about the old one's chunks — exactly the
        # bug expected_content_hash exists to prevent. Binding to the
        # captured hash means a supersession during the sweep correctly
        # turns this into a no-op instead.
        affected_documents = {
            (c["document_id"], c["source_id"], c["document_content_hash"]) for c in candidates
        }
        for document_id, doc_source_id, expected_hash in affected_documents:
            remaining = await self._store.find_unfinished_chunks(
                self.tenant_id, namespace, doc_source_id,
                provider=provider, model=model, prompt_version=PROMPT_VERSION,
            )
            if not remaining:
                await self._store.set_extraction_status(
                    self.tenant_id, document_id, "ready", expected_content_hash=expected_hash,
                )
            else:
                await self._store.set_extraction_status(
                    self.tenant_id, document_id, "partial",
                    error=f"{len(remaining)} chunk(s) still failing extraction",
                    increment_attempts=True,
                    expected_content_hash=expected_hash,
                )

        return {"retried": len(candidates), "triplets": total}

    async def delete_document(self, namespace: str, source_id: str) -> None:
        await self._store.delete_document(self.tenant_id, namespace, source_id)

    async def add_record(
        self,
        namespace: str,
        source_id: str,
        entity_type: str,
        record: Dict[str, Any],
        name_field: str = "name",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingests one structured record (e.g. a database row) as an entity,
        deterministically — no LLM call, no extraction to get wrong. The
        entity's display name is `record[name_field]` (falling back to
        `f"{entity_type}:{source_id}"` if that field is absent), and the
        full `record` dict is stored as node metadata under `attributes`
        alongside `entity_type` and `source_id`, so it's queryable later
        without re-parsing anything.

        Also creates a document/chunk pair from a deterministic text
        rendering of the record, purely so hybrid search can find it by
        keyword/semantic similarity like any other ingested text — the
        graph entity itself is what `add_triplets()` and `retrieve()`'s
        graph traversal actually operate on.

        Re-calling with an unchanged `record` is a cheap no-op (same
        content-hash comparison `add_document` uses). Use `add_triplets()`
        separately to relate this entity to others — `add_record()` itself
        never creates edges.
        """
        correlation_id = new_correlation_id()
        entity_name = record.get(name_field) or f"{entity_type}:{source_id}"
        rendered = f"{entity_type}: " + ", ".join(f"{k}={v}" for k, v in record.items())
        doc_hash = content_hash(rendered)

        # Same read-then-atomically-publish pattern as _add_document_inner:
        # peeking the hash (instead of committing it up front) means a
        # failure in embedding/node-resolution below leaves the old hash in
        # place, so a retry with the same record is correctly retried
        # rather than silently skipped.
        existing_hash = await self._store.get_document_content_hash(self.tenant_id, namespace, source_id)
        if existing_hash == doc_hash:
            return {"skipped": True, "entity_id": None}

        node_metadata = {
            "entity_type": entity_type,
            "source_id": source_id,
            "attributes": record,
            **(metadata or {}),
        }
        # Embeddings are the only provider call add_record() makes (there's
        # no LLM extraction here) — computed outside the transaction so a
        # slow embedding call never holds a connection/lock, same reasoning
        # as _add_document_inner.
        entity_embedding = await self._embed(entity_name, correlation_id, namespace)
        chunk_embedding = await self._embed(rendered, correlation_id, namespace)

        # Node resolution, document/chunk publication, and mention recording
        # all happen in one transaction. Previously node resolution ran
        # *before* this transaction: a failure publishing the document could
        # leave a newly-created or newly-updated entity node visible with no
        # backing document/mention at all. Folding it in means a failure
        # anywhere in this block rolls back the node change too — no reader
        # ever sees an entity that this call didn't actually finish adding.
        async with self._store.tenant_connection(self.tenant_id) as conn:
            await self._store.lock_document_for_publication(self.tenant_id, namespace, source_id, connection=conn)
            content_to_id = await self._store.resolve_and_upsert_nodes(
                self.tenant_id,
                namespace,
                [{"content": entity_name, "embedding": entity_embedding, "metadata": node_metadata}],
                fuzzy=self._ingestion_config["fuzzy_entity_resolution"],
                trgm_threshold=self._ingestion_config["fuzzy_trgm_threshold"],
                embedding_threshold=self._ingestion_config["fuzzy_embedding_threshold"],
                connection=conn,
            )
            entity_id = content_to_id[entity_name]

            doc = await self._store.upsert_document(
                self.tenant_id, namespace, source_id, doc_hash, metadata, connection=conn,
            )
            if not doc["content_changed"]:
                # A concurrent call already published this exact content;
                # the node resolution above is idempotent (same entity name
                # resolves to the same node either way), so just let this
                # transaction commit as a no-op document write and report skip.
                return {"skipped": True, "entity_id": entity_id}
            chunk_ids = await self._store.replace_chunks(
                self.tenant_id, doc["id"], namespace,
                [{"content": rendered, "embedding": chunk_embedding, "metadata": {"entity_type": entity_type}}],
                connection=conn,
            )
            await self._store.record_entity_mentions(self.tenant_id, chunk_ids[0], [entity_id], connection=conn)
            # No extraction stage for a deterministic record: graph-ready
            # the instant it's published.
            await self._store.set_extraction_status(self.tenant_id, doc["id"], "ready", connection=conn)

        return {"skipped": False, "entity_id": entity_id}

    async def add_triplets(self, triplets: List[Dict[str, Any]], namespace: str) -> Dict[str, Any]:
        """Creates relationships directly — deterministic edges supplied by
        the caller, not extracted by an LLM. Each item is
        `{"subject": str, "predicate": str, "object": str, "metadata":
        {...}}` (`metadata` optional). Subjects/objects are resolved
        through the same layered entity resolution as LLM-extracted
        triplets, so a subject naming an entity `add_record()` (or a prior
        `add_document()`) already created resolves to that same node rather
        than creating a duplicate.

        Use this for facts you already know structurally (e.g. a foreign
        key relationship in a source table) instead of asking an LLM to
        re-discover them from text — cheaper and can't hallucinate a
        relationship you're supplying directly.
        """
        if not triplets:
            return {"edges": 0, "entities": 0}

        unique_entities = set()
        for t in triplets:
            unique_entities.add(t["subject"])
            unique_entities.add(t["object"])

        entity_list = list(unique_entities)
        embeddings = await self._embed(entity_list, new_correlation_id(), namespace)
        entities_payload = [{"content": e, "embedding": emb, "metadata": {}} for e, emb in zip(entity_list, embeddings)]
        content_to_id = await self._store.resolve_and_upsert_nodes(
            self.tenant_id,
            namespace,
            entities_payload,
            fuzzy=self._ingestion_config["fuzzy_entity_resolution"],
            trgm_threshold=self._ingestion_config["fuzzy_trgm_threshold"],
            embedding_threshold=self._ingestion_config["fuzzy_embedding_threshold"],
        )

        edges_data = [
            {
                "source_id": content_to_id[t["subject"]],
                "target_id": content_to_id[t["object"]],
                "relation": t["predicate"],
                "metadata": t.get("metadata"),
            }
            for t in triplets
            if t["subject"] != t["object"]
        ]
        await self._store.upsert_edges(self.tenant_id, namespace, edges_data)

        return {"edges": len(edges_data), "entities": len(entity_list)}

    async def explain_connection(
        self,
        source_entity: str,
        target_entity: str,
        namespace: str,
        max_hops: int = 3,
        directed: bool = False,
        relation_types: Optional[List[str]] = None,
        exclude_relation_types: Optional[List[str]] = None,
        min_weight: float = 0.0,
    ) -> Optional[ExplainedPath]:
        """Reconstructs the actual reasoning path connecting two named
        entities — "how is Alice connected to Payments" as an ordered
        sequence of hops (`Alice --works_in--> Payments`), not just "both
        showed up in a traversal with some score." Entity names are
        resolved to nodes via nearest-embedding match (same approach
        `get_entity` uses), not exact string match, so paraphrased or
        slightly-off names still resolve.

        Returns `None` if either entity can't be resolved, or if no path
        exists within `max_hops` — both are legitimate, explicit answers
        ("not found" / "not connected"), not the same as an error.
        """
        source_embedding = await self._extractor.get_embedding(source_entity)
        target_embedding = await self._extractor.get_embedding(target_entity)
        source_matches = await self._store.vector_search_nodes(self.tenant_id, namespace, source_embedding, top_k=1)
        target_matches = await self._store.vector_search_nodes(self.tenant_id, namespace, target_embedding, top_k=1)
        if not source_matches or not target_matches:
            return None

        result = await self._store.find_path(
            self.tenant_id,
            namespace,
            str(source_matches[0]["id"]),
            str(target_matches[0]["id"]),
            max_hops=max_hops,
            directed=directed,
            relation_types=relation_types,
            exclude_relation_types=exclude_relation_types,
            min_weight=min_weight,
        )
        if result is None:
            return None

        return ExplainedPath(
            steps=[PathStep(node=s["node"], relation=s["relation"]) for s in result["path"]],
            score=result["score"],
            hop_distance=result["hop_distance"],
        )

    async def find_paths(
        self,
        source_entities: List[str],
        target_entities: List[str],
        namespace: str,
        max_hops: int = 3,
        top_k: int = 5,
        directed: bool = False,
        relation_types: Optional[List[str]] = None,
        exclude_relation_types: Optional[List[str]] = None,
        min_weight: float = 0.0,
        max_neighbors_per_node: int = 20,
    ) -> List[Dict[str, Any]]:
        """Resolves named endpoints and returns ranked, cited paths."""
        source_ids: List[str] = []
        target_ids: List[str] = []
        for name, output in (
            (source_entities, source_ids),
            (target_entities, target_ids),
        ):
            for entity in name:
                embedding = await self._extractor.get_embedding(entity)
                matches = await self._store.vector_search_nodes(
                    self.tenant_id, namespace, embedding, top_k=1
                )
                if matches:
                    output.append(str(matches[0]["id"]))

        return await self._store.find_paths(
            self.tenant_id,
            namespace,
            source_ids,
            target_ids,
            max_hops=max_hops,
            top_k=top_k,
            directed=directed,
            relation_types=relation_types,
            exclude_relation_types=exclude_relation_types,
            min_weight=min_weight,
            max_neighbors_per_node=max_neighbors_per_node,
        )

    async def retrieve(self, question: str, namespace: str, **overrides: Any) -> TenantRetrievalResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not namespace or len(namespace) > MAX_NAMESPACE_LENGTH:
            raise ValueError(f"namespace must be 1-{MAX_NAMESPACE_LENGTH} characters")
        correlation_id = new_correlation_id()
        tid_str = str(self.tenant_id)
        async with self._event_bus.timed(obs.RETRIEVAL_COMPLETED, correlation_id, tid_str, namespace) as attrs:
            result = await self._retrieve_inner(question, namespace, correlation_id, **overrides)
            attrs["chunk_count"] = len(result.chunks)
            attrs["node_count"] = len(result.nodes)
            return result

    async def _retrieve_inner(
        self, question: str, namespace: str, correlation_id: str, **overrides: Any
    ) -> TenantRetrievalResult:
        tid_str = str(self.tenant_id)
        cfg: RetrievalConfig = {**self._retrieval_config, **overrides}
        mode = cfg.get("mode", "hybrid_graph")
        if mode not in {"vector", "hybrid", "hybrid_graph"}:
            raise ValueError("mode must be 'vector', 'hybrid', or 'hybrid_graph'")
        top_k = int(cfg.get("top_k", 5))
        hops = int(cfg.get("hops", 2))
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if not 0 <= hops <= 5:
            raise ValueError("hops must be between 0 and 5")
        trace = RetrievalTrace(mode=mode)
        started = time.perf_counter()
        query_embedding = await self._embed(question, correlation_id, namespace)
        trace.embedding_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        async with self._event_bus.timed(
            obs.RETRIEVAL_HYBRID_COMPLETED, correlation_id, tid_str, namespace
        ) as attrs:
            if mode == "vector":
                hybrid_hits = await self._store.semantic_search_chunks(
                    self.tenant_id, namespace, query_embedding,
                    top_chunks=top_k * 2,
                    metadata_filter=cfg.get("metadata_filter"),
                )
            else:
                hybrid_hits = await self._store.hybrid_search(
                    self.tenant_id,
                    namespace,
                    question,
                    query_embedding,
                    top_chunks=top_k * 2,
                    metadata_filter=cfg.get("metadata_filter"),
                )
            attrs["hit_count"] = len(hybrid_hits)
        trace.search_ms = (time.perf_counter() - started) * 1000
        trace.search_candidates = len(hybrid_hits)
        chunks = [
            TenantRetrievedChunk(
                id=str(h["id"]),
                document_id=str(h["document_id"]),
                source_id=h["source_id"],
                ordinal=h["ordinal"],
                content=h["content"],
                rrf_score=h["rrf_score"],
                lexical_rank=h["lex_rank"],
                semantic_rank=h["sem_rank"],
            )
            for h in hybrid_hits[: cfg.get("top_k", 5)]
        ]

        max_context_tokens = int(cfg.get("max_context_tokens", 4000))
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        budgeted_chunks: List[TenantRetrievedChunk] = []
        used_tokens = 0
        for chunk in chunks:
            item_tokens = _estimate_tokens(f"{_citation_marker(chunk)} {chunk.content}")
            if used_tokens + item_tokens > max_context_tokens:
                trace.context_truncated = True
                continue
            budgeted_chunks.append(chunk)
            used_tokens += item_tokens
        chunks = budgeted_chunks

        if mode != "hybrid_graph":
            trace.context_tokens = used_tokens
            return TenantRetrievalResult(chunks=chunks, trace=trace)

        graph_seed_chunks = int(cfg.get("graph_seed_chunks", 1))
        if not 1 <= graph_seed_chunks <= top_k:
            raise ValueError("graph_seed_chunks must be between 1 and top_k")
        selected_seed_chunks = _graph_seed_chunks(question, chunks, graph_seed_chunks)
        seed_scores = await self._store.mentioned_node_scores_for_chunks(
            self.tenant_id,
            {c.id: c.rrf_score for c in selected_seed_chunks},
        )
        seed_ids = list(seed_scores)
        trace.seed_scores = seed_scores

        nodes: List[TenantRetrievedNode] = []
        edges: List[TenantRetrievedEdge] = []
        if seed_ids:
            started = time.perf_counter()
            async with self._event_bus.timed(
                obs.RETRIEVAL_TRAVERSAL_COMPLETED, correlation_id, tid_str, namespace, seed_count=len(seed_ids)
            ) as attrs:
                graph = await self._store.traverse_graph(
                    self.tenant_id,
                    seed_ids,
                    namespace=namespace,
                    max_hops=hops,
                    seed_scores=seed_scores,
                    directed=cfg["directed"],
                    relation_types=cfg["relation_types"],
                    exclude_relation_types=cfg["exclude_relation_types"],
                    min_weight=cfg["min_weight"],
                    score_decay=cfg["score_decay"],
                    max_neighbors_per_node=cfg["max_neighbors_per_node"],
                    metadata_filter=cfg.get("metadata_filter"),
                )
                attrs["node_count"] = len(graph["nodes"])
                attrs["edge_count"] = len(graph["edges"])
            trace.traversal_ms = (time.perf_counter() - started) * 1000
            nodes = [
                TenantRetrievedNode(
                    id=str(n["id"]), content=n["content"], metadata=n.get("metadata") or {},
                    hop_distance=n["hop_distance"], score=n["score"] or 0.0,
                )
                for n in graph["nodes"]
            ]
            nodes.sort(key=lambda n: (-n.score, n.hop_distance, n.content, n.id))
            budgeted_nodes: List[TenantRetrievedNode] = []
            for node in nodes[: cfg["max_context_nodes"]]:
                item_tokens = _estimate_tokens(
                    f"{node.content} score={node.score:.4f} hops={node.hop_distance}"
                )
                if used_tokens + item_tokens > max_context_tokens:
                    trace.context_truncated = True
                    continue
                budgeted_nodes.append(node)
                used_tokens += item_tokens
            nodes = budgeted_nodes
            kept_ids = {n.id for n in nodes}
            node_scores = {n.id: n.score for n in nodes}
            edges = [
                TenantRetrievedEdge(
                    source_id=str(e["source_node_id"]), target_id=str(e["target_node_id"]),
                    source_content=e["source_content"], target_content=e["target_content"],
                    relation=e["relation"], weight=e["weight"],
                    score=(
                        (node_scores.get(str(e["source_node_id"]), 0.0)
                         + node_scores.get(str(e["target_node_id"]), 0.0))
                        / 2.0
                    ) * (1.0 - pow(2.718281828, -float(e["weight"]))),
                    id=str(e.get("id") or ""),
                )
                for e in graph["edges"]
                if str(e["source_node_id"]) in kept_ids and str(e["target_node_id"]) in kept_ids
            ]
            edges.sort(key=lambda e: (-e.score, -e.weight, e.relation, e.source_id, e.target_id))
            budgeted_edges: List[TenantRetrievedEdge] = []
            for edge in edges[: cfg["max_context_edges"]]:
                item_tokens = _estimate_tokens(
                    f"{edge.source_content} {edge.relation} {edge.target_content}"
                )
                if used_tokens + item_tokens > max_context_tokens:
                    trace.context_truncated = True
                    continue
                budgeted_edges.append(edge)
                used_tokens += item_tokens
            edges = budgeted_edges

            # Add the exact chunks that asserted traversed edges. This makes
            # graph-derived facts available to answer generation with the
            # same resolvable [source#ordinal] citations as direct retrieval.
            edge_ids = [edge.id for edge in edges if edge.id]
            if edge_ids:
                evidence_rows = await self._store.get_edges_evidence(
                    self.tenant_id, namespace, edge_ids
                )
                existing_chunk_ids = {chunk.id for chunk in chunks}
                for row in evidence_rows:
                    chunk_id = str(row["chunk_id"])
                    if chunk_id in existing_chunk_ids:
                        continue
                    evidence_chunk = TenantRetrievedChunk(
                        id=chunk_id,
                        document_id=str(row["document_id"]),
                        source_id=row["source_id"],
                        ordinal=row["ordinal"],
                        content=row["content"],
                        rrf_score=0.0,
                        lexical_rank=None,
                        semantic_rank=None,
                    )
                    item_tokens = _estimate_tokens(
                        f"{_citation_marker(evidence_chunk)} {evidence_chunk.content}"
                    )
                    if used_tokens + item_tokens > max_context_tokens:
                        trace.context_truncated = True
                        continue
                    chunks.append(evidence_chunk)
                    existing_chunk_ids.add(chunk_id)
                    used_tokens += item_tokens

        trace.context_tokens = used_tokens
        return TenantRetrievalResult(chunks=chunks, nodes=nodes, edges=edges, trace=trace)

    async def answer(
        self, question: str, namespace: str, max_answer_tokens: int = 1500, **overrides: Any
    ) -> AnswerResult:
        """Generate a citation-validated answer from retrieved evidence.

        Retrieval remains independently available through ``retrieve()``.
        Unknown or missing citation markers cause an explicit abstention
        instead of returning an apparently grounded answer.

        `max_answer_tokens` defaults higher than it looks like it needs to
        (1500, not ~100) because of a real, evidenced failure mode with
        reasoning-tier models: the completion-token budget covers *both*
        invisible reasoning tokens and the visible answer, and reasoning
        token consumption is variable, not fixed. Confirmed directly against
        the API: three back-to-back calls with identical evidence consumed
        27, 47, and 70+ reasoning tokens respectively. At a 500-token
        budget, one real end-to-end run burned the entire budget on
        reasoning and returned an empty completion — which then failed
        citation validation not because the model formatted citations
        wrong, but because it never got to emit any visible text at all.
        `finish_reason` was still "stop", not "length", so this isn't even
        detectable from that field alone.
        """
        started = time.perf_counter()
        retrieval = await self.retrieve(question, namespace, **overrides)
        if not retrieval.chunks:
            return AnswerResult(
                answer="Insufficient evidence to answer from the indexed sources.",
                citations=[], retrieval=retrieval, grounded=False, usage={},
                latency_ms=(time.perf_counter() - started) * 1000,
                abstain_reason="no_evidence_retrieved",
            )

        evidence = "\n".join(
            f"{_citation_marker(c)} {c.content}" for c in retrieval.chunks
        )
        # Normalized comparison: a question naming "checkout-service" must
        # match evidence prose describing "checkout service" (verified
        # against a real end-to-end run that abstained purely because of
        # this hyphen-vs-space mismatch, not because the evidence was
        # actually missing) — see _normalize_identifier.
        normalized_evidence = _normalize_identifier(evidence)
        missing_anchors = [
            anchor for anchor in _identifier_anchors(question)
            if _normalize_identifier(anchor) not in normalized_evidence
        ]
        if missing_anchors:
            latency_ms = (time.perf_counter() - started) * 1000
            await self._event_bus.emit(obs.Event(
                obs.ANSWER_COMPLETED, new_correlation_id(), str(self.tenant_id), namespace,
                duration_ms=latency_ms,
                attributes={"grounded": False, "citation_count": 0, "reason": "missing_query_anchor"},
            ))
            return AnswerResult(
                answer="Insufficient evidence to answer from the indexed sources.",
                citations=[], retrieval=retrieval, grounded=False, usage={},
                latency_ms=latency_ms,
                abstain_reason="missing_query_anchor",
            )
        prompt = (
            "Answer the question using only the evidence below. Every factual "
            "sentence must end with one or more citation markers exactly as shown. "
            "If the evidence is insufficient or contradictory, answer exactly: "
            "Insufficient evidence to answer from the indexed sources.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )
        generated = (await self._extractor.generate_text(prompt, max_tokens=max_answer_tokens)).strip()
        usage = dict(self._extractor.last_usage or {})
        retried_with_larger_budget = False
        if not generated:
            # A reasoning-tier model's invisible reasoning-token consumption
            # is not a deterministic function of the prompt alone: verified
            # directly against the API that the *same* question and
            # evidence sometimes returns a full cited answer at a given
            # budget and sometimes an entirely empty completion at that same
            # budget. One bounded retry at double the budget recovers most
            # of these without permanently doubling the cost of every call
            # (only the calls that actually needed it pay for it).
            retried_with_larger_budget = True
            generated = (await self._extractor.generate_text(prompt, max_tokens=max_answer_tokens * 2)).strip()
            usage = dict(self._extractor.last_usage or {})
        raw_response = generated  # kept only for length/diagnostics below, never persisted verbatim
        allowed = {_citation_marker(c): c for c in retrieval.chunks}
        markers = re.findall(r"\[[^\[\]\n]+#\d+\]", generated)
        invalid = [marker for marker in markers if marker not in allowed]
        abstained = generated == "Insufficient evidence to answer from the indexed sources."
        grounded = bool(markers) and not invalid and not abstained
        if grounded:
            abstain_reason = None
        elif abstained:
            abstain_reason = "model_abstained"
        elif not raw_response:
            # Distinct from a non-empty-but-badly-cited response: an empty
            # completion from a reasoning-tier model usually means the
            # completion-token budget was consumed entirely by invisible
            # reasoning tokens before any visible answer text — the fix is
            # raising max_answer_tokens, not a prompt/citation-format change.
            abstain_reason = "empty_model_response"
            generated = "Insufficient evidence to answer from the indexed sources."
        else:
            abstain_reason = "invalid_or_missing_citation"
            generated = "Insufficient evidence to answer from the indexed sources."

        citations = [
            Citation(
                source_id=allowed[m].source_id or allowed[m].document_id,
                chunk_id=allowed[m].id,
                ordinal=allowed[m].ordinal,
                excerpt=allowed[m].content[:240],
            )
            for m in dict.fromkeys(markers)
            if m in allowed and grounded
        ]
        latency_ms = (time.perf_counter() - started) * 1000
        attrs: Dict[str, Any] = {
            "grounded": grounded,
            "citation_count": len(citations),
            "model": self._extractor.config["extraction_model"],
            "retried_with_larger_budget": retried_with_larger_budget,
        }
        if not grounded:
            attrs["reason"] = abstain_reason
            attrs["raw_response_length"] = len(raw_response)  # length only, never the text itself
        if usage:
            attrs["tokens"] = usage.get("total_tokens", 0)
        await self._event_bus.emit(obs.Event(
            obs.ANSWER_COMPLETED, new_correlation_id(), str(self.tenant_id), namespace,
            duration_ms=latency_ms, attributes=attrs,
        ))
        return AnswerResult(
            answer=generated, citations=citations, retrieval=retrieval,
            grounded=grounded, usage=usage, latency_ms=latency_ms,
            abstain_reason=abstain_reason,
        )
