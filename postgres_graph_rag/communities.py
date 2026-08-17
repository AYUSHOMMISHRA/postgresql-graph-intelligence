"""SQL-native community detection and summarization (v0.4).

Algorithm: weighted label propagation, updated asynchronously (Gauss-Seidel
style) in a fixed, deterministic node order. Each node starts labeled with
its own id (a singleton community); on each pass, nodes are visited in
ascending id order and each adopts the label carried by the majority of its
neighbors' *edge strength* (`1 - exp(-weight)`, the same saturating
transform traversal scoring uses), breaking ties deterministically by the
smallest label UUID.

This was originally implemented as fully *synchronous* propagation (every
node updates at once from the previous round's snapshot, in one bulk SQL
statement per round) because that reads as the more natural "SQL-native"
shape. It doesn't work: synchronous LPA has a well-known oscillation
pathology on simple structures — two nodes joined by one edge each see only
the *other's* label as a candidate and swap forever, never converging. This
was caught empirically (a 2-node, 1-edge test case that should merge into
one community kept producing two singleton communities) before being
trusted, not assumed correct from the algorithm description.

The fix is the standard one: process nodes one at a time in a fixed
deterministic order within each pass, so a node's update can see a
neighbor's *already-updated-this-pass* label. This still converges
deterministically (the visiting order is fixed by node id, not random or
concurrency-dependent) and provably avoids the 2-cycle pathology, at the
cost of O(node_count) round trips per pass instead of O(1) — each node's
label decision is still one small, self-contained SQL statement, so the
per-node update logic is exactly as "SQL-native" as before; what changed is
the iteration granularity, not where the computation happens. For very
large namespaces this per-node round-trip cost is a real scaling limit
worth knowing about, not hidden — community detection is meant to run as an
explicit, externally-scheduled background job (see the module docstring in
tenant_engine.py / README), not on the request path.
"""

import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional


from . import observability as obs
from .observability import EventBus, NULL_EVENT_BUS, new_correlation_id
from .tenancy import SCHEMA, SecureGraphStore

logger = logging.getLogger("postgres_graph_rag")

MAX_ITERATIONS = 50


class CommunityEngine:
    def __init__(self, store: SecureGraphStore, event_bus: Optional[EventBus] = None):
        self._store = store
        self._event_bus = event_bus or NULL_EVENT_BUS

    async def refresh_communities(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        force: bool = False,
        max_iterations: int = MAX_ITERATIONS,
    ) -> Dict[str, Any]:
        """Recomputes communities for (tenant, namespace) via weighted label
        propagation, unless the namespace isn't marked dirty and `force` is
        False (recomputing only namespaces marked dirty since the last run,
        per the plan — but always recomputing the *complete* dirty
        namespace for correctness, not an incremental patch).

        Protected by a Postgres advisory lock scoped to (tenant, namespace)
        so two concurrent callers can't run competing label-propagation
        passes against the same graph.

        Returns {"skipped": True} if not dirty and not forced, otherwise a
        report: {"run_id", "converged", "iterations", "node_count",
        "community_count", "duration_ms"}.
        """
        async with self._store.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                # Advisory locks are keyed by two int4s; hashtext() collapses
                # the (tenant, namespace) string into one, and a fixed
                # second key namespaces this lock class from unrelated uses.
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(1728, hashtext(%s))",
                    (f"{tenant_id}:{namespace}",),
                )

                if not force:
                    await cur.execute(
                        f"SELECT 1 FROM {SCHEMA}.community_dirty WHERE tenant_id=%s AND namespace=%s",
                        (str(tenant_id), namespace),
                    )
                    if not await cur.fetchone():
                        return {"skipped": True}

                start = time.perf_counter()

                # Seed: every node labels itself (creates the run row first
                # so memberships can FK to it).
                await cur.execute(
                    f"""
                    INSERT INTO {SCHEMA}.community_runs
                        (tenant_id, namespace, converged, iterations, node_count, community_count, duration_ms)
                    VALUES (%s, %s, false, 0, 0, 0, 0)
                    RETURNING id
                    """,
                    (str(tenant_id), namespace),
                )
                run_id_row = await cur.fetchone()
                assert run_id_row is not None  # INSERT ... RETURNING always returns exactly one row
                run_id = run_id_row["id"]

                await cur.execute(
                    f"""
                    INSERT INTO {SCHEMA}.community_memberships (tenant_id, run_id, node_id, community_id)
                    SELECT %s, %s, id, id FROM {SCHEMA}.graph_nodes
                    WHERE tenant_id = %s AND namespace = %s
                    """,
                    (str(tenant_id), run_id, str(tenant_id), namespace),
                )
                await cur.execute(
                    f"SELECT count(*) AS c FROM {SCHEMA}.community_memberships WHERE tenant_id=%s AND run_id=%s",
                    (str(tenant_id), run_id),
                )
                node_count_row = await cur.fetchone()
                assert node_count_row is not None  # SELECT count(*) always returns exactly one row
                node_count = node_count_row["c"]

                converged = False
                iterations_run = 0
                if node_count > 0:
                    await cur.execute(
                        f"SELECT id FROM {SCHEMA}.graph_nodes WHERE tenant_id=%s AND namespace=%s ORDER BY id",
                        (str(tenant_id), namespace),
                    )
                    node_order = [r["id"] for r in await cur.fetchall()]

                    for i in range(max_iterations):
                        any_changed = False
                        for node_id in node_order:
                            await cur.execute(
                                f"""
                                WITH neighbor_labels AS (
                                    SELECT m.community_id AS label, SUM(1 - exp(-e.weight)) AS strength
                                    FROM {SCHEMA}.graph_edges e
                                    JOIN {SCHEMA}.community_memberships m
                                        ON m.tenant_id = e.tenant_id AND m.run_id = %(run_id)s
                                       AND m.node_id = (CASE WHEN e.source_node_id = %(node_id)s
                                                             THEN e.target_node_id ELSE e.source_node_id END)
                                    WHERE e.tenant_id = %(tenant_id)s
                                      AND (e.source_node_id = %(node_id)s OR e.target_node_id = %(node_id)s)
                                    GROUP BY m.community_id
                                ),
                                best AS (
                                    SELECT label FROM neighbor_labels ORDER BY strength DESC, label ASC LIMIT 1
                                )
                                UPDATE {SCHEMA}.community_memberships cs
                                SET community_id = best.label
                                FROM best
                                WHERE cs.tenant_id = %(tenant_id)s AND cs.run_id = %(run_id)s
                                  AND cs.node_id = %(node_id)s
                                  AND cs.community_id IS DISTINCT FROM best.label
                                RETURNING 1
                                """,
                                {"tenant_id": str(tenant_id), "run_id": run_id, "node_id": node_id},
                            )
                            if await cur.fetchone():
                                any_changed = True
                        iterations_run = i + 1
                        if not any_changed:
                            converged = True
                            break

                await cur.execute(
                    f"SELECT count(DISTINCT community_id) AS c FROM {SCHEMA}.community_memberships WHERE tenant_id=%s AND run_id=%s",
                    (str(tenant_id), run_id),
                )
                community_count_row = await cur.fetchone()
                assert community_count_row is not None  # SELECT count(*) always returns exactly one row
                community_count = community_count_row["c"]
                duration_ms = int((time.perf_counter() - start) * 1000)

                await cur.execute(
                    f"""
                    UPDATE {SCHEMA}.community_runs SET
                        converged=%s, iterations=%s, node_count=%s, community_count=%s, duration_ms=%s
                    WHERE tenant_id=%s AND id=%s
                    """,
                    (converged, iterations_run, node_count, community_count, duration_ms, str(tenant_id), run_id),
                )
                await cur.execute(
                    f"DELETE FROM {SCHEMA}.community_dirty WHERE tenant_id=%s AND namespace=%s",
                    (str(tenant_id), namespace),
                )

        await self._event_bus.emit(obs.Event(
            obs.COMMUNITY_REFRESH_COMPLETED, new_correlation_id(), str(tenant_id), namespace,
            duration_ms=duration_ms,
            attributes={"converged": converged, "iterations": iterations_run, "node_count": node_count, "community_count": community_count},
        ))
        return {
            "skipped": False,
            "run_id": str(run_id),
            "converged": converged,
            "iterations": iterations_run,
            "node_count": node_count,
            "community_count": community_count,
            "duration_ms": duration_ms,
        }

    async def latest_run(self, tenant_id: uuid.UUID, namespace: str) -> Optional[Dict[str, Any]]:
        async with self._store.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT id, converged, iterations, node_count, community_count, duration_ms, created_at
                    FROM {SCHEMA}.community_runs
                    WHERE tenant_id=%s AND namespace=%s
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (str(tenant_id), namespace),
                )
                return await cur.fetchone()

    async def list_communities(
        self, tenant_id: uuid.UUID, namespace: str, run_id: Optional[uuid.UUID] = None
    ) -> List[Dict[str, Any]]:
        """Communities from the given run (or the most recent run if
        omitted), each with its member entities."""
        async with self._store.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                if run_id is None:
                    await cur.execute(
                        f"SELECT id FROM {SCHEMA}.community_runs WHERE tenant_id=%s AND namespace=%s ORDER BY created_at DESC LIMIT 1",
                        (str(tenant_id), namespace),
                    )
                    row = await cur.fetchone()
                    if not row:
                        return []
                    run_id = row["id"]

                await cur.execute(
                    f"""
                    SELECT m.community_id, array_agg(n.content ORDER BY n.content) AS members, count(*) AS member_count
                    FROM {SCHEMA}.community_memberships m
                    JOIN {SCHEMA}.graph_nodes n ON n.tenant_id = m.tenant_id AND n.id = m.node_id
                    WHERE m.tenant_id = %s AND m.run_id = %s
                    GROUP BY m.community_id
                    ORDER BY member_count DESC
                    """,
                    (str(tenant_id), str(run_id)),
                )
                rows = await cur.fetchall()
        return [
            {
                "run_id": str(run_id),
                "community_id": str(r["community_id"]),
                "member_count": r["member_count"],
                "members": r["members"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    async def summarize_communities(
        self,
        tenant_id: uuid.UUID,
        namespace: str,
        extractor: Any,
        run_id: Optional[uuid.UUID] = None,
        max_chunks_per_community: int = 8,
        max_tokens: int = 300,
    ) -> List[Dict[str, Any]]:
        """Generates one summary per community via a single LLM call over
        its member entities and a sample of chunks that mention them
        (map-only; a full hierarchical map/reduce pass is only needed when
        evidence exceeds the model's context budget, which a single
        `max_chunks_per_community`-bounded sample avoids for the graph
        sizes this has been exercised at — treat that as a known scaling
        limit for very large communities, not a solved problem).

        Reuses a previous summary instead of calling the LLM again when a
        community's evidence (member set + sampled chunk content) hashes
        identically to a summary already stored for that `community_id`,
        regardless of which run produced it.
        """
        communities = await self.list_communities(tenant_id, namespace, run_id)
        if not communities:
            return []
        resolved_run_id = communities[0]["run_id"]

        results = []
        total_tokens_used = 0
        async with self._store.tenant_connection(tenant_id) as conn:
            for community in communities:
                community_id = community["community_id"]
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""
                        SELECT DISTINCT dc.content
                        FROM {SCHEMA}.community_memberships m
                        JOIN {SCHEMA}.entity_mentions em ON em.tenant_id = m.tenant_id AND em.node_id = m.node_id
                        JOIN {SCHEMA}.document_chunks dc ON dc.tenant_id = em.tenant_id AND dc.id = em.chunk_id
                        WHERE m.tenant_id = %s AND m.run_id = %s AND m.community_id = %s
                        LIMIT %s
                        """,
                        (str(tenant_id), resolved_run_id, community_id, max_chunks_per_community),
                    )
                    chunks = [r["content"] for r in await cur.fetchall()]

                evidence_hash = hashlib.sha256(
                    json.dumps({"members": community["members"], "chunks": sorted(chunks)}, sort_keys=True).encode()
                ).hexdigest()

                async with conn.cursor() as cur:
                    await cur.execute(
                        f"SELECT summary FROM {SCHEMA}.community_summaries WHERE tenant_id=%s AND community_id=%s AND evidence_hash=%s LIMIT 1",
                        (str(tenant_id), community_id, evidence_hash),
                    )
                    existing = await cur.fetchone()

                if existing:
                    summary = existing["summary"]
                else:
                    prompt = (
                        "Summarize the following cluster of related entities and supporting text "
                        "in 2-3 sentences. Focus on what connects them.\n\n"
                        f"Entities: {', '.join(community['members'])}\n\n"
                        f"Supporting text:\n" + "\n".join(f"- {c}" for c in chunks)
                    )
                    summary = await extractor.generate_text(prompt, max_tokens=max_tokens)
                    if extractor.last_usage:
                        total_tokens_used += extractor.last_usage["total_tokens"]

                async with conn.cursor() as cur:
                    await cur.execute(
                        f"""
                        INSERT INTO {SCHEMA}.community_summaries
                            (tenant_id, run_id, community_id, summary, evidence_hash, member_count, model)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (tenant_id, run_id, community_id) DO UPDATE SET
                            summary = EXCLUDED.summary, evidence_hash = EXCLUDED.evidence_hash
                        """,
                        (
                            str(tenant_id), resolved_run_id, community_id, summary, evidence_hash,
                            community["member_count"], extractor.config["extraction_model"],
                        ),
                    )
                # No commit here: the tenant GUC set by tenant_connection()
                # is transaction-local, so committing mid-loop would drop
                # tenant context for the remaining communities and the next
                # write would get rejected by RLS. tenant_connection()'s own
                # context manager commits once, after the whole loop.

                results.append({
                    "community_id": community_id,
                    "summary": summary,
                    "member_count": community["member_count"],
                    "reused": existing is not None,
                })

        summarize_attrs = {
            "community_count": len(results),
            "reused_count": sum(1 for r in results if r["reused"]),
        }
        if total_tokens_used:
            summarize_attrs["tokens"] = total_tokens_used
        await self._event_bus.emit(obs.Event(
            obs.COMMUNITY_SUMMARIZE_COMPLETED, new_correlation_id(), str(tenant_id), namespace,
            attributes=summarize_attrs,
        ))
        return results

    async def query_global(
        self, tenant_id: uuid.UUID, namespace: str, question: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Answers "what are the key themes/trends" style questions by
        ranking existing community summaries against the question via
        lexical overlap (no extra LLM call for the ranking step itself —
        summarize_communities() must have been run first). Returns ranked
        summaries as evidence for a caller to synthesize an answer from,
        rather than synthesizing one itself."""
        async with self._store.tenant_connection(tenant_id) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT cs.community_id, cs.summary, cs.member_count
                    FROM {SCHEMA}.community_summaries cs
                    JOIN {SCHEMA}.community_runs cr ON cr.tenant_id = cs.tenant_id AND cr.id = cs.run_id
                    WHERE cs.tenant_id = %s AND cr.namespace = %s
                    ORDER BY cr.created_at DESC
                    """,
                    (str(tenant_id), namespace),
                )
                rows = await cur.fetchall()

        question_words = set(question.lower().split())
        scored = []
        for r in rows:
            summary_words = set(r["summary"].lower().split())
            overlap = len(question_words & summary_words)
            scored.append((overlap, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"community_id": str(r["community_id"]), "summary": r["summary"], "member_count": r["member_count"], "relevance": score}
            for score, r in scored[:top_k]
        ]
