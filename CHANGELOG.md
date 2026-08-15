# Changelog

All notable changes to `postgres-graph-rag` are documented here.

## Unreleased — Release 2: verified grounding (PRs 1-5)

Closes the gap the citation-only validator couldn't: a citation naming a
real, retrieved chunk previously passed as "grounded" regardless of
whether that chunk's text actually supported the claim — a reversed
relationship, a swapped predicate, or a wrong number all cited real
evidence and still passed. `answer()` now supports three grounding modes.

### Added

- `postgres_graph_rag.grounding`: the verification contract —
  `AnswerClaim`, `ClaimVerification`, `VerifiedAnswerResult`,
  `GroundingStatus` (7 states, replacing an overloaded `grounded: bool`),
  a `runtime_checkable` `Verifier` protocol, `VerifierUnavailableError`.
- `postgres_graph_rag.verification`: the deterministic (non-model)
  layers — citation existence, identifier/relevance checks (reusing
  `tenant_engine.py`'s own normalization), server-side quote location,
  a negation-based conflict heuristic, policy evaluation, deterministic
  answer rendering, and `DeterministicVerifier` as an injectable offline
  implementation. Deliberately conservative: only ever positively confirms
  "supported" from a located quote, "contradicted" from an explicit
  numeric/date mismatch or an explicit negation in a competing retrieved
  chunk — everything requiring real semantic entailment is reported
  "insufficient" rather than guessed.
- `postgres_graph_rag.model_verifier.ModelEntailmentVerifier`: batched
  model entailment layered on top of the deterministic layers — only
  claims the deterministic layer genuinely can't decide reach the model,
  one bounded provider call per answer, server-side validation of every
  model-proposed quote (a claimed quote absent from the cited evidence is
  rejected, never trusted), confidence treated as uncalibrated metadata,
  any provider failure raising `VerifierUnavailableError` for the whole
  batch rather than a silent partial substitution.
- `TenantGraphRAG.answer(..., grounding_mode=...)`: `"citation_only"`
  (default, today's unchanged behavior), `"verified"` (drops claims the
  verifier can't confirm, keeps the rest), `"verified_strict"` (abstains
  the whole answer if any claim isn't fully supported). A verifier failure
  reports `grounding_status="verification_failed"` and abstains — never
  silently "verified". `AnswerResult` gains `grounding_mode`,
  `grounding_status`, `claims`, `verifications` as additive, defaulted
  fields; existing callers checking only `.grounded`/`.abstain_reason` are
  unaffected.
- `benchmarks/grounding/`: a frozen evaluation harness (140 cases, 14
  categories, `grounding-benchmark-v1-candidate`) built *before* the
  verifier, plus reviewer tooling (blinded packets, reconciliation) for
  the independent human review still required before the sealed split is
  a validated gate — see `benchmarks/grounding/LABELING.md`.

### Measured

- Citation-only baseline: 100% contradicted-claim escape, 81.8% overall
  unsupported-claim escape (`docs/results/grounding-benchmark-baseline.md`).
- Deterministic layers alone: both escape rates to 0%, at a documented,
  expected cost to supported-claim retention (100% → 66.7%, entirely the
  multi-source compound-claim category, which requires cross-citation
  entailment no non-model layer can do) —
  `docs/results/grounding-benchmark-deterministic-verifier.md`.
- `ModelEntailmentVerifier` is unit-tested (mocked extractor) but has no
  live-provider benchmark run yet — needs real API credentials this
  environment doesn't have.

## 0.1.0 — Project consolidation

This release consolidates the current project direction under Ayush Mishra's
ownership and presents the production-oriented PostgreSQL GraphRAG architecture.

### Added

- PostgreSQL RLS-secured tenant API with transaction-local tenant context.
- Evidence-aware documents, chunks, entity mentions, and edge mentions.
- Lease-based, provider-aware extraction caching and retryable ingestion.
- Hybrid lexical/vector retrieval with bounded multi-hop graph traversal.
- Explainable connection paths with document and chunk citations.
- Community detection and community summaries.
- Offline evaluation, benchmark tooling, MCP integration, and observability hooks.

### Improved

- Async connection pooling, bulk writes, idempotent ingestion, and metadata filters.
- Entity resolution using normalization, trigram candidates, and embedding confirmation.
- Documentation for architecture, security, operations, evaluation, and known limits.

### Fixed — ingestion and retrieval correctness (schema v5 → v7)

- **Permanent retry hole.** `content_hash` was previously committed (its own
  transaction) before chunking/embedding ran; a failure afterward left the
  hash pointing at content whose chunks never existed, so a retry with
  identical text saw no change and skipped forever. `content_hash` and
  `document_chunks` now publish atomically in one transaction, guarded by a
  per-document advisory lock (`lock_document_for_publication`) so concurrent
  publishers of the same `source_id` can't race each other into both
  reporting a fresh publish for identical content.
- **`add_record()` partial visibility.** Entity node resolution now happens
  inside the same transaction as document/chunk publication, instead of
  before it — a failure rolls the node back too.
- **Graph enrichment is now an observable, separate stage.** `documents`
  gains `extraction_status` (`pending`/`ready`/`partial`/`failed`,
  CHECK-constrained), `extraction_error`, `extraction_attempts`. Status
  writes are compare-and-set against the exact revision hash the work was
  done for (`expected_content_hash`), so a worker finishing extraction after
  a newer revision has already been published can't overwrite that
  revision's status — verified for both the direct-ingestion path and the
  `retry_failed_chunks()` sweep (which binds to the hash captured when
  candidates were fetched, not one re-read at write time).
- **Stale-extraction race.** A chunk whose extraction finishes after a
  concurrent update has already replaced it is detected
  (`filter_existing_chunk_ids`) before its facts are wired to the wrong
  revision.
- **Unsupported edges no longer traversable.** An evidence-backed edge is
  created before its first mention is recorded; a failure in between could
  leave a zero-support edge that `weight >= min_weight` (default `0.0`)
  would not filter out. `prune_unsupported_edges()` cleans these up, and
  read-time traversal (both the recursive expansion and the induced-edge
  query) independently enforces `support_count > 0 OR manual_weight > 0` so
  correctness never depends entirely on that cleanup having already run.
- **`merge_entities()` evidence loss.** Repointing an edge onto a merged
  node previously carried over only `weight`, never `manual_weight`,
  `support_count`, or `edge_mentions` — silently discarding a merged edge's
  accumulated evidence. Now repoints mentions and recomputes support
  explicitly.
- **Induced-edge query had no filters.** `traverse_graph()`'s edge-loading
  step (edges between already-reached nodes) applied none of the recursive
  expansion's own filters — no support check, `min_weight`, or relation
  allow/deny list — so a caller-excluded or zero-support edge between two
  otherwise-reachable nodes could still surface in returned context.

### Before publishing

- Transfer the GitHub repository to the project owner's account or organization.
- Run the full PostgreSQL-backed test and benchmark suite.
- Complete the security and provenance hardening roadmap.
