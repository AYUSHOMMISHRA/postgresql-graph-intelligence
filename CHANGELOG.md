# Changelog

All notable changes to `postgres-graph-rag` are documented here.

## Unreleased — Remaining `0.2.0`-tracked follow-ups closed

Three of the four items `0.2.0` (below) recorded as deliberately deferred
are now done:

- **`database.py` renamed to `_db_utils.py`.** After `DatabaseManager`'s
  removal it was leaf utilities only; the name described a class that no
  longer existed. `tests/test_database.py` renamed to `tests/test_db_utils.py`
  to match. `tenancy.py`'s function-local import inside `migrate_schema()`
  (annotated "avoid cycle at module load") is removed and hoisted to the
  top-level import — the module imports nothing from this package, so no
  cycle was ever possible.
- **The citation-only grounding rule's two implementations
  (`tenant_engine.answer()` and `benchmarks/grounding/baseline_runner.py`)
  are now bound by a characterization test**
  (`tests/test_grounding_rule_parity.py`), rather than consolidated into one
  shared predicate — coupling the frozen benchmark baseline to what it
  measures would let it drift silently whenever production changes, which
  the plan rejected on purpose. `verification.evaluate_policy()`'s
  unreachable `citation_only` branch is annotated with why it's kept
  (exported public API) and why `answer()` never reaches it.
- **`benchmarks/bench_scale_secure.py`** replaces the deleted
  `bench_scale.py`: every timed query runs through
  `SecureGraphStore.tenant_connection()`, reflecting RLS
  policy-evaluation and transaction-local tenant-context overhead the
  legacy tool never measured. Smoke-tested at small scale; a full 1K/10K
  run against a properly isolated schema, with published numbers, remains
  separate follow-up work.

**Real-provider secured-path E2E tests** (`tests/test_live_provider_e2e.py`,
marked `live_provider`, excluded from routine runs): one real, billed
Gemini call through `setup_secure()` -> `for_tenant()` -> `add_document()`
-> `answer()` -- **run and passing**. The OpenAI counterpart is written and
gated on `OPENAI_API_KEY` (currently unset in this environment); it will
run automatically once a real key is available, no code change needed.

## 0.2.0 — Legacy-removal residue cleaned up; `AnswerResult` hardened

A follow-up pass (`LEGACY_DELETION_PLAN.md`) on the legacy single-tenant
engine's removal (below), closing residue the first pass missed and one
duplicate-construction defect it exposed. Version bumped `0.1.0` → `0.2.0`
(pre-1.0 minor bump for a breaking removal); see
`docs/decisions/003-remove-legacy-engine.md`'s addendum for full rationale.

### Removed

- `PostgresGraphRAG.__init__()`'s `postgres_url` parameter — had zero reads
  anywhere in `core.py`. Removed atomically (parameter, `postgres-graph-
  rag-eval`'s `--admin-url` flag, `postgres-graph-rag-mcp`'s
  `--postgres-url` flag, and all call sites in one commit — no working
  intermediate state exists otherwise). Because it was required, both CLIs
  previously refused to start without a privileged admin DSN neither ever
  used (neither calls `setup_secure()`), pushing that credential into
  processes meant to hold only the restricted runtime role.
- `grounding.VerifiedAnswerResult` — a parallel result type with no
  producer or consumer outside its own tests. Its validation invariants
  were promoted onto `tenant_engine.AnswerResult` first (see below), then
  the dead type was deleted.

### Fixed

- `tenant_engine.AnswerResult` now validates `grounding_status` (typed
  `Optional[GroundingStatus]`, validated in `__post_init__` via
  `grounding.validate_grounding_status()`) and derives `grounded` as a
  property from it, rather than storing `grounded` as a separate field
  that could disagree with `grounding_status`. `verifications` is now
  typed `List[ClaimVerification]` (was `List[Any]`).
- `mcp_server.py`'s server lifespan and `PostgresGraphRAG.for_tenant()`
  independently lazy-constructed `SecureGraphStore`, and had diverged: the
  lifespan skipped `for_tenant()`'s missing-`runtime_url` validation and
  would construct a store around `None` for a misconfigured deployment.
  Both now share one `PostgresGraphRAG._get_or_create_store()`.
- Stale CI/docs referencing the already-removed legacy engine and
  `benchmarks/bench_scale.py` as if they still existed (`ci.yml`,
  `docs/architecture.md`, `docs/operations.md`).

### Added

- `tests/test_secure_path_characterization.py` (22 tests): restores
  coverage `tests/test_database.py` and the `live_provider`-marked
  `tests/test_scenarios.py` used to provide for behavior that is still
  live in the secure path — `relation_types`/`exclude_relation_types`
  filtering, `max_neighbors_per_node`, `min_weight`, `score_decay`, the
  `max_hops` hard limit, namespace isolation within one tenant, exact/fuzzy
  entity resolution (including the negative adversarial case), invalid
  `embedding_dimension` rejection, and facade forwarding of retrieval
  config into `SecureGraphStore.traverse_graph()`. Proven green against the
  pre-legacy-removal tree before that removal was committed.
- `tests/test_tenant_engine.py::test_answer_result_*`: `AnswerResult`
  invariant tests absorbing the intent of the deleted
  `VerifiedAnswerResult` tests.
- `tests/test_tenancy.py`: tests confirming `for_tenant()` and the MCP
  lifespan share one store instance and raise an identical error without
  `runtime_url`, and that `postgres-graph-rag-eval` runs end-to-end with no
  admin DSN anywhere on its arguments.

### Still open (tracked, not silently dropped)

A `SecureGraphStore`-based scale benchmark (replacing the deleted
`bench_scale.py`) and real-provider (OpenAI/Gemini) secured-path
end-to-end tests. The former gates any published performance claim; the
latter gates the next package release.

## Unreleased — `docs/results/` removed

The `docs/results/` directory (5 dated benchmark/e2e-run markdown files:
`incident-benchmark-v1.md`, `grounding-benchmark-baseline.md`,
`grounding-benchmark-deterministic-verifier.md`,
`openai-four-question-e2e-2026-08-15.md`,
`google-complex-e2e-2026-08-15.md`) has been deleted, along with the
convention of writing benchmark output there.

### Why

These were point-in-time snapshots from the initial demo-prep/benchmarking
work. The CTO demo they were partly kept as backup material for has
already happened, so their operational role (backup material if a live
demo fails, evidence to cite live) is no longer live. The substantive
numbers they recorded (e.g. contradicted-claim escape rate 100% → 0% from
the deterministic grounding layers, the 12.5% → 100% multi-hop recall gap
between vector-only and hybrid+graph retrieval) remain accurately
described in this changelog's own historical entries below and are not
being re-litigated or lost — only the standalone result files themselves
are gone.

### Updated references

- `README.md`: removed the dead "Benchmark result" nav link and the
  grounding-modes section's reference to the deleted deterministic-verifier
  result file (now points to this changelog instead).
- `benchmarks/grounding/README.md`, `LABELING.md`, `gates.py`,
  `model_verifier_runner.py`: updated prose/help-text pointers that
  previously referenced `docs/results/...` paths to instead point at this
  changelog, or to a bare output filename for future runs (no more implied
  dedicated results directory).
- `docs/reviews/demo-readiness-review-2026-08-16.md`: left as an unedited,
  dated snapshot (per this repo's established convention of not rewriting
  historical documents), with a short addendum noting the files it cited as
  demo backup material no longer exist.

## Unreleased — Legacy single-tenant engine removed (not just deprecated)

The legacy single-tenant engine (`DatabaseManager` in `database.py`, and
`PostgresGraphRAG.setup()`/`.add_texts()`/`.query()`/`.query_structured()`
in `core.py`) has been deleted entirely, ahead of the `1.0.0` removal
target the prior deprecation entry below announced.

### Why now, not at `1.0.0`

The original deprecation timeline assumed there could be existing external
callers depending on the legacy API, and gave them a full major-version
notice window before removal. At this stage there are no customers and no
external installs depending on it, so that compatibility window was
protecting against a case that doesn't exist yet — removing it now avoids
carrying two parallel architectures for no one's benefit.

### Removed

- `DatabaseManager` class (`database.py`) and its connection-pool/query
  surface. The handful of shared leaf utility functions the secure path
  also uses (`normalize_entity`, `content_hash`, `_vector_column_type`,
  `_as_float_list`, `_cosine_similarity`, `MAX_HOPS_HARD_LIMIT`,
  `MAX_ROWS_PER_STATEMENT`) remain in `database.py`, untouched.
- `PostgresGraphRAG.setup()`, `.add_texts()`, `.query()`,
  `.query_structured()`, and the `RetrievedNode`/`RetrievedEdge`/
  `RetrievalResult` dataclasses those methods returned.
- `benchmarks/bench_scale.py` (it exclusively measured `DatabaseManager`
  and cannot function without it). A `SecureGraphStore`-based replacement
  is planned; see the README's Benchmarks section and Roadmap.
- `tests/test_core.py`, `tests/test_integration.py`,
  `tests/test_scenarios.py` in full (legacy-only); `tests/test_database.py`
  trimmed to just the two leaf-utility-function tests it still covers.
  `test_integration.py`/`test_scenarios.py` were the only real-LLM
  (`live_provider`-marked) end-to-end tests in the suite — that category of
  coverage is at zero for the secure path until a replacement is written;
  tracked as a known, accepted gap, not silently dropped.

### Kept, unaffected

- The legacy-data migration feature (`migrate_legacy_data=True` on
  `setup_secure()`, `LEGACY_TENANT_ID`) — it only ever read
  `public.graph_nodes`/`public.graph_edges` via raw SQL, never
  `DatabaseManager` itself, so it survives this removal untouched.
  `tests/test_tenancy.py::test_legacy_data_migration_preserves_ids` now
  seeds that legacy-shaped data with raw SQL instead of `DatabaseManager`,
  since the class it used to construct is gone; the migration behavior it
  verifies is unchanged.

### Documentation

- README's "Legacy API migration" section removed; the still-relevant
  "Migrating existing legacy data" content moved into "Multi-Tenancy &
  Security" under "Migrating data from a prior single-tenant deployment."
  The cross-reference to it from the prior deprecation entry below is now
  stale as a result — that entry is left otherwise unedited as an accurate
  historical record.
- See `docs/decisions/003-remove-legacy-engine.md` for the ADR recording
  this decision, which supersedes the "no breaking removal before a major
  release" policy stated in `docs/decisions/002-secure-api-primary.md`.

## Unreleased — PR 6 closure: reviewed-answer-key integrity validation

A third correction to `benchmarks/grounding/model_verifier_runner.py`:
`--reviewed-answer-key` parsed a reconciled-labels file without validating
it, so a partial, mismatched, or malformed key could silently corrupt
`human_verifier_agreement_rate` instead of failing loudly -- e.g. a key
covering 1 of the sealed split's 28 cases would report 100% agreement,
satisfying `--strict-release-gate` while 27 cases were never reviewed.

### Fixed

- `load_reviewed_answer_key()` now validates the file before use, raising
  `AnswerKeyError` (surfaced as a clean CLI exit code 2, not a traceback)
  if: the key's `split` isn't `"sealed"`; it doesn't list exactly two
  distinct reviewers; any case id is duplicated; any label isn't one of
  `supported`/`contradicted`/`insufficient`; or its case ids don't exactly
  equal the sealed split's full id set (missing or extra ids both
  rejected).
- `--strict-release-gate` now also requires `--split sealed` and the
  finalized `grounding-benchmark-v1` dataset version (rejecting the
  current `...-v1-candidate`) before running anything -- a release
  decision made against the wrong split or an unreviewed candidate
  dataset was never a valid one, even before an answer key was involved.

## Unreleased — PR 6 closure: strict release gate can actually pass

A second correction pass on `benchmarks/grounding/model_verifier_runner.py`,
closing the gap the previous pass left: `--strict-release-gate` was
structurally incapable of ever passing on a genuine run, because two of
its required gates were only ever populated by a separate, manually-run
command.

### Fixed

- **`verifier_failure_safely_represented_rate` is now measured on every
  run, live or simulated**, via a new `run_failure_safety_check()` that
  runs a dedicated `AlwaysFailingExtractor` exercise unconditionally --
  exactly parallel to how `run_fabrication_check()` already runs on every
  invocation. Previously this gate was `None` on any run that wasn't
  itself `--simulate-failure`, meaning a real release run against a real
  provider could never populate it, and `--strict-release-gate` would
  therefore always see it as N/A and always fail -- defeating the purpose
  of the strict mode existing at all.
- **`--strict-release-gate` now also fails when
  `verification_failure_count > 0`**, independent of the gate-verdict
  check. A run with real provider outages could previously pass strict
  mode as long as no gate's own subpopulation happened to include a
  failed case (e.g. an outage on a `supported`-labeled, correctly-cited
  case affects none of the gated denominators) -- an incomplete run is
  not a valid release signal regardless of which gates it touches.
- **`human_verifier_agreement_rate` is now computable** via a new
  `--reviewed-answer-key <path>` flag, pointing at a
  `reconcile_reviews.py`-produced reconciled-labels JSON file. Computes
  the fraction of resolved cases where the verifier's verdict matches the
  reconciled human label; stays `None`/N/A when the flag is omitted, same
  as before.

### Changed

- `.github/workflows/ci.yml`'s `unit` job now runs `uv run pytest -q`
  (bare) instead of an explicit test-file list, which had drifted behind
  several suites added for Release 2
  (`test_grounding_benchmark.py`, `test_grounding_contract.py`,
  `test_grounding_fixtures.py`, `test_grounding_verification.py`,
  `test_model_verifier.py`, `test_model_verifier_runner.py`,
  `test_extractor_generation.py` were never actually run in CI). Safe
  because `pyproject.toml`'s `addopts` already excludes `live_provider` by
  default and `test_database.py`/`test_tenancy.py` self-skip without
  `POSTGRES_URL` (only set in the separate `postgres` job).

## Unreleased — PR 6 metric correctness fixes, live-provider test isolation

A correction pass on `benchmarks/grounding/model_verifier_runner.py`
(PR 6), found by re-auditing its own metric semantics rather than trusting
the first-pass implementation:

### Fixed

- **A `verification_failed` outcome no longer pollutes escape/retention/
  rejection rates.** Previously these cases stayed in every denominator,
  counted as trivially "not escaped" (verdict is `None`, so `==
  "supported"` is always false) or, worse, as a correct rejection for
  `unknown_citation_rejection_rate` (`None != "supported"` is true). A
  provider outage could therefore dilute an escape rate to look better
  than the verifier's real behavior on the cases it actually completed,
  or masquerade as a successful rejection. Incomplete cases are now
  excluded from every such rate's denominator entirely and disclosed only
  via `verification_failure_count`.
  `test_outage_cannot_improve_escape_or_rejection_rates` and
  `test_outage_cannot_improve_unknown_citation_rejection_rate` lock this
  in.
- **`verifier_failure_safely_represented_rate` is now only ever computed
  from a `--simulate-failure` run.** It previously measured "fraction of
  attempted model calls that failed" across *any* run — meaning a healthy
  live run (near-zero real failures) would report a value near 0% and
  wrongly fail a gate requiring ≥100%. It's `None` outside
  `--simulate-failure`, regardless of how many real failures a live run
  happens to hit.
- **`fabricated_quote_rejection_rate` is now actually measured**, via a
  new `run_fabrication_check()` using the existing
  `fabrication_fixtures.json` pairs and a stub extractor that claims a
  fabricated quote supports a claim — entirely offline, no real model
  needed, since a fixture's `fabricated_quote` is guaranteed by
  construction to never be a literal substring of its evidence.
- **`--strict-release-gate`**: a new, stricter gate-check mode where an
  `N/A` verdict fails the run, not just an actual `FAIL` — the previous
  `--fail-on-gate` let required-but-unmeasured gates (human review,
  fabricated-quote rejection) pass CI silently. `--fail-on-gate` keeps its
  original, permissive behavior for `dev`/`calibration` iteration;
  `--strict-release-gate` is what an actual release go/no-go check should
  use.

### Changed

- `tests/test_integration.py`/`tests/test_scenarios.py` (7 tests making
  real, billed OpenAI/Gemini calls, gated only on API keys being present
  with no mocking) are marked `live_provider`. `pyproject.toml` now
  defaults `addopts` to exclude that marker, so a plain `pytest`/
  `uv run pytest` never triggers them even with real keys set in the
  environment — run them deliberately with `pytest -m live_provider`.
  (The existing CI workflow already listed test files explicitly and
  never included these two, so this closes a local/ad-hoc-run gap, not a
  CI one.)

## Unreleased — Legacy path deprecation and migration hardening

Prompted by an architecture audit: `PostgresGraphRAG.for_tenant()` already
delegated to the secure engine, and every real caller in this repo
(`demo.py`, `mcp_server.py`, `evaluation.py`, both CLI entry points) already
used it exclusively — but the raw legacy path (`add_texts()`/`query()`/
`query_structured()`, no RLS, no `answer()` at all) was still presented in
docs as a roughly equal option, and had received none of Release 1's
correctness fixes or Release 2's grounding work.

### Fixed

- **Migrated legacy edges were invisible to secure traversal.**
  `_migrate_legacy_data()` copied a legacy edge's `weight` but left
  `manual_weight` at its default `0.0` — since `support_count` is also `0`
  for migrated edges (the legacy schema never recorded evidence/mentions),
  every migrated edge failed the `support_count > 0 OR manual_weight > 0`
  read-time filter added for the unsupported-edge leak, and simply
  vanished from traversal despite `weight` being correctly populated.
  Fixed by copying `weight` into `manual_weight` too — the same
  representation `upsert_edges(evidence_backed=False)` already uses for
  deterministic, non-evidence-backed edges, which is exactly what a
  migrated legacy edge is. The existing migration test only checked node
  vector search; it now also creates an edge and confirms it survives
  migration and remains traversable.

### Deprecated

- `PostgresGraphRAG.setup()`, `.add_texts()`, `.query()`, and
  `.query_structured()` now emit `DeprecationWarning` (`stacklevel=2`,
  pointing at the caller). No functional change otherwise. Removal
  targeted for `1.0.0`, or after at least one published deprecation
  release; no correctness features will be backported to this path,
  security-critical fixes only. See README.md's new "Legacy API
  migration" section for the full gap list (atomic publication,
  `extraction_status`, evidence provenance, retry recovery, RLS, verified
  grounding) and the `migrate_legacy_data=True` migration path.

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
