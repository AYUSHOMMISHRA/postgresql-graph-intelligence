# ADR 003: Remove the legacy single-tenant engine entirely

**Status:** Accepted (supersedes the removal-timeline clause of ADR-002)

The legacy single-tenant engine (`DatabaseManager` in `database.py`, and
`PostgresGraphRAG.setup()`/`.add_texts()`/`.query()`/`.query_structured()`
in `core.py`) is deleted rather than kept deprecated until `1.0.0`.

ADR-002 committed to "no breaking removal ... before a major release,"
reasoning that existing callers might depend on the legacy path. At the
point this decision was made, there were no customers and no external
installs using this library at all — the compatibility window ADR-002
protected was protecting against a case that did not exist. Waiting for a
version milestone to remove code nobody uses is unnecessary process
overhead, not caution.

The one genuinely shared capability the legacy schema enabled — backfilling
a pre-existing single-tenant deployment's data into the secure schema
(`migrate_legacy_data=True`, `LEGACY_TENANT_ID`) — does not depend on
`DatabaseManager` itself (it reads `public.graph_nodes`/`graph_edges` via
raw SQL) and is unaffected by this removal.

**Trade-off accepted:** `benchmarks/bench_scale.py` and the `live_provider`
end-to-end test suite were legacy-only and removed with no replacement yet
written. Scale-benchmarking and real-LLM e2e coverage of the secure path
are tracked as follow-up work, not silently dropped.

**When this reasoning would no longer apply:** once real external callers
exist, a similar removal would need the deprecation window ADR-002
originally specified.

## Addendum (0.2.0): residue this ADR's removal left behind

The initial removal above deleted the executable legacy engine but left a
few things describing it as if it still existed. A follow-up pass (tracked
in `LEGACY_DELETION_PLAN.md`) closed these, released as `0.2.0`:

- **`PostgresGraphRAG.__init__()`'s `postgres_url` parameter** had zero
  reads anywhere in `core.py` — a pure compatibility shim for the same
  "callers that don't exist" case this ADR already addressed, missed in the
  first pass. Removing it is more than cosmetic: because it was a required
  parameter, both `postgres-graph-rag-eval` and `postgres-graph-rag-mcp`
  (neither of which ever calls `setup_secure()`) refused to start without a
  privileged admin DSN they never used, pushing that credential into
  processes meant to hold only the restricted, `NOBYPASSRLS` runtime role.
  Removed atomically (parameter, both CLIs' flags, all call sites) in one
  commit, since no intermediate state without it is possible.
- **`grounding.VerifiedAnswerResult`**, a parallel result type with no
  producer (PR 3/4/5 wired the verifier into `tenant_engine.AnswerResult`
  instead), was deleted after promoting its validation invariants
  (`grounding_status` validation, a `grounded` property derived from it
  rather than separately stored) onto `AnswerResult` first.
- **`mcp_server.py`'s server lifespan duplicated `for_tenant()`'s lazy
  store construction**, and the two had diverged: `for_tenant()` validated
  a missing `runtime_url`; the lifespan did not, and would construct a
  store around `None`. Both now share one
  `PostgresGraphRAG._get_or_create_store()`.
- **Coverage restored, not just deleted-with-it:** `tests/test_database.py`
  (the deleted `DatabaseManager` tests) and the `live_provider`-marked
  `tests/test_scenarios.py` covered several still-live secure-path
  behaviors with no other test — `relation_types`/`exclude_relation_types`
  filtering, `max_neighbors_per_node`, `min_weight`, `score_decay`, the
  `max_hops` hard limit, namespace isolation within a tenant, and
  exact/fuzzy entity resolution (including the negative adversarial case).
  `tests/test_secure_path_characterization.py` restores this against the
  secure path directly, proven green against this ADR's pre-removal tree
  before the removal was committed.

**Still open, unchanged from the original trade-off above:** a
`SecureGraphStore`-based scale benchmark and real-provider (OpenAI/Gemini)
secured-path end-to-end tests. Tracked as follow-up work, gating the next
package release and any published performance claim respectively — not
silently dropped.
