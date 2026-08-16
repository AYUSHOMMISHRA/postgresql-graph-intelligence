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
