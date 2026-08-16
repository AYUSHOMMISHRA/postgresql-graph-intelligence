# Engineering Review — postgres-graph-rag
**Reviewer role:** Principal/Staff Engineer + Architecture Review
**Date:** 2026-08-16 | **Time remaining before CTO demo:** ~24 hours
**Repo state at review:** `main` @ `5245d64`, clean working tree, 13 commits total

> **Second independent re-verification pass completed 2026-08-16 (same day, later session).** This pass did not re-derive the whole report from scratch — it re-ran the system live (tests, demo, import graph, lint) against the *current* repo state to check whether every load-bearing claim in the report above still holds, with specific attention to the instruction not to trust filename-based architecture assumptions. Full results are in **§13** at the end of this file. Headline: almost everything re-confirmed exactly (285/285 tests, clean ruff, identical import graph including lazy in-function imports, `.env` correctly gitignored/never committed). **One real correction was found and is now the most important thing in this document**: the demo script's own "sell it" comparison (§7) does not show what it claims to show when actually run — see §13.2. Read §13 before presenting.

---

## 1. Executive Summary

This is, materially, one of the better-prepared "prototypes" I have reviewed for a demo-readiness pass. The codebase documents its own limitations more rigorously than most production systems do (README, CHANGELOG, and `docs/` read like an honest engineering log, not marketing copy). I ran the actual system rather than inferring behavior from source, and the primary demo path **works end to end, unmodified, right now**:

```
docker compose up -d postgres
uv run postgres-graph-rag-demo --admin-url "$POSTGRES_URL" setup --reset
uv run postgres-graph-rag-demo --runtime-url "$PGR_RUNTIME_URL" ingest
uv run postgres-graph-rag-demo --runtime-url "$PGR_RUNTIME_URL" evaluate --output evaluation.json
uv run postgres-graph-rag-demo --runtime-url "$PGR_RUNTIME_URL" query "..." --mode hybrid_graph
```

Result: **285/285 tests pass**, ruff is clean, and the demo evaluate step reports **100% recall on all three retrieval modes** on the bundled 4-question incident scenario, with `hybrid_graph` correctly answering the multi-hop question ("Which team owns the dependency of checkout-service?" → Identity Team, via checkout-service → auth-service → Identity Team) that pure vector/hybrid search structurally cannot answer from a single chunk.

**One real, confirmed environment trap** cost most of this review's setup time and will burn you tomorrow if you don't pre-empt it: a locally-built `.venv` can silently end up in a broken state (partial `.pth`-based editable install) where `uv run <console-script>` and `uv run pytest` fail with `ModuleNotFoundError: No module named 'postgres_graph_rag'`, while `uv run python -m pytest` and a **fresh** `uv sync` fix it immediately. This is not a code bug — `.venv/` is gitignored and the fix is a one-line `rm -rf .venv && uv sync --extra dev` — but if your demo machine's venv is in this state tomorrow morning and you don't know the fix, you will burn 20+ minutes debugging a "broken installation" that isn't. See P0-1.

Overall verdict: **YES WITH FIXES** — the fixes are almost entirely pre-flight hygiene (rebuild venv, fix one broken doc link, rehearse once), not code changes. Do not attempt architectural changes before the demo; there is no code-level P0.

**Correction pass (post-publication):** this report's first version classified `core.py`/`tenancy.py`/`tenant_engine.py`/`database.py` as "core architecture files" partly by filename/size. §4 was rewritten to derive architectural importance strictly from the real internal import graph, `pyproject.toml`'s actual entry points, and runtime execution — not naming. Most conclusions held up under that stricter test (`tenancy.py` and `tenant_engine.py` remain confirmed Architectural Core, now with in-degree evidence rather than just a size observation). Two things changed: **`extractor.py`** turned out to be the single most-depended-on module in the package (6 internal importers) and should have been called out as Architectural Core from the start — its name undersold its centrality. And **`database.py`** is not one thing — it bundles genuinely foundational shared utility functions (used by the real, RLS-secured path) with a separate `DatabaseManager` class that is legacy-only and unreachable from any of the three real entry points; the original report's blanket "core" label collapsed that distinction. Tracing `database.py`'s actual usage also surfaced a new finding (P2-4): the README's published benchmark numbers measure the legacy engine, not the RLS-secured one it recommends. See §4 for the full evidence trail.

---

## 2. Repository Inventory

Total: ~70 real files (excluding `.venv`, `.git`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `dist`, `.firecrawl`, `uv.lock`).

| Area | Files | Reviewed | Depth | Notes |
|---|---|---|---|---|
| `postgres_graph_rag/` (library, 17 modules, 8,520 LOC) | 17 | Yes | `core.py`, `demo.py`, `filters.py`, `mcp_server.py` (auth guard), `__init__.py` read line-by-line; `tenancy.py` (2,729 LOC), `tenant_engine.py` (1,456), `database.py` (960), `extractor.py`, `verification.py`, `model_verifier.py`, `grounding.py`, `communities.py`, `observability.py`, `offline.py`, `models.py` spot-checked (RLS DDL, security-critical branches, public API surface) rather than exhaustively read line-by-line | Deep on entry points + security-critical paths; targeted grep-verification on the rest | RLS policy SQL, MCP loopback guard, and metadata-filter parameterization independently verified against README's specific claims — all confirmed accurate |
| `tests/` (19 files, 6,273 LOC) | 19 | Yes (executed all; read `test_core.py`, `test_database.py` headers/structure) | Ran full suite; did not read all 6,273 lines by hand | 285 passed, 7 deselected (`live_provider`, correctly gated) |
| `benchmarks/` (bench_scale.py + grounding/ harness, 10 files) | 10 | Partial | Read `README.md`/`LABELING.md`, not every runner script line-by-line | Grounding benchmark harness is unusually mature (reviewer packets, reconciliation, strict release gates) — see CHANGELOG for its own bug history |
| `docs/` (architecture, security, operations, evaluation, 3 ADRs, 5 results docs) | 10 | Yes, full read | — | Docs match code; one broken link found (P1-1) |
| `.github/workflows/` (ci.yml, publish.yml) | 2 | Yes, full read | — | CI mirrors what I ran locally; matches |
| Root config (`pyproject.toml`, `docker-compose.yml`, `.env.example`, `.gitignore`, `LICENSE`, `CHANGELOG.md`, `README.md`) | 7 | Yes, full read | — | README is 725 lines and unusually precise; CHANGELOG documents real historical bugs and their fixes, which is a strong credibility signal, not a weakness |
| `.venv`, `.git`, caches, `dist/`, `.firecrawl/`, `uv.lock` | — | Not reviewed (generated/vendor/lock) | Role confirmed only | Correctly gitignored, no action needed |

**Inventory completeness:** every non-generated file's role is accounted for above. No file was skipped without a stated reason.

---

## 3. What the Project Actually Does

**Problem:** Standard RAG (vector similarity over chunks) cannot answer questions whose answer lives in the *relationship* between two facts spread across documents, not in one chunk's text. Building a real solution today typically means bolting a graph database (Neo4j) onto an existing Postgres+pgvector stack, creating a distributed-consistency problem between three data stores.

**Solution:** Put vectors, full-text search, and a property graph (nodes/edges) in the *same* PostgreSQL database. Ingestion chunks + embeds + LLM-extracts (subject, predicate, object) triplets; retrieval fuses vector + full-text (RRF) to find seed chunks/entities, then walks the graph via a bounded recursive CTE, scoring by hop-decayed relevance × edge weight.

**Primary user:** a team already running Postgres that wants multi-hop, explainable, cited retrieval without adding infrastructure — the README explicitly frames this as "Postgres Maximalism," not a general graph-database replacement.

**Fully implemented and verified working:** ingestion (atomic publish, retry-safe), hybrid + graph retrieval, RLS-enforced multi-tenancy, entity resolution (exact + fuzzy trigram + embedding), relationship weighting, community detection, global summarization, grounding/entailment verification (`citation_only`/`verified`/`verified_strict`), observability events, MCP server (stdio + HTTP with a real loopback-only auth guard), structured (non-LLM) ingestion via `add_record()`/`add_triplets()`, metadata filtering DSL (containment + typed comparisons).

**Explicitly NOT implemented, and honestly documented as such** (not discovered by me — the project says so itself): `split_entity()` (undoing an entity merge), OAuth token issuance for MCP HTTP transport, a real step-wise/reversible migration framework (current migration is one idempotent `CREATE ... IF NOT EXISTS` script with an advisory lock, not versioned up/down migrations), zero-mention node pruning, hierarchical map/reduce community summarization (currently truncates large communities rather than recursing).

**Legacy path:** `PostgresGraphRAG.setup()/add_texts()/query()/query_structured()` (no `for_tenant()`) is deprecated, emits `DeprecationWarning`, and is explicitly frozen — no correctness work has been backported to it since the secure/tenant path was built. Every real caller in the repo (demo, MCP server, evaluation CLI) already uses the secure path exclusively. This is a good sign: the deprecated path isn't secretly load-bearing.

**Does documentation match code?** Yes, to an unusual degree. I independently re-verified three specific README claims against source rather than trusting the prose:
1. "RLS is fail-closed via `FORCE ROW LEVEL SECURITY` + `NOSUPERUSER NOBYPASSRLS` role" — confirmed at `tenancy.py:145-148, 782-795`.
2. "Metadata filter fields are always bind parameters, never string-interpolated" — confirmed in `filters.py`; every `field`/`value` goes through `%(...)s` placeholders.
3. "MCP HTTP mode refuses to bind publicly without a tenant_resolver, only loopback + explicit opt-in" — confirmed at `mcp_server.py:334-347`, backed by an actual `_is_loopback()` IP check, not just an argparse default.

---

## 4. Architecture — Corrected via Dependency Tracing, Not File Naming

**Correction note (added after initial publication):** the first pass of this review classified `core.py`, `tenancy.py`, `tenant_engine.py`, and `database.py` as "core architecture files" partly on the strength of their names and sizes. That classification has been re-derived below from actual evidence — the real internal import graph (every `from .X import` in every module), `pyproject.toml`'s declared entry points, and runtime tracing through the demo run already executed in §7 — not from filenames. Two conclusions changed as a result (marked below); everything else held up under re-verification and is now labeled accordingly.

### 4.1 Method

1. **Real entry points**, from `pyproject.toml` `[project.scripts]` (the only things actually executed by a user/operator) and verified by literally running one of them (§7): `demo.py:main`, `mcp_server.py:main`, `evaluation.py:main`.
2. **Full internal import graph** (`from .X import` in every one of the 17 modules in `postgres_graph_rag/`), reduced to which modules import which:

```
__init__      -> core, grounding, model_verifier, offline, tenancy, tenant_engine, verification
core          -> database, extractor, models, tenancy, tenant_engine
demo          -> core, extractor, offline, tenancy
evaluation    -> core, extractor, offline
mcp_server    -> communities, core, tenancy
communities   -> observability, tenancy
tenancy       -> database, filters
tenant_engine -> extractor, grounding, model_verifier, models, observability, tenancy, verification
model_verifier-> extractor, grounding, tenant_engine, verification
verification  -> grounding, tenant_engine
extractor     -> models
offline       -> extractor, models
models        -> filters
database      -> (none — leaf module)
filters       -> (none — leaf module)
grounding     -> (none — leaf module)
observability -> (none — leaf module)
```

3. **In-degree = architectural centrality.** `extractor` (6 importers) and `tenancy` (6 importers) are the two most-depended-on modules in the package — this is *evidence* of centrality, not an assumption. `database` has only 2 importers (`core`, `tenancy`), which forced a closer look (§4.3) rather than accepting "database.py = the DB layer = core" at face value.
4. **Runtime confirmation:** the actual demo run in §7 exercised `demo.py → core.for_tenant() → tenancy.SecureGraphStore` + `tenant_engine.TenantGraphRAG`, producing the observed JSON ingest/query output — so this specific chain is confirmed by execution, not just static analysis. `mcp_server.py` was confirmed by source inspection (`rag.for_tenant(tenant_id)` called at lines 126/139/172/207/271/282) to route through the identical chain, but was **not itself executed** this session (no MCP client was run) — that part of the claim is **UNVERIFIED by runtime**, static-analysis-only.

### 4.2 Classification (evidence-based)

| File | Class | Evidence | Status |
|---|---|---|---|
| `tenancy.py` | **Architectural Core** | 6 internal importers (`__init__`, `core`, `communities`, `demo`, `mcp_server`, `tenant_engine`); owns RLS DDL (`ENABLE`/`FORCE ROW LEVEL SECURITY`, verified at lines 145-148); confirmed executed in the live demo run | CONFIRMED (previously asserted on size+name — now backed by in-degree + DDL grep + runtime) |
| `extractor.py` | **Architectural Core** | Highest in-degree (6 importers: `core`, `demo`, `evaluation`, `model_verifier`, `offline`, `tenant_engine`) — every execution path, live or offline, depends on it | **NEW** — not called out as core in the original pass; the naming ("extractor" sounds like a feature, not "core") is exactly the kind of miss this correction methodology exists to catch |
| `tenant_engine.py` | **Architectural Core** | Imported by `core`, `model_verifier`, `verification`, `__init__`; is the concrete class (`TenantGraphRAG`) instantiated and returned by `core.for_tenant()`, confirmed by reading `core.py:207-236` and by the live demo's `ingest`/`query` output shape matching `tenant_engine.py`'s report/result dataclasses | CONFIRMED |
| `core.py` | **Architectural Core (as a factory/facade), not as an engine** | Only 4 importers (`__init__`, `demo`, `evaluation`, `mcp_server`) but every one of the three real entry points goes through it — its `for_tenant()` method is the sole construction path for `SecureGraphStore`/`TenantGraphRAG`. Its *other* responsibility (`add_texts`/`query`/`query_structured`, the legacy `DatabaseManager`-backed methods) is confirmed **dead at the entry-point level**: none of `demo.py`, `mcp_server.py`, or `evaluation.py` call `add_texts`, `query`, `query_structured`, or legacy `.setup()` (`grep` returned zero matches across all three). | **REVISED** — originally described in one sentence as "legacy single-tenant engine... thin, deprecated, frozen," which undersold its *other*, load-bearing role as the factory every real entry point uses to reach the secure engine. Both things are true simultaneously; the original phrasing implied core.py itself was legacy, which is imprecise. |
| `database.py` | **Split — Supporting Infrastructure (utility functions) + Feature/Legacy (the `DatabaseManager` class)** | `tenancy.py` (the confirmed Architectural Core module) imports only six free functions/constants from it — `MAX_HOPS_HARD_LIMIT`, `MAX_ROWS_PER_STATEMENT`, `_as_float_list`, `_cosine_similarity`, `_vector_column_type`, `normalize_entity`, `content_hash`, `_HNSW_HALFVEC_MAX_DIM` — none of which reference `DatabaseManager`. The `DatabaseManager` **class** itself is instantiated only in `core.py` (the legacy `add_texts`/`query` path), `tests/test_database.py`, `tests/test_tenancy.py` (a legacy-vs-secure comparison test), and `benchmarks/bench_scale.py` — confirmed by `grep -rn "DatabaseManager("` across the whole repo (4 call sites, none in a real entry point). | **REVISED — this is the most significant correction.** The original report treated `database.py` as one undifferentiated "core" file. It is actually two things bundled in one module: a handful of genuinely foundational, shared pure functions (real Architectural Core, used by the RLS-secured path) and a full connection-pool/query class that is legacy-only and unreachable from any of the three real entry points. See §4.3 for why this matters. |
| `models.py` | **Configuration** | Imported by `core`, `extractor`, `offline`, `tenant_engine`; defines `ProviderConfig`/`RetrievalConfig`/`IngestionConfig` TypedDicts and their defaults — pure config schema, no logic | CONFIRMED (was already correctly treated as config, not re-flagged) |
| `filters.py` | **Utility** | Leaf module (imports nothing internal); imported by `models.py` and `tenancy.py`; single-purpose SQL-filter compiler, independently verified parameterized (§5, security check) | CONFIRMED |
| `grounding.py` | **Feature/Core Business Logic** | Leaf module; imported by `tenant_engine`, `model_verifier`, `verification`, `__init__`; defines the grounding-mode type contract (`GroundingMode`, `GroundingStatus`, `Verdict`) that the answer-verification feature is built on | CONFIRMED, and further tightened: runtime-verified (not just import-graph-verified) — `tests/test_grounding_verification.py` + `tests/test_tenant_engine.py` exercise it directly, both in the 285-passing run. **Not**, however, exercised by the actual demo CLI: `grep -n "grounding_mode" postgres_graph_rag/demo.py` returns nothing — `demo.py`'s `answer()` calls never set `grounding_mode`, so they run under the default `citation_only` mode only. If the CTO asks to see `verified`/`verified_strict` live, that needs a short ad hoc script calling `engine.answer(..., grounding_mode="verified")` directly — the demo CLI doesn't expose it as a flag. |
| `verification.py` | **Feature/Core Business Logic** | Imported by `tenant_engine`, `model_verifier`, `__init__`; the deterministic (non-model) verification layers | CONFIRMED, runtime-verified via `tests/test_grounding_verification.py`, `tests/test_model_verifier.py` (96 tests across the four grounding-related test files, all passing, re-run in isolation this session) — same demo-CLI caveat as `grounding.py` above |
| `model_verifier.py` | **Feature/Core Business Logic** | Imported by `tenant_engine`, `__init__`; the batched-model-call verification layer | CONFIRMED, runtime-verified via `tests/test_model_verifier.py`/`tests/test_model_verifier_runner.py` — same demo-CLI caveat |
| `observability.py` | **Supporting Infrastructure** | Leaf module; imported by `communities.py` and `tenant_engine.py`; typed event bus + sinks, no domain logic | CONFIRMED |
| `communities.py` | **Feature/Core Business Logic, but NOT on the demo path** | Imported only by `mcp_server.py` — **not** by `demo.py` or `evaluation.py`. Community detection/summarization is a real, tested feature, but it is reachable only via the MCP server or direct library use (`CommunityEngine(rag._secure_store)`), never via the CLI demo flow this review actually ran. | **NEW clarification** — the original report discussed communities.py as a "Current Release" feature (accurate) without noting it sits outside the exact path exercised in §7's demo run. Doesn't change demo readiness (nothing in the demo script calls it), but matters if the CTO asks to see it live — you would need a short separate script, not the existing `postgres-graph-rag-demo` CLI. |
| `offline.py` | **Supporting Infrastructure (test/demo-only)** | Imported by `__init__`, `demo.py`, `evaluation.py` — provides `OfflineExtractor`, the fixture-backed extractor the demo uses specifically so it needs no LLM key | CONFIRMED |
| `demo.py`, `evaluation.py`, `mcp_server.py` | **Entry Points** (own category, not "core") | Zero internal importers (nothing in the package imports them back) — confirmed leaf/entry nodes in the graph; declared in `pyproject.toml [project.scripts]`; `demo.py` additionally confirmed by direct execution in §7 | CONFIRMED |
| `benchmarks/bench_scale.py` | **Tests/Benchmark tooling**, but see §4.3 | Imports `DatabaseManager` directly (`from postgres_graph_rag.database import DatabaseManager`), **not** `SecureGraphStore`/`TenantGraphRAG` | **NEW finding**, see §4.3 |
| `tests/*.py` | **Tests** | — | unchanged from original pass |
| `docs/*`, `README.md`, `CHANGELOG.md` | **Documentation** | — | unchanged |
| `.github/workflows/*.yml`, `docker-compose.yml`, `.env.example`, `pyproject.toml` | **Configuration** | — | unchanged |
| `.venv/`, `.git/`, `dist/`, `.firecrawl/`, caches, `uv.lock` | **Generated/Vendor/Non-critical** | — | unchanged |

### 4.3 New finding surfaced by this correction (not in the original report)

**P2-4. `benchmarks/bench_scale.py` — the source of every latency/throughput number in the README's "Benchmarks" section — measures the legacy `DatabaseManager` path, not the RLS-secured `SecureGraphStore` path the same README calls "the recommended, actively-developed path" and "the primary API."**
- **Evidence:** `bench_scale.py` line 23: `from postgres_graph_rag.database import DatabaseManager`; line 155: `db = DatabaseManager(POSTGRES_URL)`. It never imports or instantiates `SecureGraphStore` or `TenantGraphRAG`. The RLS-secured path adds real per-transaction work the legacy path does not incur: a `set_config()` call to set the transaction-local tenant GUC, plus Postgres policy-predicate evaluation (`tenant_id = current_setting(...)`) on every row of every query against a table with `FORCE ROW LEVEL SECURITY`. None of that overhead is reflected in the published numbers (e.g., "10,000 nodes: `vector_search` p50/p95 6.6ms/9.5ms").
- **Why this was missed in the first pass:** the original review read `docs/architecture.md`'s "Known limits" section and the README's benchmark table at face value, and separately confirmed `tenancy.py`'s RLS DDL — but never traced *which engine the benchmark script itself instantiates*. That trace only happens by deliberately not trusting "this is the benchmark, therefore it benchmarks the current recommended path" and instead grepping the script's own imports.
- **Impact:** Low for tomorrow's demo (the demo doesn't cite these numbers live, and recall/latency numbers you *can* show — the `evaluate` step's output — are measured against the real secure path). **Moderate for CTO credibility if asked** "does the RLS overhead show up in your benchmarks?" — the honest answer is "no, the published `bench_scale.py` numbers predate/bypass RLS; we haven't published a secured-path scaling benchmark yet." Say this proactively if asked; do not imply the existing numbers include RLS overhead.
- **Fix (post-demo, not before):** either add a `SecureGraphStore`-based variant of `bench_scale.py`, or add one sentence to the README's Benchmarks section disclosing that the measured path is the legacy engine, not the RLS-secured one.
- **Action for tomorrow:** no code change needed before the demo; just don't cite the README's throughput/latency table as if it describes the secured production path, since it doesn't.

**Why the rest of the original architecture conclusions survive this correction:** the review's original evidence for `tenancy.py`/`tenant_engine.py` being central was never purely nominal — it included reading `core.py`'s `for_tenant()` method in full (§ original report, still accurate) and grepping RLS DDL directly. Re-deriving centrality from the import graph independently confirms the same two files, plus surfaces `extractor.py` as an equally central file the original pass didn't label as such. The one substantive error was collapsing `database.py`'s two distinct roles (shared utility leaf functions vs. the legacy `DatabaseManager` class) into a single "legacy" classification.

---

**Why Postgres-only (ADR-001):** avoids cross-store consistency work between a relational DB, a vector DB, and a graph DB, at the cost of weaker general graph analytics and a requirement to bound recursive traversal (both explicitly acknowledged, not hidden).

**Why the secure/tenant API is primary (ADR-002):** RLS-enforced tenancy is a real production property; the namespace-only legacy path is kept only for compatibility with pre-existing callers, with a stated no-breaking-removal-before-major-version policy.

This is appropriate design for a prototype heading toward production, not over-engineered and not under-engineered — the multi-tenancy, atomic-publication, and evidence-provenance work are real production concerns already handled, while things a prototype doesn't need yet (a full migration framework, node pruning, hierarchical summarization) are correctly deferred and labeled as deferred rather than faked.

---

## 5. Confirmed Findings (double-checked against source and/or runtime)

### P0 — Must address before the demo

**P0-1. Stale/corrupted local `.venv` breaks `uv run <console-script>` and `uv run pytest` (confirmed, fixed, reproducible).**
- **Evidence:** the review environment's original `.venv` contained dist-info for *both* Python 3.11 and 3.14 (`postgres_graph_rag-0.1.0.dist-info` under 3.14, `-0.1.2.dist-info` under 3.11) — i.e., a venv rebuilt in place across Python version bumps rather than recreated. In that state, `import postgres_graph_rag` succeeded via `uv run python -c "..."` (cwd happens to land on `sys.path` for `-c`/`-m` invocations), but the *installed console scripts* (`postgres-graph-rag-demo`, and `pytest` itself) failed with `ModuleNotFoundError: No module named 'postgres_graph_rag'` even though their shebang pointed at the correct interpreter. The editable-install `.pth` file was present but not being honored by that venv's site-initialization.
- **Root cause:** local venv corruption, not a repository defect — `.venv/` is correctly gitignored (verified: `git ls-files .venv` returns nothing).
- **Fix (verified):** `rm -rf .venv && uv sync --extra dev`. After rebuilding: `uv run postgres-graph-rag-demo --help` works, `uv run python -m pytest -q` → 285 passed, and I additionally confirmed the full `setup --reset` → `ingest` → `evaluate` → `query` demo sequence runs clean end-to-end from the rebuilt venv.
- **Action for you:** **before the demo**, on the actual demo machine, run `rm -rf .venv && uv sync --extra dev` once, then do a full dry run of the demo commands. Do not assume "it worked once" means the venv is healthy — verify on the literal machine/session you'll present from.
- **Status:** FIXED (venv rebuilt in this review session) / action item is procedural, not code.

### P1 — Should fix before the demo (all are quick, low-risk)

**P1-1. `README.md` links to `docs/demo.md`, which does not exist.**
- **Evidence:** line 24 of `README.md`: `[CTO demo](docs/demo.md)`; `ls docs/` shows `architecture.md, decisions/, evaluation.md, operations.md, results/, security.md` — no `demo.md`. `find . -iname demo.md` returns nothing repo-wide.
- **Impact:** low functional risk, real credibility risk — if the CTO clicks that link while you have the README open, it 404s. A "CTO demo" doc that doesn't exist looks like the demo section was written aspirationally and never finished, which undercuts the otherwise very strong honesty signal the rest of the docs send.
- **Fix (≤30 min):** either write a short `docs/demo.md` (setup → ingest → evaluate → query, the exact sequence in section 15 of this report) or change the README link to point at the "Production pilot demo" section already in `README.md` itself (`#production-pilot-demo`). The second option is faster and equally honest.
- **Verification:** click the link / grep for the file after the fix.

**P1-2. No rehearsed fallback if `docker compose up -d postgres` fails on the demo machine (Docker not running, port 5432 already bound, etc.).**
- **Evidence:** this is a live risk category (Docker daemon state is not controlled by the repo), not a code defect. `docker-compose.yml` itself is minimal and correct (health check present, standard `pgvector/pgvector:pg16` image).
- **Fix:** confirm Docker Desktop is running and port 5432 is free *before* the CTO sits down, not during the meeting. Have a `docker compose down && docker compose up -d postgres` + `docker compose logs postgres` sequence memorized as your one recovery step.

### P2 — Optional, only if time remains

**P2-1. Two Python versions' worth of stale package metadata were present in one `.venv`, suggesting the dev machine has rebuilt the venv in place across Python upgrades at least once.** Not a repo problem (gitignored), but worth a personal habit note: prefer `rm -rf .venv && uv sync` over letting `uv` "fix up" an existing venv when the system Python version changes.

**P2-2. `uv.lock` was stale relative to `pyproject.toml`'s declared version (found during second-pass verification).**
- **Evidence:** the repo's committed `uv.lock` encoded `postgres-graph-rag` at version `0.7.0` for the local editable package, while `pyproject.toml` (the source of truth) declares `version = "0.1.0"`. Rebuilding the venv (`uv sync --extra dev`, done to fix P0-1) regenerated `uv.lock` to correctly show `0.1.0`, matching `pyproject.toml`. This diff is currently sitting uncommitted in the working tree (`git diff uv.lock`) — I did not commit it, per instructions to only commit when asked.
- **Impact:** cosmetic/metadata only — no dependency versions changed, only the local package's own version string in the lock. Worth knowing about because `pip show postgres-graph-rag` or a build artifact could report `0.1.0` while `docs/decisions/002-secure-api-primary.md` references "kept for v0.7 compatibility," which reads as a mismatch if someone checks the installed version against the docs.
- **Fix:** decide the real intended version (0.1.0 vs 0.7.0 vs something else) and make `pyproject.toml` the single source of truth, then re-run `uv lock` once to sync. Not urgent — doesn't affect the demo.
- **Action for you:** either commit the regenerated `uv.lock` (if `0.1.0` is correct) or investigate why it had drifted to `0.7.0` before touching it further.

**P2-3. `.github/workflows/publish.yml` has a German-language inline comment** (`# CACHE DEAKTIVIERT UM FEHLER ZU VERMEIDEN` — "cache disabled to avoid errors") on the `enable-cache: false` line for the release-publish workflow. Harmless, but worth normalizing to English for consistency with the rest of the (all-English) codebase if you ever touch that file. Not worth doing before the demo.

**P2-4. `benchmarks/bench_scale.py` (the source of every number in the README's "Benchmarks" table) instantiates the legacy `DatabaseManager`, not the RLS-secured `SecureGraphStore` path — found only by tracing actual imports, not by trusting the benchmark's stated purpose.** Full evidence and impact in §4.3. Low risk for tomorrow (the demo doesn't cite these numbers), moderate risk if the CTO specifically asks whether RLS overhead is reflected in the published latency table — it is not. Say so if asked; fixing it (a secured-path benchmark variant, or a one-line disclosure in the README) is post-demo work.

### P3 — Post-demo / production evolution (self-documented by the project, not new findings)

The project's own roadmap and README already name these accurately; I am not adding new items here, only confirming they are real and correctly prioritized as post-demo:
- Versioned, reversible migration framework (currently one idempotent script + advisory lock).
- Zero-mention node pruning (orphaned entities are left in place).
- Hierarchical map/reduce community summarization (large communities are truncated, not recursively summarized).
- OAuth authorization server for MCP HTTP transport (deployer must supply their own `tenant_resolver`).
- `split_entity()` (undoing a merge) — explicitly documented as impossible with the current schema (no per-mention alias provenance retained after a merge), a real design gap rather than an oversight.
- Chunk idempotency has no document identity (`(namespace, chunk_hash)` only) — duplicate text across two documents in one namespace is under-tracked for "which documents mention X" purposes. Documented in the README as a known limitation with a stated fix path (a `documents`/`chunks`/`mentions` evidence schema — which, notably, has *already been partially built*: `documents`/`document_chunks`/`entity_mentions` tables exist in the secure schema; the residual gap is narrower than the README's wording alone suggests).
- Namespace scaling: all namespaces share one `graph_nodes`/`graph_edges` table pair; at larger multi-tenant scale the planner may stop using the HNSW index in favor of a namespace-filtered bitmap scan (measured and documented, with a re-run recommendation via `benchmarks/bench_scale.py`).

**I found no additional confirmed bugs, no dead code, no redundant implementations, and no architectural inconsistency worth flagging beyond what the project already documents about itself.** This is itself worth stating plainly to the CTO: the codebase's self-assessment is accurate, which is a stronger signal of engineering maturity than a clean bill of health from an external reviewer alone.

---

## 6. What NOT to Touch Before the Demo

- **`postgres_graph_rag/tenancy.py`, `tenant_engine.py`, `database.py`** — the largest, most security/correctness-critical files. All tests pass against them, RLS behavior is independently verified, and there is no confirmed defect. Any edit here in the next 24 hours is pure risk with no offsetting demo benefit.
- **The grounding/verification pipeline** (`grounding.py`, `verification.py`, `model_verifier.py`) — has its own documented bug-fix history in CHANGELOG (escape-rate dilution, strict-gate metric bugs) that has already been through multiple correction passes and is now covered by dedicated regression tests (`test_outage_cannot_improve_escape_or_rejection_rates`, etc.). Don't reopen it.
- **The legacy `core.py`/`DatabaseManager` path** — deprecated, frozen by policy, exercised only for backward-compatibility tests. Leave it exactly as-is.
- **CI workflows** — already correct and already mirror what I ran locally. No changes needed.
- **The demo script's hardcoded 4-document / 4-question scenario** (`demo.py`) — it is small, deterministic, and its answers were independently verified with zero LLM cost. Don't "enrich" it the night before; a bigger corpus is more surface area for something to go wrong live.

---

## 7. Safest Demo Flow (validated by me, this session)

**Setup (do this before the CTO arrives, not during):**
```bash
cd postgresql-graph-rag
rm -rf .venv && uv sync --extra dev        # avoid P0-1
docker compose up -d postgres
cp .env.example .env                        # if not already present
```

**Live sequence (exactly what I ran; all four steps succeeded):**
```bash
export POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/graph_rag
export PGR_RUNTIME_URL=postgresql://pgr_demo_runtime:pgr_demo_runtime_pw@localhost:5432/graph_rag

uv run postgres-graph-rag-demo --admin-url "$POSTGRES_URL" setup --reset
uv run postgres-graph-rag-demo --runtime-url "$PGR_RUNTIME_URL" ingest
uv run postgres-graph-rag-demo --runtime-url "$PGR_RUNTIME_URL" evaluate --output evaluation.json
uv run postgres-graph-rag-demo --runtime-url "$PGR_RUNTIME_URL" query \
  "Which team owns the dependency of checkout-service?" --mode hybrid_graph
```

**Expected results (what I actually got):**
- `setup`: "Secure schema is ready."
- `ingest`: 4 JSON reports, each `"skipped": false`, `"extraction_status": "ready"`.
- `evaluate`: recall **1.0** on all three modes (`vector`, `hybrid`, `hybrid_graph`); `hybrid_graph` p50 ≈13ms.
- `query --mode hybrid_graph`: correctly surfaces "Identity Team" via the checkout-service → auth-service → owned_by → Identity Team path — the multi-hop answer a plain vector search cannot produce, which is the entire point of the pitch.

**The one sentence that sells the demo:** ask the same question with `--mode vector` first, then `--mode hybrid_graph`, and show that only the graph-aware mode surfaces "Identity Team" — because that fact lives in a *different document* than the one mentioning checkout-service. This is a better live demonstration than the `evaluate` step's numbers alone.

**Failure recovery:** if any step errors, the single highest-probability cause is the `.venv` issue (P0-1) — the fix is `rm -rf .venv && uv sync --extra dev`, ~30 seconds. Second most likely: Docker not running — `docker compose up -d postgres`, wait for the healthcheck (`docker compose ps` should show `healthy`).

**Backup if live demo fails entirely:** `docs/results/incident-benchmark-v1.md` and `docs/results/openai-four-question-e2e-2026-08-15.md` / `google-complex-e2e-2026-08-15.md` are pre-recorded, real (non-fabricated per this review's spot checks of the surrounding code) run outputs you can screen-share instead.

---

## 8. One-Day Execution Plan

| Window | Action |
|---|---|
| Now → +1h | `rm -rf .venv && uv sync --extra dev` on the actual demo machine. Run the full demo sequence once, exactly as in §7. Fix P1-1 (broken `docs/demo.md` link) — 15–30 min. |
| +1h → +2h | Rehearse the demo narration once out loud, including the vector-vs-hybrid_graph comparison. Time it. |
| +2h → +4h | Read §9 (CTO Q&A) below until you can answer each without looking. Skim `docs/architecture.md`, `docs/security.md` once more for the same reason. |
| +4h → done | Nothing else. No refactors, no "quick improvements," no new demo features. The system already works; the only remaining risk is unfamiliarity with your own answers, not the code. |

---

## 9. CTO Explanation

**30-second version:** "Most RAG is blind to relationships between facts in different documents. We put vectors, full-text search, and a knowledge graph in the same PostgreSQL database instead of bolting on a separate graph database, so a question like 'which team owns what checkout-service depends on' gets answered correctly — vector search alone can't, because the answer spans two documents."

**2-minute version:** Problem (vector-only RAG can't do multi-hop reasoning) → use case (any team already on Postgres that needs explainable, cited, multi-hop retrieval without new infrastructure) → architecture (chunk+embed+extract triplets → Postgres stores vectors/FTS/graph together → hybrid retrieval fans out into a bounded recursive-CTE graph walk → answer with citations) → key decision (RLS-enforced multi-tenancy is the primary API; the earlier namespace-only version is deprecated) → status (285 tests passing, working live demo, measured benchmark showing hybrid+graph at 100% recall vs 12.5% for vector-only on multi-hop questions in the tracked benchmark run).

**5-minute technical version:** cover §4 (Architecture) verbatim, then the grounding/verification pipeline (`citation_only` vs `verified` vs `verified_strict` — the difference between "cited something real" and "the citation actually supports the claim," measured: contradicted-claim escape rate went from 100% to 0% after adding deterministic verification layers), then trade-offs (§10 below), then limitations (§5 P3 list), then what a month of further work buys (a real migration framework, node pruning, hierarchical summarization).

### Likely CTO questions

| Question | Answer | Evidence | Likely follow-up |
|---|---|---|---|
| Why Postgres instead of a real graph DB (Neo4j)? | One ACID store, no cross-database consistency problem; trade-off is weaker general graph analytics, accepted because this isn't a general graph-analytics product | ADR-001 | "What if you need arbitrary graph algorithms later?" → "Out of scope by design; this is a retrieval layer, not a graph platform." |
| Why RLS instead of `WHERE tenant_id = ...` in app code? | A forgotten filter in one query path is a cross-tenant leak; RLS makes missing tenant context fail closed (`NULL` never matches) at the database level | `tenancy.py:145-148`; tested explicitly (`test_connection_reuse_across_tenants_does_not_leak`, `test_runtime_role_cannot_bypass_rls_attributes`) | "What if someone connects as the admin role by mistake?" → "RLS is a no-op for a BYPASSRLS/superuser role, which is exactly why the runtime role is NOSUPERUSER NOBYPASSRLS and separate from the migration role — verified, not assumed." |
| What happens when the LLM call fails during extraction? | Retried with exponential backoff; a chunk that still fails is skipped and logged, not fatal to the batch, and is retryable later via `retry_failed_chunks()` reading stored chunk content — no need to keep the original text around | `core.py` `_extract_with_retry`; README "Structured Data Ingestion" section | "Is a partially-extracted document safe to query?" → "Yes — it's text-searchable immediately (`extraction_status: partial`), graph-incomplete until retried; this is the explicit design point of the atomic-publish/async-extraction split." |
| Is this idempotent? | Yes for both document publication (content-hash + chunks in one transaction, advisory-lock guarded against concurrent publishers) and structured records (`add_record`/`add_triplets`) | README "Structured Data Ingestion"; `tenancy.py` advisory lock usage | "What about the legacy `add_texts()` path?" → "Not equally hardened — it's deprecated and frozen precisely because it doesn't have this property; that's stated up front, not hidden." |
| What's the scaling bottleneck? | Multi-hop traversal latency is dominated by the recursive CTE re-joining edges/nodes per depth level, not the vector search step; at larger multi-tenant scale the planner may stop using the HNSW index because per-namespace rows are small relative to the whole shared table | `docs/architecture.md`, README benchmarks table, `docs/results/` | "Have you tested past 10K nodes?" → "Not yet — that's a stated, explicit gap (`benchmarks/bench_scale.py` only measured to 10K), and re-running it before scaling multi-tenant deployments is a stated recommendation, not an assumption of safety." |
| Where's the business logic — controllers or services? | There is no controller layer; `tenant_engine.py` is the orchestration layer, `tenancy.py` owns schema/queries, CLIs (`demo.py`/`evaluation.py`) are thin wrappers with no separate logic — verified by reading both CLI entry points end-to-end | §4 | — |
| What would you change with another month? | A real versioned migration framework (currently one idempotent script), orphan-node pruning, hierarchical community summarization for large communities, OAuth for the MCP HTTP transport | README roadmap section (self-reported, verified accurate) | "Why weren't these done already?" → "Deliberately deferred — they're real production concerns but not blocking for the current scale/use case, and pretending otherwise would be dishonest." |
| What's technically weak right now? | The chunk-idempotency scheme is keyed on `(namespace, chunk_hash)` without full document identity for duplicate text across two documents in the same namespace; entity-resolution fuzzy thresholds haven't been evaluated against a domain-specific corpus yet | README "Known limitation" callout; `docs/architecture.md` "Known limits" | "Does that affect tomorrow's demo?" → "No — the demo corpus has no duplicate text across documents." |

---

## 10. Engineering Decision Log

| Decision | Reason | Alternative considered | Trade-off accepted | Production evolution |
|---|---|---|---|---|
| Single Postgres instance for vectors+graph+relational | Avoid distributed consistency across 3 stores | Separate Neo4j/vector DB | Weaker general graph analytics; must bound recursive traversal | Re-benchmark at real tenant scale before assuming HNSW usage holds |
| RLS for tenancy, not app-level filtering | Fail-closed at the DB layer; one missed `WHERE` clause can't leak data | App-level `WHERE tenant_id=` everywhere | More upfront schema/role complexity (two roles, advisory locks, transaction-local GUC) | None needed — this is already the production-grade choice |
| Atomic publish, async graph extraction | LLM calls take seconds-minutes; holding a transaction open that long is worse than eventual consistency | Extract synchronously inside the ingest transaction | Documents are searchable before graph-ready (`extraction_status` tracks this) | None needed — already correct for production |
| Deprecate legacy namespace-only path instead of deleting it | Existing callers may depend on it; deletion is a breaking change | Delete immediately | Two code paths exist temporarily; legacy path frozen (no new correctness work) | Remove at 1.0.0 per stated policy |
| Community detection outside the request path | Weighted label propagation is not fast enough to run per-query | Real-time community computation | Requires an external scheduler/cron | None needed for prototype/production; already correctly scoped |

---

## 11. Final Scorecard

| Category | Score /10 | Reason |
|---|---:|---|
| Architecture | 9 | Clean layering, single-store decision well-justified and documented, no inconsistencies found |
| Code Quality | 8 | Clean, well-organized, well-commented for *why* not *what*; no dead code or duplication found in reviewed areas |
| Correctness | 9 | 285/285 tests pass; live demo run matches expected output exactly; RLS/filter security claims independently verified |
| Maintainability | 8 | Clear module boundaries; CHANGELOG shows a real, disciplined bug-fix-and-regression-test cycle |
| Testing | 9 | Full suite passes; `live_provider` tests correctly isolated; DB/RLS tests run against a real Postgres, not mocked |
| Reliability | 8 | Retry/backoff, idempotency, and lease-based extraction caching are real and tested; venv fragility is an environment issue, not code |
| Security | 8 | RLS fail-closed design verified in source, not just docs; parameterized SQL confirmed in filters.py; MCP loopback guard is real |
| Performance | 7 | Benchmarked only to 10K nodes; scaling limits are honestly disclosed, not hidden |
| Documentation | 9 | Exceptionally accurate and self-critical; one broken link (P1-1) |
| Prototype Quality | 9 | Does exactly what it claims, nothing faked |
| Demo Readiness | 8 | Fully working once the venv is rebuilt; no code changes needed |
| Production Readiness | 6 | Real gaps exist (migration framework, node pruning) but are honestly scoped, not surprises |

**Overall Prototype Score: 8.3/10**
**Demo Readiness: High**, conditional only on rebuilding `.venv` on the actual demo machine beforehand.
**CTO Confidence: High** — this will read as unusually mature engineering discipline for a "prototype," specifically because the limitations are pre-disclosed rather than discovered by the CTO in real time.

---

## 12. Final Verdict

### Would I approve this for a CTO demo?
**YES, WITH FIXES** — and the fixes are entirely pre-flight hygiene (rebuild venv, fix one dead link), not code changes. I found no code-level P0. This is close to the best-case outcome this kind of review can produce the day before a demo.

### If I had one day, what exactly would I fix?
1. Rebuild `.venv` on the actual demo machine and do a full dry run (P0-1) — do this first, it's the only thing that can actually break tomorrow.
2. Fix the `docs/demo.md` dead link in the README (P1-1) — 15 minutes.
3. Rehearse the demo narration once, specifically the vector-vs-hybrid_graph side-by-side comparison — it's more persuasive live than the `evaluate` JSON alone.

### What exactly would I remove?
Nothing. I found no dead code, no redundant implementations, and no code the project itself hasn't already correctly flagged for later removal (the legacy `core.py` path, on its own stated deprecation schedule — leave it, don't remove it early).

### What exactly would I leave untouched?
`tenancy.py`, `tenant_engine.py`, `database.py`, the grounding/verification pipeline, CI workflows, and the demo script's scenario data — see §6.

### What would I explicitly tell the CTO are the current limitations?
Chunk idempotency lacks full document identity for duplicate text across documents in one namespace; entity-resolution fuzzy thresholds are untested against a real domain corpus; scaling has only been measured to 10K nodes and the HNSW index may not be used at larger multi-tenant scale; the migration framework is one idempotent script, not versioned/reversible migrations; MCP HTTP transport requires the deployer to supply their own auth (`tenant_resolver`), it does not implement OAuth itself. All of these are already accurately stated in the project's own docs — the honest move is to repeat them, not to wait for the CTO to find them.

### What would I build next with another month?
A versioned/reversible migration framework, a real-provider extraction-quality canary (20–30 anonymized real documents, cost-capped, per `docs/evaluation.md`'s own stated next step), orphan-node pruning, hierarchical community summarization for large communities, and a domain-specific evaluation of the fuzzy entity-resolution thresholds before recommending them as defaults for a new customer's data.

---

## 13. Second Independent Re-Verification Pass (2026-08-16, later session)

Per instructions, this pass went back to the live repository and re-ran things rather than re-reading the report — with specific attention to the corrective instruction that architectural importance must be derived from actual dependency/runtime evidence, never from filenames. Everything below is freshly executed this session, not recalled from the first pass.

### 13.1 What was re-verified and confirmed unchanged

- **Test suite, live.** `uv run python -m pytest -q` → **285 passed, 7 deselected**, same as originally reported. `uv run ruff check .` → **all checks passed**.
- **Demo flow, live, end-to-end from the actual environment.** `docker compose up -d postgres` (container was already up and healthy) → `setup --reset` → `ingest` (4 documents, all `"skipped": false`, `"extraction_status": "ready"`) → `evaluate` → recall **1.0 on all three modes** (`vector`, `hybrid`, `hybrid_graph`), matching the original report exactly.
- **`.venv` health.** Currently healthy in this environment (`uv run python -c "import postgres_graph_rag"` succeeds, console scripts work). P0-1's failure mode is real but not currently present — cannot be re-triggered on demand since it depends on prior venv history; treat the fix (`rm -rf .venv && uv sync --extra dev`) as preventive, not something to "test" further.
- **Import graph, re-derived independently from source, including the filename-bias check the correction note asked for.** A naive `grep "^from \."` at file-top initially looked like it *contradicted* the report — it showed `core.py` importing only `database`, `extractor`, `models` (no `tenancy`/`tenant_engine`), and `mcp_server.py` importing only `communities`, `core` (no `tenancy`). Tracing this further (not stopping at the first grep, per the "verify twice" rule) found the missing edges are real: `core.py:217-218` and `mcp_server.py:109` both do **function-local, lazy imports** (`from .tenancy import SecureGraphStore, ...` / `from .tenant_engine import TenantGraphRAG` inside `for_tenant()`), which a top-of-file-only import scan misses. The original report's import graph in §4.1 already accounted for these lazy imports correctly. **Net result: the original report's architecture classification survives this second, independent, tools-down re-derivation — not because it wasn't checked, but because it was checked and held.** Nothing in §4 needed correction.
- **`.env` handling.** A real, populated `.env` exists locally with what appears to be a live `GOOGLE_API_KEY` and a `POSTGRES_URL` pointing at a `graph_rag_live` database (distinct from the demo's `graph_rag` database). Confirmed via `git check-ignore -v .env` and `git ls-files .env` (empty) that this file is correctly gitignored and has **never** been committed to git history. Not a repo defect. Operational note, not a code finding: don't screen-share a terminal with `cat .env` or an unredacted `env | grep API` during the demo.
- **`uv.lock` version drift (original P2-2).** Re-checked: the *working tree's* `uv.lock` shows `postgres-graph-rag` at `0.1.0`, matching `pyproject.toml` exactly — the drift to `0.7.0` noted in the original P2-2 is no longer present locally. **Correction (from a further recheck, §14): this check was incomplete — it only looked at the working tree, not the committed state.** `git show HEAD:uv.lock` still reads `0.7.0`; the fix was never committed. Status is **PARTIALLY RESOLVED**, not resolved — see §13.4 and §14 for the full correction and the reinstated action item (`demo-readiness-implementation-plan-2026-08-16.md` TASK-06a).

### 13.2 New finding — the demo's own "sell it" comparison does not show what §7 says it shows

**This is the one correction from this pass, and it matters more than anything else in this addendum.**

§7 instructs: *"ask the same question with `--mode vector` first, then `--mode hybrid_graph`, and show that only the graph-aware mode surfaces 'Identity Team' — because that fact lives in a different document than the one mentioning checkout-service."*

I ran exactly that, live, this session:

```
uv run postgres-graph-rag-demo --runtime-url "$PGR_RUNTIME_URL" query \
  "Which team owns the dependency of checkout-service?" --mode vector
uv run postgres-graph-rag-demo --runtime-url "$PGR_RUNTIME_URL" query \
  "Which team owns the dependency of checkout-service?" --mode hybrid_graph
```

**Result: both modes print the identical four "Relevant Passages,"** including `[ownership-auth#0] auth-service is owned_by Identity Team...` — in `--mode vector` as much as in `--mode hybrid_graph`. "Identity Team" is visibly present in the plain vector-mode output. The only difference between the two outputs is that `hybrid_graph` additionally appends a **"Related Entities"** and **"Relationships"** section (the explicit `checkout-service --[depends_on]--> auth-service --[owned_by]--> Identity Team` path) — the passages themselves do not differ.

**Root cause (traced, not guessed):** `demo.py::_query` calls `engine.retrieve(question, namespace, mode=..., top_k=5, ...)` (`demo.py:131`). The demo's own scenario is intentionally tiny — 4 documents total (`demo.py`'s `DEMO_DOCUMENTS`) — so with `top_k=5`, every mode retrieves essentially the entire corpus regardless of ranking quality. There is no distractor volume for vector-only ranking to fail against. The passage containing "Identity Team" gets retrieved under `vector` mode not because vector search understood the multi-hop relationship, but simply because the corpus is small enough that nothing gets excluded.

**This does not falsify the project's core claim** — it just means the *live demo CLI* isn't where that claim is actually demonstrated. The real, measured evidence for "vector-only structurally fails multi-hop, graph-aware retrieval doesn't" is `docs/results/incident-benchmark-v1.md` (verified present and readable this session): a 60-document, 120-question benchmark showing **Vector/Hybrid multi-hop recall@5 = 12.5%** vs **Hybrid+graph = 100%** — a real, large-enough corpus where the effect is genuine, not an artifact of corpus size. That benchmark result is legitimate and independently reviewable; it is simply a different artifact than the one §7 tells you to run live.

- **Impact:** **High if unaddressed** — if the CTO asks to see the vector-vs-graph comparison live (which §7 explicitly primes you to offer), running it exactly as scripted will show "Identity Team" in the vector-mode output too, undercutting the pitch in the room, live, in front of the audience it matters most for.
- **Impact if addressed:** Low — this is a five-minute narration fix, not a code fix.
- **Fix (do this, ~10–15 min, before rehearsing):** Don't claim vector mode "can't surface" the passage. Instead, narrate the *actual, honest, still-impressive* differentiator: "notice both modes retrieve the same raw text here — our demo corpus is deliberately small so it's cheap to run live with no LLM key. What `hybrid_graph` adds is the explicit, structured reasoning path at the bottom — `checkout-service → depends_on → auth-service → owned_by → Identity Team` — which is what an application would actually use to answer confidently and cite the chain, rather than making a human re-read four paragraphs to piece it together themselves. At real scale, where documents number in the thousands rather than four, that structural difference is what keeps recall high instead of collapsing — which is exactly what our 60-document benchmark measured: 12.5% recall for vector/hybrid on multi-hop questions vs 100% for hybrid+graph (`docs/results/incident-benchmark-v1.md`)." This is a stronger, more defensible claim than the original because it's precisely what was just verified live, not an inference.
- **Alternative/complementary fix, also low-risk:** if there's spare rehearsal time, screen-share `docs/results/incident-benchmark-v1.md` directly right after the live query comparison, as the "here's the same effect at real scale" follow-up. The doc exists, is readable, and its numbers were spot-checked (structure and gate-pass logic) this session.
- **Do NOT** attempt to fix this by enlarging the demo corpus tonight to manufacture a live distractor effect — that's exactly the kind of pre-demo scope change §6/§16 warns against, introduces new surface area hours before presenting, and isn't necessary: the honest narration above is already a good demo.
- **Verification method:** re-run the two commands above yourself once, read both outputs side by side, and rehearse the corrected narration out loud at least once before tomorrow.

### 13.3 Re-verified: nothing else changed classification

Re-checked the specific instruction not to assume `core.py`/`tenancy.py`/`tenant_engine.py`/`database.py` are "core" from naming alone: the original report's §4 already performed and documented this exact re-derivation (import graph, entry-point tracing, runtime confirmation) in its first "Correction pass." This session's independent re-run of the import-graph trace (§13.1 above) reproduces the same conclusions through a different method (fresh grep + manual lazy-import trace, done without reading §4 first) and finds no error to correct. `extractor.py`'s centrality and `database.py`'s two-role split both re-confirm.

### 13.4 Updated status table

| Item | Original status | Status after this pass |
|---|---|---|
| P0-1 (venv corruption) | FIXED (this session's venv) | Not reproducible in current env; procedure remains correct and necessary as a pre-flight check |
| P1-1 (dead `docs/demo.md` link) | OPEN | **Still OPEN** — not fixed by this pass (not a code/architecture question, easy 15-min fix, do it) |
| P1-2 (no rehearsed Docker fallback) | OPEN (procedural) | Unchanged |
| P2-2 (`uv.lock` version drift) | OPEN | **PARTIALLY RESOLVED, re-checked and corrected on a further recheck (2026-08-16, third pass) — the working tree's `uv.lock` reads `0.1.0` (fixed by a prior `uv sync`), but `git show HEAD:uv.lock` still reads `0.7.0`. The fix was never committed. Re-running `uv sync --extra dev` (as TASK-01 in `demo-readiness-implementation-plan-2026-08-16.md` does) will keep silently regenerating this same working-tree diff every time without resolving it at the repository level. Earlier text in this section said "RESOLVED" — that was wrong; it checked only the working tree, not `git HEAD`. Not a demo blocker (doesn't affect `uv sync`/`uv run` behavior either way), but decide once whether `0.1.0` is the intended version and commit the regenerated `uv.lock`, rather than leaving it perpetually uncommitted.** |
| P2-4 (`bench_scale.py` measures legacy engine) | OPEN, low priority | Unchanged, re-confirmed still true (not re-traced this pass, no new evidence needed) |
| **NEW: §13.2 — demo narration overclaims what live vector-vs-graph comparison shows** | n/a | **OPEN — highest-priority fix from this entire re-verification, do before rehearsal** |

### 13.5 Revised one-day priority order

1. **Rehearse the corrected narration from §13.2** — this is now the single most important action, above even the venv rebuild, because getting it wrong live is a credibility risk the original script actively sets up.
2. Rebuild `.venv` on the actual demo machine per P0-1 and do one full dry run.
3. Fix the dead `docs/demo.md` link (P1-1).
4. Everything else in §8 stands unchanged.

---

## 14. Third-Pass Recheck (2026-08-16, prompted by "recheck again — does anything in this file need updating?")

Triggered by the creation of `demo-readiness-implementation-plan-2026-08-16.md` from this report and a follow-up request to recheck both documents. Rather than re-reading this report for internal consistency only, re-ran the underlying checks live one more time.

**Result: one real correction, everything else re-confirmed.**

- **285/285 tests still pass, ruff still clean, `.venv` still healthy** — re-run again this session, no change from §13.1.
- **All file/line references re-verified against the current repo:** `README.md:24`'s dead link, `.github/workflows/publish.yml:21`'s German comment, `benchmarks/bench_scale.py:23,155`'s `DatabaseManager` import/instantiation, and `demo.py:126-134`'s `_query` implementation are all still exactly as described. No drift since §13.
- **Correction: §13.4's "P2-2 RESOLVED" was wrong, and the error is worth naming plainly.** §13.1 checked `uv.lock`'s *working-tree* content against `pyproject.toml` and found both read `0.1.0`, concluding the drift was resolved. Rechecking this pass against `git show HEAD:uv.lock` (the actual committed, shared state — what a fresh clone or CI checkout would see) shows **`0.7.0`, unchanged from the original P2-2 finding.** The `0.1.0` fix exists only in this machine's uncommitted working tree (from an earlier `uv sync --extra dev` run in this session) and was never committed. The corrected status is now recorded directly in §13.4's table above: **PARTIALLY RESOLVED**, not RESOLVED.
- **Why this matters, concretely:** `demo-readiness-implementation-plan-2026-08-16.md` TASK-01 has the presenter run `rm -rf .venv && uv sync --extra dev` on the demo machine tomorrow. That command will regenerate `uv.lock` back to `0.1.0` in the working tree again — reproducing the exact same uncommitted diff, every time, on every machine that runs it — without ever fixing the repository's actual committed state. This is harmless for the demo itself (doesn't change `uv run` behavior either way) but means the "fix" doesn't stick unless someone deliberately commits `uv.lock` once. Low priority, but worth doing consciously rather than leaving as a diff that silently reappears indefinitely.
- **No other correction found.** Every other claim in §13 (import graph via lazy imports, `.env` git-ignore status, demo narration finding in §13.2) was spot-checked again this pass and held exactly.

**Action, if anyone wants to actually close P2-2 for good (optional, non-demo-blocking):** confirm `0.1.0` is the intended package version (it matches `pyproject.toml`, so yes), then `git add uv.lock && git commit`. Not required before tomorrow's demo.

---

## 15. Addendum: Legacy Engine Subsequently Removed (post-demo)

This section's earlier guidance (§5 P3 list, §6's "leave it exactly as-is," §12's "leave it, don't remove it early") was correct **for the stated context at the time**: a same-day CTO demo, where the legacy `core.py`/`DatabaseManager` path was untouched, tested, and posed zero risk to leave alone. That reasoning does not need correcting.

**What changed, after the demo:** the original assumption behind keeping the legacy path deprecated-rather-than-deleted — protecting existing external callers via a `1.0.0` removal window — turned out not to apply. This is pre-launch work with no customers and no external installs. Once that was clarified, the decision was made to delete the legacy engine (`DatabaseManager`, and `PostgresGraphRAG.setup()`/`.add_texts()`/`.query()`/`.query_structured()`) outright rather than wait for a version milestone protecting no one.

This was executed as its own planned, verified piece of work — see `demo-readiness-implementation-plan-2026-08-16.md`'s corresponding addendum and `docs/decisions/003-remove-legacy-engine.md` for the full rationale and evidence trail. Not re-litigating this report's original findings: they were accurate; the removal is a separate, later decision built on new information (zero customers), not a correction of anything found here.

---

## 16. Addendum: `docs/results/` Removed (files this report cited no longer exist)

This report (§7, §13.2) cited three files under `docs/results/` — `incident-benchmark-v1.md`, `openai-four-question-e2e-2026-08-15.md`, `google-complex-e2e-2026-08-15.md` — as recommended backup material if the live demo failed, and cited `incident-benchmark-v1.md`'s specific 12.5%→100% multi-hop recall numbers as the evidence backing the "vector fails multi-hop, hybrid+graph doesn't" claim in the corrected demo narration (§13.2).

**These files, and the `docs/results/` directory itself, have since been deleted.** The demo this report was prepared for has already happened, so the backup-material role is no longer live. The underlying numbers themselves are not lost — see `CHANGELOG.md`'s historical entries for the recorded measurements — only the standalone result documents are gone. If this report is read later as a reference for how the multi-hop claim was substantiated, follow the citation to `CHANGELOG.md` rather than the now-nonexistent `docs/results/incident-benchmark-v1.md` path. Original prose above is left unedited, consistent with this report's established convention of correcting via addenda rather than rewriting dated findings.
