# Implementation Plan — postgres-graph-rag

**Derived from:** `demo-readiness-review-2026-08-16.md` (including its §13 second-pass addendum) + direct re-inspection of the repository.
**Status:** This document was generated and then immediately revalidated in the same pass (see §25, Plan Validation History, for what changed between the first draft and this final version). It is the single authoritative implementation document — no second plan file exists.

---

## 1. Executive Summary

`demo-readiness-review-2026-08-16.md` found **no confirmed code-level P0** and only a handful of real, small findings: one environment trap (stale `.venv`), one dead documentation link, one non-English inline comment, one benchmark-vs-README mismatch, one demo-narration overclaim discovered by actually running the demo live, and a short, already-self-documented list of deferred production work. There is intentionally **no large body of "implementation work"** here — inventing substantial tasks for a prototype that the review verified works correctly would itself be a review failure (over-engineering the plan to look thorough). This plan is deliberately small and heavily weighted toward *today's* demo-safety items; the production-evolution items are recorded as **Deferred Work**, not scheduled.

**What changed in this document vs. a naive pass-through of the review:** one task originally considered (enlarging the demo corpus to make the live vector-vs-graph comparison "honestly" differentiate) was drafted, then rejected during revalidation as a pre-demo scope change with no offsetting benefit — see §25.

---

## 2. Source Review Validation

Every task below was checked against the live repository this session, not assumed from `demo-readiness-review-2026-08-16.md`'s prose:

| Review finding | File/claim checked | Verified state |
|---|---|---|
| P1-1: dead `docs/demo.md` link | `README.md:24` → `[CTO demo](docs/demo.md)`; `find . -iname demo.md` | **Confirmed** — link target does not exist |
| P2-2: `uv.lock` version drift | Working tree: `uv.lock` line 1325 vs `pyproject.toml` line 3; committed state: `git show HEAD:uv.lock` | **Corrected on recheck — only partially resolved.** The working tree reads `0.1.0` (matches `pyproject.toml`), but `git show HEAD:uv.lock` still reads `0.7.0` — the fix was never committed. TASK-01's `uv sync --extra dev` will keep regenerating the same uncommitted working-tree diff without ever fixing the committed state. See TASK-06a below (new, added on recheck) |
| P2-3: German comment | `.github/workflows/publish.yml:21` | **Confirmed**: `enable-cache: false # CACHE DEAKTIVIERT UM FEHLER ZU VERMEIDEN` |
| P2-4: `bench_scale.py` measures legacy engine | `benchmarks/bench_scale.py:23,155` | **Confirmed**: imports and instantiates `DatabaseManager` only, never `SecureGraphStore`/`TenantGraphRAG` |
| §13.2: demo narration overclaim | `demo.py:126-134` (`_query` calls `engine.retrieve()`, not `answer()`); live run of `--mode vector` vs `--mode hybrid_graph` | **Confirmed live, this session**: identical "Relevant Passages" in both modes; only `hybrid_graph` adds the Entities/Relationships section |
| P0-1: venv corruption | Current `.venv` state | Not reproducible right now (environment-dependent); procedure itself is sound and worth keeping as a pre-flight step regardless |
| P3 items (migration framework, node pruning, hierarchical summarization, OAuth, `split_entity`, chunk/document identity, namespace scaling) | `README.md` roadmap section, `docs/architecture.md` "Known limits" | **Confirmed accurate self-reporting** — not re-litigated line-by-line here since the review already traced each one; carried forward as Deferred Work only |

---

## 3. Current Architecture (as verified, not assumed)

Restated only to the extent it constrains what follows — full detail is in `demo-readiness-review-2026-08-16.md` §4, independently re-derived twice now (original pass + §13.1 second pass) from the actual import graph, including lazy in-function imports:

```
Entry points (pyproject.toml [project.scripts], zero internal importers):
  demo.py, mcp_server.py, evaluation.py

core.py           -- factory/facade: for_tenant() lazily imports tenancy.SecureGraphStore
                     + tenant_engine.TenantGraphRAG (core.py:217-218); also still hosts the
                     frozen legacy add_texts()/query()/query_structured() path over DatabaseManager
tenancy.py        -- Architectural Core: RLS DDL, SecureGraphStore, 6 internal importers
tenant_engine.py  -- Architectural Core: TenantGraphRAG orchestration
extractor.py      -- Architectural Core: highest in-degree (6 importers), used by every path incl. offline
database.py       -- split: leaf utility functions (used by tenancy.py, real) + DatabaseManager
                     class (legacy-only, unreachable from any of the 3 real entry points)
grounding.py / verification.py / model_verifier.py -- entailment/grounding pipeline, not wired into demo.py
communities.py    -- reachable only via mcp_server.py, not demo.py/evaluation.py
```

None of the tasks below change this shape. No task in this plan touches `tenancy.py`, `tenant_engine.py`, `database.py`'s leaf functions, or the grounding pipeline's internals — the review found no defect there, and §6/§16 of the review explicitly say not to touch them before the demo.

---

## 4. Target Architecture

**No target architecture change is proposed by this plan.** The review's verdict was that the current architecture is appropriate for a prototype heading toward production (ADR-001, ADR-002 both hold up under re-inspection). Introducing a "target architecture" section with new abstractions here would itself be over-engineering relative to what the evidence supports. The only architecture-adjacent item is **already-deferred, self-documented** production work (§22, Deferred Work) — none of it is being designed or scheduled now.

---

## 5. Architecture Migration Strategy

**Not applicable.** There is no migration between architectures in this plan — only documentation fixes, one comment normalization, and operational rehearsal steps. Stating a migration strategy here would be inventing structure the work doesn't need.

---

## 6. Implementation Workstreams

| Workstream | Contains | Timing |
|---|---|---|
| **A — Demo-Critical** | TASK-01…04 | Must complete before the CTO demo (today) |
| **B — Low-Risk Hygiene** | TASK-05…06 | Optional, only if time remains after A |
| **C — Deferred / Post-Demo** | Listed in §22, not tasked in detail | Explicitly NOT scheduled for today |

---

## 7. Detailed Tasks

### Workstream A — Demo-Critical

---

**TASK-01 — Pre-flight: rebuild `.venv` and dry-run the full demo sequence**
- **Type:** Operational (no code change)
- **Status:** VALID → **VERIFIED (executed 2026-08-16)**. `rm -rf .venv` hit a transient "Directory not empty" error on the first attempt (a background process briefly holding a file lock) — retried immediately and it succeeded; not a repository issue, noted here per the "document deviations" rule. `uv sync --extra dev` completed clean. `uv run postgres-graph-rag-demo --help` and `uv run python -m pytest -q` (`285 passed, 7 deselected`) both verified from the freshly rebuilt venv. Full `setup --reset` → `ingest` → `evaluate` → `query` sequence run end-to-end with no errors.
- **Problem:** A `.venv` that was rebuilt in place across a Python version bump can leave a broken editable install where `uv run <console-script>` and `uv run pytest` fail with `ModuleNotFoundError`, even though `uv run python -c/-m` still works (misleadingly making it look like only some commands are broken).
- **Evidence:** `demo-readiness-review-2026-08-16.md` P0-1; reproduced and fixed in that session. Not currently reproducible in this session's environment (the venv here is healthy), which is expected — the failure mode depends on the specific machine's venv history, not the repo.
- **Root cause:** Local environment state, not a repository defect (`.venv/` is correctly gitignored — confirmed via `git ls-files .venv` returning nothing).
- **Fix:** `rm -rf .venv && uv sync --extra dev`, then run the exact demo sequence once (§ demo flow in `demo-readiness-review-2026-08-16.md` §7) end-to-end on the **actual machine** the demo will run from.
- **Dependencies:** None. Do this first.
- **Verification:** `uv run postgres-graph-rag-demo --help` succeeds; `uv run python -m pytest -q` reports `285 passed, 7 deselected`; full `setup --reset` → `ingest` → `evaluate` → `query` sequence completes with no errors.
- **Effort:** 10–15 minutes.
- **Definition of Done:** All four demo commands run clean, in order, from a freshly rebuilt venv, on the presenting machine, today.

---

**TASK-02 — Fix dead `docs/demo.md` link in `README.md`**
- **Type:** Documentation
- **Status:** VALID → **VERIFIED (executed 2026-08-16)**. `README.md:24` changed exactly as specified: `[CTO demo](docs/demo.md)` → `[CTO demo](#production-pilot-demo)`. `grep -n "docs/demo.md" README.md` now returns nothing; the anchor resolves to the existing `## Production pilot demo` section.
- **Problem:** `README.md:24` links to `docs/demo.md`, which does not exist anywhere in the repo (`find . -iname demo.md` returns nothing).
- **Root cause:** The "Production pilot demo" section (`README.md`, `#production-pilot-demo`) was written later in the same file and never had the top-of-file link retargeted to it.
- **Files:** `README.md` (only).
- **Fix (smallest correct option, per §6/§15 — no new file needed):** Change the link at `README.md:24` from `[CTO demo](docs/demo.md)` to `[CTO demo](#production-pilot-demo)`, pointing at the existing section that already contains the exact working demo sequence.
- **Alternative rejected:** Writing a new `docs/demo.md` was considered and rejected — it would duplicate content that already exists in `README.md`'s "Production pilot demo" section, creating two sources of truth for the same sequence. Repointing the link is strictly simpler and equally honest.
- **Dependencies:** None. Independent of TASK-01.
- **Tests:** None applicable (doc-only). Verification is manual: click/grep the link.
- **Verification:** `grep -n "docs/demo.md" README.md` returns nothing after the fix; the in-page anchor resolves.
- **Effort:** 5 minutes.
- **Definition of Done:** No reference to a non-existent `docs/demo.md` remains anywhere in the repo; the README's demo link resolves to real content.

---

**TASK-03 — Correct the demo narration for the vector-vs-hybrid_graph comparison**
- **Type:** Operational / presentation (no code change)
- **Status:** MODIFIED → **VERIFIED (re-run 2026-08-16)**. Re-executed both `query --mode vector` and `query --mode hybrid_graph` against a freshly reset schema; reconfirmed identical "Relevant Passages" (including the "Identity Team" text) in both, with only `hybrid_graph` adding the Entities/Relationships section. The corrected narration in this task still matches live output exactly — nothing about the underlying behavior changed since the finding was first made.
- **Problem:** `demo-readiness-review-2026-08-16.md` §7 instructs presenting `--mode vector` then `--mode hybrid_graph` on the same question to show "only the graph-aware mode surfaces Identity Team." Verified live this session: **both modes print identical retrieved passages**, including the "Identity Team" text, because the demo's 4-document corpus is too small to have distractors under either mode (`top_k=5` against 4 total documents retrieves nearly everything regardless of ranking).
- **Root cause (traced, not assumed):** `demo.py::_query` (`demo.py:126-134`) calls `engine.retrieve(question, namespace, mode=..., top_k=5, hops=3)` and prints `result.to_context_string()` — a passage dump, not an `answer()` call. The demo scenario (`DEMO_DOCUMENTS`, `demo.py:24`) has exactly 4 documents. With that few candidates, vector-mode ranking quality doesn't matter — everything gets retrieved either way. This is a demo-corpus-size artifact, not a retrieval-quality difference.
- **Why the root cause exists:** The demo is deliberately small so it runs offline, deterministically, with no LLM key and no cost — a correct and worthwhile design choice for a repeatable CTO demo. The narration script just didn't account for the side effect of that smallness on this specific comparison.
- **Correct fix (narration only, not code):** Don't claim vector mode "can't surface" the passage. Say: *"Notice both modes retrieve the same four passages here — the corpus is intentionally tiny so the demo runs with no LLM key. What `hybrid_graph` adds is the explicit reasoning path at the bottom: `checkout-service → depends_on → auth-service → owned_by → Identity Team`. At real scale that structural difference is what keeps recall high instead of collapsing — see our 60-document benchmark: 12.5% multi-hop recall for vector/hybrid vs 100% for hybrid+graph (`docs/results/incident-benchmark-v1.md`)."*
- **Rejected alternative (and why):** Adding a 5th "distractor" document to the demo corpus so vector-mode genuinely fails on this query was considered. Rejected: it's a scope change to a scenario that currently works perfectly and is verified end-to-end; introducing new content hours before presenting adds regression surface for zero necessary benefit, since the corrected narration already tells a true, still-compelling story. This mirrors `demo-readiness-review-2026-08-16.md` §6/§16's explicit "don't enrich the demo corpus the night before" guidance — re-confirmed as still correct advice under this specific case.
- **Dependencies:** None; independent of TASK-01/02, but should be rehearsed before TASK-01's dry run so the dry run also validates the narration, not just the commands.
- **Verification:** Run both `query` commands once more, read the outputs side by side, and rehearse the corrected sentence out loud at least once.
- **Effort:** 10 minutes (read + rehearse).
- **Definition of Done:** Presenter can say the corrected narration from memory and it matches what the terminal actually shows when run live.

---

**TASK-04 — Rehearse Docker failure recovery**
- **Type:** Operational (no code change)
- **Status:** VALID → **VERIFIED (executed 2026-08-16)**. Ran `docker compose down` → `docker compose up -d postgres` → `docker compose logs postgres` → `docker compose ps`; container reached `healthy` within ~15 seconds of the up command. Recovery sequence confirmed to work, not just documented.
- **Problem:** `docker compose up -d postgres` failing (daemon not running, port 5432 already bound) is an external-environment risk, not a code defect — `docker-compose.yml` itself was reviewed and is correct (healthcheck present, standard `pgvector/pgvector:pg16` image).
- **Fix:** Before the CTO arrives: confirm Docker Desktop is running, confirm port 5432 is free, and have `docker compose down && docker compose up -d postgres && docker compose logs postgres` memorized as the one recovery sequence, plus `docker compose ps` to confirm `healthy` status.
- **Dependencies:** Do after TASK-01 (so the venv + Docker + demo sequence is validated together in one dry run).
- **Verification:** One full dry run today ends with `docker compose ps` showing the postgres container healthy and all four demo commands succeeding.
- **Effort:** 10 minutes.
- **Definition of Done:** Presenter has actually typed the recovery sequence once today, not just read it.

---

### Workstream B — Low-Risk Hygiene (optional)

---

**TASK-05 — Normalize the German inline comment in `publish.yml`**
- **Type:** Code cleanup (single line)
- **Status:** VALID, but genuinely optional
- **Problem:** `.github/workflows/publish.yml:21` — `enable-cache: false # CACHE DEAKTIVIERT UM FEHLER ZU VERMEIDEN` is the only non-English content in an otherwise all-English codebase.
- **Fix:** Replace with `enable-cache: false  # cache disabled to avoid errors`.
- **Dependencies:** None. Touches CI config, so verify CI still parses correctly (YAML comment, no functional risk) — but per `demo-readiness-review-2026-08-16.md` §6, CI workflows are explicitly "don't touch before the demo" territory since they're correct and unrelated to demo risk.
- **Recommendation:** **Defer to after the demo.** Zero demo-day value, non-zero (if small) risk of a fat-fingered YAML edit to a file the review explicitly says to leave alone before presenting. Listed here for completeness, not for execution today.
- **Effort:** 2 minutes, whenever it's done.
- **Definition of Done:** Comment reads in English; `.github/workflows/publish.yml` still parses (`git diff` shows only the comment line changed).

---

**TASK-06 — Disclose that `bench_scale.py`'s published numbers measure the legacy engine, not the RLS-secured path**
- **Type:** Documentation
- **Status:** VALID → **VERIFIED (executed 2026-08-16)**. Added the disclosure as a blockquote immediately under the Benchmarks section's intro paragraph, before the results table (`README.md`, in the `## Benchmarks` section). **Deviation from the plan's exact wording (documented per the execution rules):** the plan's suggested sentence ended "...see Deferred Work" — that term is `demo-readiness-implementation-plan-2026-08-16.md`-specific and doesn't exist as a concept in `README.md`. Changed to "...see the Roadmap," pointing at `README.md`'s actual "🗺️ Roadmap & Future Vision" section, which is where a reader of the README would actually find forward-looking items. Same intent, correct target.
- **Problem:** `README.md`'s "Benchmarks" section presents `bench_scale.py`'s latency/throughput table without noting which engine it measures. `bench_scale.py:23,155` confirms it imports and instantiates `DatabaseManager` only — never `SecureGraphStore`/`TenantGraphRAG`, the path the same README calls "the recommended, actively-developed path."
- **Root cause:** The benchmark script was written against the (then-current) legacy engine before the secure/tenant path existed, and the README's benchmarks section was never revisited to note the engines have since diverged.
- **Fix:** Add one sentence to the README's "Benchmarks" section: *"These numbers measure the legacy `DatabaseManager` engine; they do not include RLS policy-evaluation or transaction-local tenant-context overhead. A `SecureGraphStore`-based benchmark variant is planned — see Deferred Work."*
- **Rejected alternative:** Building a `SecureGraphStore` variant of `bench_scale.py` tonight. Rejected for today — it's real, useful work, but it's net-new benchmark code the night before a demo, with no demo-day payoff (the demo doesn't cite `bench_scale.py`'s numbers at all). Moved to Deferred Work (§22) as TASK-D3.
- **Dependencies:** None.
- **Verification:** The disclosure sentence appears in `README.md`; re-read the Benchmarks section once to confirm it now can't be misread as measuring the secured path.
- **Effort:** 5 minutes.
- **Definition of Done:** Anyone reading the Benchmarks table can't come away believing it includes RLS overhead.

---

**TASK-06a — Commit the already-regenerated `uv.lock` (added on recheck, not in the original draft)**
- **Type:** Repository hygiene (one file, no code change)
- **Status:** NEW → **PREPARED, NOT COMMITTED (2026-08-16).** The working-tree file is confirmed ready (`git diff uv.lock` shows exactly the expected one-line `0.7.0` → `0.1.0` change, nothing else). Execution stopped short of the actual `git commit` — committing is a repository-visible action, and per standing operating rules, commits are only made on the user's explicit instruction, not implied by a task list. Asked the user directly; they chose to leave all changes (including README's TASK-02/06 edits) uncommitted in the working tree for their own review. **This is not a deviation from the plan's intent** — the fix is done and verified in the working tree; only the final `git commit` step is deferred to the user's discretion.
- **Problem:** The working tree's `uv.lock` reads `postgres-graph-rag` version `0.1.0` (matching `pyproject.toml`, regenerated by an earlier `uv sync --extra dev` in this session), but the **committed** `uv.lock` (`git show HEAD:uv.lock`) still reads `0.7.0`. A first pass of this plan claimed this drift was "resolved" by checking only the working tree — it was not resolved at the repository level, and a fresh `git clone` would still see `0.7.0`.
- **Root cause:** The lockfile was regenerated locally at some point but the resulting diff was never committed.
- **Compounding factor:** TASK-01 (`rm -rf .venv && uv sync --extra dev`) will keep re-regenerating this exact same uncommitted diff every time it's run, on every machine — the fix silently reappears and silently never lands, unless someone commits it once.
- **Fix:** Confirm `0.1.0` is the intended version (it is — matches `pyproject.toml`, the source of truth), then `git add uv.lock && git commit -m "Sync uv.lock version to match pyproject.toml (0.1.0)"`.
- **Dependencies:** None. Independent of every other task.
- **Priority:** Low — does not affect `uv sync`/`uv run` behavior either way, so it is not a demo blocker. Do it if there's spare time; otherwise it's safe to leave for after the demo, same as TASK-05.
- **Verification:** `git diff uv.lock` shows no output after committing; `git show HEAD:uv.lock | grep -A1 'name = "postgres-graph-rag"'` reads `0.1.0`.
- **Effort:** 2 minutes.
- **Definition of Done:** `git status` shows a clean working tree with respect to `uv.lock`, and `HEAD`'s copy matches `pyproject.toml`.

---

## 8. File-by-File Change Map

| File | Task | Change | Confirmed to exist? |
|---|---|---|---|
| `README.md` | TASK-02 | Line 24: retarget link from `docs/demo.md` to `#production-pilot-demo` | Yes (725 lines, read in full during review) |
| `README.md` | TASK-06 | Add one disclosure sentence to the Benchmarks section (~line 354-368) | Yes |
| `.github/workflows/publish.yml` | TASK-05 (deferred) | Line 21: comment text only | Yes |
| `uv.lock` | TASK-06a (optional) | Commit already-regenerated working-tree content (`0.7.0` → `0.1.0`), no further edit needed | Yes — diff confirmed via `git diff uv.lock` |
| *(no other files change)* | — | TASK-01, 03, 04 are operational, no file edits | — |

No task in this plan touches `postgres_graph_rag/*.py` or `tests/*.py`. This is intentional — the review found no confirmed code defect, and inventing a code change here would violate §14/§15 (unnecessary work, over-engineering).

---

## 9. Dependency Order

```
TASK-01 (venv rebuild)
   ↓
TASK-04 (Docker check, part of the same dry run)
   ↓
TASK-02 (link fix) ──┐   TASK-03 (narration) — can run in parallel with TASK-02,
                     │    both independent of the dry run's command sequence
   ↓                 │
Full dry run of demo sequence (validates TASK-01/02/03/04 together)
   ↓
[optional, only if time remains] TASK-06, then TASK-05
```

TASK-02, TASK-03, and TASK-06 have no code dependency on each other or on TASK-01/04 and can be done in any order or in parallel; they're sequenced above only for a sane single-person work session, not because of a technical dependency.

---

## 10. Database Changes

**None.** No task in this plan touches schema, migrations, RLS policies, or data. Confirmed by re-reading every task above: TASK-01/03/04 are operational, TASK-02/05/06 are documentation/comment-only. Any database-related item that exists (see §22) is explicitly deferred and unscheduled.

---

## 11. API Changes

**None, and none proposed.** No public method signature, CLI flag, or MCP tool changes anywhere in this plan. (§13.2's fix is narration-only — it explicitly does not add a `--corpus-size` flag or otherwise change `demo.py`'s CLI surface, which would be exactly the kind of last-minute change the review warns against.)

---

## 12. Testing Strategy

No task in this plan changes behavior, so no new automated tests are required — adding tests for a doc-link fix or a comment translation would be busywork with zero confidence value (§9 of the validation instructions: "if the plan adds tests that don't provide meaningful confidence, remove them"). The verification method for each task is stated inline in §7 and is manual/operational:

| Task | Verification |
|---|---|
| TASK-01 | `uv run python -m pytest -q` → `285 passed, 7 deselected`; full demo sequence runs clean |
| TASK-02 | `grep -n "docs/demo.md" README.md` returns nothing |
| TASK-03 | Live re-run of both `query` modes, narration rehearsed |
| TASK-04 | `docker compose ps` shows `healthy` after a deliberate down/up cycle |
| TASK-05 | `git diff` shows only the comment text changed; workflow YAML still valid |
| TASK-06 | Benchmarks section re-read, disclosure present |

The existing 285-test suite and `ruff check .` (both re-run live this session, both clean) remain the actual regression safety net — this plan doesn't need to add to it because it isn't changing any code the suite covers.

---

## 13. Security Changes

**None proposed, none needed.** The review's security-relevant findings (RLS fail-closed design, parameterized metadata filters, MCP loopback guard, `.env` correctly gitignored) were all independently re-verified in `demo-readiness-review-2026-08-16.md` §13.1 and found accurate with no confirmed vulnerability. No task in this plan touches `tenancy.py`, `filters.py`, or `mcp_server.py`'s auth guard — consistent with §6's explicit "do not touch" list.

---

## 14. Performance Changes

**None proposed for today.** TASK-06 is a *disclosure*, not a performance fix. The one performance-adjacent idea that surfaced during drafting — building a `SecureGraphStore` benchmark variant — was moved to Deferred Work (§22, TASK-D3) precisely because there is no current evidence it's urgent (the demo doesn't cite the numbers) and building it under time pressure risks a rushed, unreliable benchmark that's worse than not having one.

---

## 15. Observability Changes

**None.** Not touched by the review's findings, not touched by this plan.

---

## 16. Code Removal

**None proposed.** `demo-readiness-review-2026-08-16.md` §5 explicitly states: *"I found no additional confirmed bugs, no dead code, no redundant implementations..."* — re-confirmed in §13's second pass with no new dead-code finding. Nothing in this plan recommends deleting anything. The legacy `core.py`/`DatabaseManager` path remains on its own stated deprecation schedule (target: `1.0.0`), per the project's own removal policy in `README.md`'s "Legacy API migration" section — not accelerated by this plan.

---

## 17. Backward Compatibility

**Not affected.** TASK-02/05/06 are documentation/comment changes with no runtime effect. TASK-01/03/04 are operational. No consumer of the library, CLI, or MCP server is affected by anything in this plan.

---

## 18. Risk Register

| Task | Risk if done | Risk if NOT done | Net recommendation |
|---|---|---|---|
| TASK-01 | Negligible (well-understood, reversible: `rm -rf .venv && uv sync`) | High — the single most likely thing to actually break tomorrow morning | Do it |
| TASK-02 | Negligible (one link) | Low functional risk, real credibility risk if the CTO clicks it | Do it |
| TASK-03 | Negligible (no code touched) | **High** — the review found this is the one place where following the *original* script live would visibly undercut the pitch in front of the CTO | Do it — highest priority of all tasks |
| TASK-04 | Negligible | Low-moderate (Docker state is genuinely outside repo control) | Do it |
| TASK-05 | Low (one-line YAML comment, but touches a file §6 says to leave alone) | None | Defer past the demo |
| TASK-06 | Negligible (doc sentence) | Moderate only if CTO specifically asks about RLS overhead in benchmarks | Do only if time remains |
| TASK-06a | Negligible (`git commit` of an already-generated file) | None (cosmetic metadata drift only) | Do only if time remains |

---

## 19. Effort Estimates

| Task | Estimate | Basis |
|---|---|---|
| TASK-01 | 10–15 min | One shell command + one full command-sequence dry run, timed during this session's own re-run |
| TASK-02 | 5 min | Single-line edit |
| TASK-03 | 10 min | Read + rehearse once; no code change |
| TASK-04 | 10 min | One deliberate down/up/logs cycle |
| TASK-05 | 2 min | Single comment line |
| TASK-06 | 5 min | One sentence addition |
| TASK-06a | 2 min | `git add && git commit` of a file already regenerated |
| **Total, Workstream A (mandatory)** | **~35–45 min** | |
| **Total, Workstream B (optional)** | **~9 min** | |

These are lower than a first-pass estimate might suggest because none of these tasks touch code paths covered by the 285-test suite — there is no "write test, run test, debug, re-run" loop for any of them, which is the actual reason implementation estimates usually balloon. Estimates were sanity-checked against actually doing TASK-01 and TASK-03's verification live in this session.

---

## 20. Implementation Checklist

- [x] TASK-01: `.venv` rebuilt; full demo sequence dry-run passes — **VERIFIED 2026-08-16**
- [x] TASK-04: Docker down/up/logs cycle rehearsed; `docker compose ps` shows healthy — **VERIFIED 2026-08-16**
- [x] TASK-02: `README.md` demo link retargeted to `#production-pilot-demo` — **VERIFIED 2026-08-16**
- [x] TASK-03: Narration reconfirmed live against a fresh reset — **VERIFIED 2026-08-16** (still needs a spoken rehearsal by the presenter — that part is inherently not something to execute on their behalf)
- [x] *(optional)* TASK-06: Benchmarks disclosure sentence added — **VERIFIED 2026-08-16** (wording adapted, see task entry)
- [x] *(optional)* TASK-06a: `uv.lock` regenerated content confirmed correct in the working tree — **PREPARED, NOT COMMITTED** (user chose to hold all commits for their own review)
- [ ] *(optional, defer if rushed)* TASK-05: German comment translated — **not executed, per the plan's own explicit recommendation to leave CI config untouched before the demo**

---

## 21. Verification Strategy

1. Run `uv run python -m pytest -q` — expect `285 passed, 7 deselected`.
2. Run `uv run ruff check .` — expect `All checks passed!`.
3. Run the full demo sequence exactly as in `demo-readiness-review-2026-08-16.md` §7, end to end, on the presenting machine.
4. Run both `query --mode vector` and `query --mode hybrid_graph` on the checkout-service question; confirm the output matches what TASK-03's narration describes.
5. `grep -n "docs/demo.md" README.md` → expect no output.
6. `docker compose ps` → expect the postgres service `healthy`.

All six of these were executed at least once during this session's own validation pass (§2 above) — this isn't a hypothetical checklist, it's what was actually run.

---

## 22. Deferred Work (explicitly NOT scheduled for today)

Carried forward from `demo-readiness-review-2026-08-16.md`'s P3 list and TASK-06's rejected alternative, unchanged in substance, only reorganized here as named tasks for whenever post-demo work is planned:

| ID | Item | Why deferred |
|---|---|---|
| TASK-D1 | Versioned/reversible migration framework (currently one idempotent `CREATE ... IF NOT EXISTS` script + advisory lock) | Real production concern, not blocking current scale/use case; already self-documented |
| TASK-D2 | Zero-mention node pruning (orphaned entities left in place) | Same as above |
| TASK-D3 | `SecureGraphStore`-based variant of `bench_scale.py`, so RLS overhead is actually measured | New-code task under time pressure is worse than no benchmark; do carefully, post-demo |
| TASK-D4 | Hierarchical map/reduce community summarization (large communities currently truncated) | Self-documented limitation, fine at current scale |
| TASK-D5 | OAuth authorization server for MCP HTTP transport | Deployer-supplied auth is a deliberate, documented design choice, not an oversight |
| TASK-D6 | `split_entity()` (undoing an entity merge) | Documented as structurally impossible with the current schema (no per-mention alias provenance after merge) — needs a schema change, not a quick patch |
| TASK-D7 | Full document-identity tracking for chunk idempotency across documents in one namespace | Partially already built (`documents`/`document_chunks`/`entity_mentions` tables exist); residual gap is narrower than README's wording alone suggests, per `demo-readiness-review-2026-08-16.md` §5's P3 note — worth re-scoping before committing effort, not worth doing blind |
| TASK-D8 | Domain-specific evaluation of fuzzy entity-resolution thresholds + a real-provider extraction-quality canary (20–30 anonymized documents, cost-capped) | `docs/evaluation.md`'s own stated next step; requires real budget/data decisions, not a code-only task |
| TASK-D9 | Re-run `benchmarks/bench_scale.py` at real multi-tenant scale to check whether the HNSW index is still used once per-namespace row counts grow | Depends on TASK-D3 existing first (measuring the right engine) |

None of these have detailed task breakdowns (files, steps, DoD) in this plan — doing so now would be designing production work before the prototype has even been demoed, which is scope the review explicitly says is out of bounds for the next 24 hours.

---

## 23. Definition of Done (plan-level)

The plan as a whole is done when:
- All Workstream A tasks are checked off in §20, verified via §21's six commands, on the actual presenting machine.
- The presenter can recite TASK-03's corrected narration without notes.
- No Workstream A task has introduced a code change (confirmed via `git status`/`git diff` showing only `README.md` touched, if TASK-02/06 were done).

---

## 24. Final Implementation Sequence

1. TASK-01 (venv rebuild)
2. TASK-04 (Docker check) — same sitting as TASK-01, one combined dry run
3. TASK-02 (link fix)
4. TASK-03 (narration rehearsal)
5. *(only if time remains)* TASK-06, then TASK-06a, then TASK-05
6. Final full dry run of the demo sequence, narrated out loud, timed

---

## 25. Plan Validation History

### Initial Plan (first draft, before revalidation)

The first draft, produced directly from `demo-readiness-review-2026-08-16.md`, proposed six tasks corresponding to P0-1/P1-1/P1-2/P2-3/P2-4/§13.2, plus — as an initial idea for TASK-03 — **adding a 5th "distractor" document to the demo corpus** so that `--mode vector` would genuinely fail to retrieve the "Identity Team" passage, making the live comparison in `demo-readiness-review-2026-08-16.md` §7 literally true rather than narration-dependent. It also initially proposed building a `SecureGraphStore` variant of `bench_scale.py` as part of TASK-06 rather than a one-sentence disclosure.

### Revalidation Findings

Applying the reconciliation checklist (three-way check between review, repo, and plan) surfaced two problems with the initial draft:

1. **The distractor-document idea, while it would make the demo narrative literally true, is itself exactly the kind of pre-demo scope change `demo-readiness-review-2026-08-16.md` §6/§16 warns against** — modifying the demo's scenario data (`demo.py::DEMO_DOCUMENTS`) hours before presenting, to a scenario that currently runs clean end-to-end with 285/285 tests passing. It trades a narration fix (zero regression risk) for a code+data change (non-zero regression risk) to solve a presentation problem, which is a worse trade under one-day time pressure. This is a textbook case of §16's "is this too complex/risky relative to the actual problem" check.
2. **Building `bench_scale.py`'s secured-path variant tonight has no demo-day payoff** — the demo doesn't cite `bench_scale.py`'s numbers at all (confirmed: `evaluate`'s output is the only benchmark figure shown live, and it already runs against the real secure path). Writing new benchmark code under time pressure, with no time to validate its own correctness the way the existing benchmark was validated, risks producing a second, unreliable set of numbers — worse than clearly disclosing the gap in one sentence.

### Corrections

- TASK-03 was rewritten from "fix the demo corpus" to "fix the narration" — no code or data change, same underlying honesty goal, much lower risk, achievable in 10 minutes instead of requiring a full re-validation of the demo's evaluate/recall numbers against a changed corpus.
- TASK-06 was narrowed from "build a secured-path benchmark" to "add a one-sentence disclosure," with the secured-path benchmark moved to Deferred Work as TASK-D3, explicitly scoped for post-demo, unhurried work.
- Confirmed P2-2 (`uv.lock` drift) was believed resolved in the current repo state at the time and removed from the active task list entirely — **this specific conclusion was itself wrong; see the second recheck entry below.**
- Verified every file reference (§8) against the actual repository rather than trusting `demo-readiness-review-2026-08-16.md`'s line numbers as still current — all confirmed accurate.
- Confirmed no task in the plan touches `tenancy.py`, `tenant_engine.py`, `database.py`, or the grounding pipeline, consistent with the review's explicit "do not touch" list — no task needed to be split or merged as a result, since none were proposed there in the first place.

### Second Recheck (prompted by "recheck again — does demo-readiness-review-2026-08-16.md need updating too?")

This pass re-ran `git status`/`git diff` on the repository rather than trusting the first revalidation's own conclusions, per the same "repository is the ultimate source of truth" principle the first pass was supposed to apply.

- **Finding: the first revalidation's P2-2 conclusion was itself wrong.** It checked the *working tree's* `uv.lock` against `pyproject.toml` (both `0.1.0`) and concluded the drift was resolved. Rechecking against `git show HEAD:uv.lock` — the actual committed state, what a fresh clone sees — shows `0.7.0`, unchanged from the original finding. The fix was never committed; it only exists locally. This is exactly the class of error the validation instructions warn about (checking a symptom's current appearance without checking whether the underlying state actually changed).
- **Correction:** Reinstated P2-2 as an active (optional, low-priority) item — **TASK-06a**, added to Workstream B, §8's file map, §18's risk register, §19's effort table, §20's checklist, and §24's sequence. Also corrected the corresponding claim in `demo-readiness-review-2026-08-16.md` (§13.4 and a new §14) so both documents now agree on the true state instead of both repeating the same working-tree-only check.
- **Everything else re-verified and held:** all other file/line references (`README.md:24`, `publish.yml:21`, `bench_scale.py:23,155`, `demo.py:126-134`) were spot-checked again against the live repo and matched exactly; 285/285 tests and clean `ruff` were re-confirmed again.

### Final State

This plan is implementation-ready, now including one small correction the first revalidation pass itself missed: every task maps to a verified, currently-true repository fact — verified against committed state (`git HEAD`), not just the local working tree; no task invents unnecessary code change, database change, API change, or new architecture; the one high-risk item found during the first revalidation (the demo narration, §13.2) remains the top-priority task; TASK-06a closes the one gap the first pass left open without realizing it; and post-demo production work is clearly separated into Deferred Work rather than smuggled into today's scope. A Senior Engineer (or, in this case, the presenter themselves) can execute Workstream A end-to-end in under an hour with no ambiguity about what "done" means for each item.

---

## 26. Addendum: Legacy Engine Removed (separate, post-demo initiative)

§16 above ("Code Removal: None proposed... not accelerated by this plan") was correct for this plan's own stated scope: a same-day demo, where deleting the legacy `DatabaseManager`/`PostgresGraphRAG` legacy-method path had no demo-day payoff and real regression risk under time pressure. That reasoning stands unedited.

**Separately, after the demo**, once it was established there are no customers and no external installs of this project, the original "wait for `1.0.0`" removal timeline was revisited: it existed to protect external callers that don't exist yet. The legacy engine was removed entirely as its own dedicated, planned piece of work — see `/Users/ayushmishra/.claude/plans/so-if-i-am-merry-lemur.md` for the full removal plan (import-graph tracing, test-coverage mapping, exact file-by-file changes) and `docs/decisions/003-remove-legacy-engine.md` for the ADR. `CHANGELOG.md`'s top entry documents exactly what was removed and what was kept (notably, the legacy-data migration feature, which never depended on `DatabaseManager` and survives untouched).

This addendum exists so a reader of this plan later doesn't mistake §16's "not accelerated by this plan" for the current state of the repository — it described this plan's scope at the time, not a permanent decision.

---

## 27. Addendum: `docs/results/` Removed

TASK-03's narration above cites `` `docs/results/incident-benchmark-v1.md` `` as the evidence source for the 12.5%→100% multi-hop recall claim. That file, and the whole `docs/results/` directory, have since been deleted — the demo this plan prepared for has already happened, so the backup/citation role that file played is no longer live. The recorded numbers themselves are preserved in `CHANGELOG.md`'s historical entries, not lost. See `docs/reviews/demo-readiness-review-2026-08-16.md` §16 for the fuller version of this same note.

## 28. Addendum: `database.py` Split Resolved; Legacy-Removal Residue Cleaned Up (0.2.0)

§3's architecture diagram describes `database.py` as *"split: leaf utility functions (used by tenancy.py, real) + `DatabaseManager` class (legacy-only, unreachable from any of the 3 real entry points)."* The `DatabaseManager` half of that split is now gone — it was deleted along with the rest of the legacy engine (§26). `database.py` today is only the leaf utility functions; the module's name understates that (it no longer manages a database connection at all), which is a tracked, deliberately deferred rename (`_db_utils.py`), not something this addendum's release touched.

Separately, a follow-up pass (`LEGACY_DELETION_PLAN.md`, released as `0.2.0`) found and closed three pieces of residue the original removal (§26) left behind: a dead `postgres_url` constructor parameter with zero reads that was pushing an unnecessary admin DSN requirement onto `postgres-graph-rag-eval`/`postgres-graph-rag-mcp`, a parallel `grounding.VerifiedAnswerResult` type with no producer, and a duplicated (and diverged) store-construction path between `for_tenant()` and the MCP server lifespan. See `docs/decisions/003-remove-legacy-engine.md`'s addendum for the full account.
