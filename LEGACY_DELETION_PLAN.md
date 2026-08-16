# Legacy Deletion & Consolidation Plan

**Repository:** `postgresql-graph-rag`
**Date:** 2026-08-16
**Revision:** v4 — supersedes v3 after a fourth principal-level review.
**Status:** Follow-on plan not implemented. The prerequisite engine removal
exists **uncommitted** in the working tree (see §0).

**Release parameters (decided, not open):**

```text
Version baseline:   0.1.0   (pyproject.toml + tag v0.1.0 — authoritative)
Removal release:    0.2.0   (breaking API removal, pre-1.0 minor bump)
L4 (grounding):     separate refactor PR, after this change set
L5 (database.py):   deferred outright; optional future rename to _db_utils.py
```

**Gate rule governing this entire plan:**

> **Correctness and packaging evidence blocks the merge.
> External-provider verification blocks the release.
> Performance evidence blocks performance claims.**

---

## 0. Headline finding

**The major legacy architecture has already been deleted.** It is sitting
uncommitted in the working tree: `DatabaseManager` stripped from
`database.py`, the legacy `PostgresGraphRAG.setup()`/`.add_texts()`/`.query()`/
`.query_structured()` methods removed from `core.py`, and
`tests/test_core.py`, `tests/test_integration.py`, `tests/test_scenarios.py`,
`tests/timing_utils.py`, `benchmarks/bench_scale.py`, `docs/results/` deleted.
`docs/decisions/003-remove-legacy-engine.md` records the ADR.

That work is sound and is not re-litigated here. This plan covers what a trace
of the *current* tree finds: residue the removal left behind, one dead parallel
type it never touched, one duplicate construction path, and two items
explicitly deferred with reasons.

### Verified baseline

| Check | Result |
|---|---|
| `pytest -q` (Postgres reachable) | **259 passed, 0 failed, 0 skipped** |
| — of which `tests/test_tenancy.py` | 83, against live Postgres via `load_dotenv()` (`tests/test_tenancy.py:33`) |
| `ruff check postgres_graph_rag tests benchmarks` | clean |
| `import postgres_graph_rag{,.demo,.evaluation,.mcp_server}` | OK |
| **`dist/` wheel** | **STALE — contains `class DatabaseManager` + 4 legacy `core.py` methods** |
| **`dist/` sdist** | **STALE — contains `DatabaseManager`, and ships `tests/test_core.py`** |

> ⚠️ **Verification trap.** `tests/test_tenancy.py` calls `load_dotenv()` then
> `skipif(not POSTGRES_URL)`. Without Postgres those 83 tests **silently skip
> and the suite still reports green**. Most of the call sites this plan edits
> live in that file. Every gate below is invalid unless the run reports
> **0 skipped**.

---

## 1. Component inventory

| ID | Component | Verdict | Phase |
|---|---|---|---|
| **L1** | `postgres_url` constructor parameter + CLI requirements | Dead compat shim — **delete atomically** | 4 |
| **L2** | `VerifiedAnswerResult` | Dead parallel type — **harden `AnswerResult` first, then delete** | 2 → 4 |
| **L3** | Stale CI / docs residue | **Fix** | 4 |
| **L6** | MCP duplicate store construction | Duplicate architecture — **fix** | 2 |
| **L4** | Citation-only rule implemented 3× | **Deferred** — separate grounding-refactor PR | — |
| **L5** | `database.py` vestigial module | **Deferred outright** | — |

### Cleared — investigated, found *not* legacy

- **`PostgresGraphRAG` as a class.** The canonical public facade and sole
  construction path to `TenantGraphRAG`. Thin, not duplicate. **Retain.**
- **`migrate_legacy_data=True` / `LEGACY_TENANT_ID` (`tenancy.py:843`).** A
  bounded, idempotent import adapter that reads old tables via raw SQL and
  writes into the secure schema. It never depended on `DatabaseManager`. The
  name says legacy; the code is live migration support. **Retain — do not
  delete on the strength of a name.**
- **The three `benchmarks/grounding/*_runner.py` files.** Three distinct
  measurement layers over one shared `gates.py`. **Retain.**
- **`OfflineExtractor` + the two fixture-map builders.** One class, two
  one-line callers. **Retain.**
- **`communities.py`.** Narrow reach (MCP only) but live, tested, documented.
  Narrow reach ≠ legacy. **Retain.**

---

## 2. Component detail

### L1 — `postgres_url` parameter and the admin-credential requirement

**Evidence.** `core.py:37` declares it required. AST analysis finds **zero
reads** anywhere in `core.py` — never assigned, never passed on. Its own
docstring (`core.py:57-60`) calls it *"accepted for constructor-signature
compatibility with existing callers."* ADR-003 established there are no such
callers. This is the shim the engine removal missed.

**Not cosmetic — it is a credential-handling defect.** Because the parameter is
required, both runtime CLIs enforce it:

- `mcp_server.py:373` — `parser.error("--postgres-url and --runtime-url ... are required")`
- `evaluation.py:227` — `raise SystemExit("POSTGRES_URL and PGR_RUNTIME_URL are required")`

Neither calls `setup_secure()`. Both **refuse to start without a privileged
admin DSN they never use**, pushing a superuser connection string into the
environment of processes architecturally meant to hold only the restricted
`NOBYPASSRLS` runtime role — the exact failure ADR-002 and `docs/security.md`
exist to prevent.

**Replacement.** None needed. `setup_secure(admin_url=...)` already takes its
own admin DSN (`core.py:105`); `for_tenant()` uses `runtime_url`.

**All references — 24 code call sites + 5 doc:**

| Location | Lines | Action |
|---|---|---|
| `core.py` | 37, 57-60 | Delete param + docstring paragraph |
| `demo.py` | 86, 100 | Drop argument |
| `evaluation.py` | 230; 226-228 | Drop argument; drop `admin_url` fetch and its half of the `SystemExit` guard |
| `mcp_server.py` | 377; 361, 373-374 | Drop argument; delete `--postgres-url` flag and its half of `parser.error` |
| `tests/test_tenancy.py` | 21 sites (552, 868, 911, 1035, 1070, 1122, 1178, 1234, 1283, 1323, 1374, 1497, 1614, 1661, 1704, 1773, 1808, 1844, 1966, 2539, +1) | Drop argument |
| `README.md` | 74, 115, 132, 195, 364 | Update examples; **line 364's comment is actively false** — it claims the parameter is "used only for `setup_secure()`" |

> 🚨 **This must be ONE ATOMIC COMMIT.**
>
> v1 of this plan split it into "remove call-site arguments, verify green" then
> "remove the parameter." **That intermediate state cannot exist.** With the
> parameter still declared without a default and no caller passing it, every
> construction raises:
>
> ```text
> TypeError: PostgresGraphRAG.__init__() missing required argument: postgres_url
> ```
>
> The reverse order fails identically. There is no green intermediate in either
> direction — parameter, all 24 call sites, both CLI flags, and the README
> examples change together in a single commit. Giving the parameter a temporary
> default would work but adds a transitional shim to a change whose whole point
> is removing one; rejected.

**Tests affected.** None deleted. No test asserts on `postgres_url`; all 21
sites are mechanical argument removals and retain full coverage.

**Retain.** `demo.py`'s `_urls()` admin-URL handling (lines 71-80) — genuinely
used by `_setup()` for `setup_secure()` and to derive `runtime_url` when
`PGR_RUNTIME_URL` is unset. `demo.py --admin-url` stays. `.env.example`'s
`POSTGRES_URL` stays (demo setup and the test suite both need it).

**New tests required.** One constructor/CLI test proving runtime operation
needs no admin DSN — see Phase 3 matrix row R4.

**Risk: Medium** — wide mechanical fan-out, most of it behind the Postgres gate.

---

### L2 — `VerifiedAnswerResult`: harden the survivor, then delete

**Evidence.** `grounding.py:135-139` declares it *"Successor to
`tenant_engine.py`'s `AnswerResult`, once PR 3/4 wire a real verifier into the
answer path."* PRs 3, 4 and 5 all landed (`8ff507a`, `d69484e`, `6316ed4`) —
and wired the verifier into **`AnswerResult`**, not into this type.
`answer()` returns `AnswerResult` at all five return points (1266, 1294, 1377,
1393, 1450). Zero producers, zero consumers outside its own tests.

**Why deletion cannot come first.** `VerifiedAnswerResult` is
`@dataclass(frozen=True)` with `__post_init__` validation and `grounded`
computed as a **property** — specifically so it can never diverge from
`grounding_status`. `AnswerResult` is a plain mutable `@dataclass` with no
`__post_init__` and no validation:

| Field | `AnswerResult` (survivor) | `VerifiedAnswerResult` (to delete) |
|---|---|---|
| `grounding_status` | `Optional[str]` — unvalidated (`tenant_engine.py:134`) | `GroundingStatus`, validated |
| `verifications` | `List[Any]`, correct type only in a **comment** (`:139`) | `List[ClaimVerification]` |
| `grounded` | **stored `bool`** (`:100`) — can disagree with status | **derived property** — cannot diverge |
| Mutability | mutable | frozen |

Deleting the dead type first would remove the only correctly-modelled version
of the contract and leave the loosely-modelled one as sole survivor — a net
regression in type safety disguised as a cleanup.

**Sequence (Phase 2 → Phase 4):**

1. **Phase 2 (hardening, behavior-preserving):**
   - **Retain and relocate both status sets.** `_GROUNDED_STATUSES` and
     `_VALID_GROUNDING_STATUSES` encode concepts the hardened survivor
     *needs*; they are not `VerifiedAnswerResult`'s private detail. Promote
     them into a shared `grounding.validate_grounding_status()` (and a
     `grounding.is_grounded_status()`) that both the hardened `AnswerResult`
     and any future verifier call.
   - Type `grounding_status` as `Optional[GroundingStatus]`.
   - Type `verifications` as `List[ClaimVerification]` (delete the comment).
   - Add `__post_init__` validating `grounding_status` via the shared function.
   - **Remove `grounded` as a stored dataclass field** (`tenant_engine.py:100`)
     and add it as a **derived property**, making the invariant structural
     rather than asserted.
   - **Update all five `AnswerResult(...)` production construction sites** —
     `tenant_engine.py:1265, 1293, 1376, 1393, 1450` — to stop passing
     `grounded=`. Verified: all five currently pass it, so this is not
     optional; skipping it raises `TypeError` on an unexpected keyword.
   - **Define the `None` case explicitly:** if `grounding_status=None` remains
     permitted, `grounded` is `False`. Decide deliberately whether `None`
     should remain permitted at all now that every production path sets it.
   - Confirm against `test_tenant_engine.py`'s 11 `grounded`/`abstain_reason`
     assertions — a derived property must keep every one of them passing.
2. **Phase 4 (deletion):** remove **only** `VerifiedAnswerResult` itself, both
   `__init__.py` entries (24, 56), the `benchmarks/grounding/README.md:29`
   mention, and the 3 type-only tests.

> ⚠️ **v2 said to delete `_GROUNDED_STATUSES` and `_VALID_GROUNDING_STATUSES`
> in Phase 4. That is self-contradictory** — Phase 2 hardening depends on
> exactly those two concepts. Deleting them would either break construction or
> silently remove the validation the hardening exists to preserve. They are
> **retained and relocated, never deleted.**

**Tests to delete (3 tests / 9 executed cases)** — they exercise only the dead
type's own accessors:
- `test_grounded_property_derives_from_grounding_status` (7 params)
- `test_verified_answer_result_rejects_invalid_grounding_status`
- `test_verification_for_looks_up_by_claim_id`

Their *intent* survives as new `AnswerResult` invariant tests (Phase 3, R1) —
this is a migration of coverage, not a loss, and the matrix records it as such.

**Tests that must remain** in the same file: `AnswerClaim` defaults, all four
`ClaimVerification` validation tests, both `validate_grounding_mode` tests,
both `Verifier`-protocol structural tests.

**Retain.** `GroundingStatus` — consumed by `evaluate_policy` and
`tenant_engine`. `AnswerClaim`, `ClaimVerification`, `Verdict`, `ReasonCode`,
`GroundingMode`, `Verifier`, `VerifierUnavailableError`.

**Risk: Low** once sequenced this way. **Medium if reversed.**

---

### L3 — Stale CI, config and documentation residue

| # | Location | False claim | Fix |
|---|---|---|---|
| a | `.github/workflows/ci.yml:29` | *"`test_database.py`/`test_tenancy.py` self-skip without `POSTGRES_URL`"* — `test_database.py` lost its `skipif` with `DatabaseManager` | Name `test_tenancy.py` only |
| b | `.github/workflows/ci.yml:63` | Postgres job runs `tests/test_database.py`, which needs no Postgres and already runs in `unit` | Drop it from the Postgres job |
| c | `docs/architecture.md:10-11` | *"The namespace-only engine is a compatibility layer"* — deleted | The secure tenant engine is the **only** engine; legacy support is **data import only**, not API compatibility |
| d | `docs/operations.md:33-35` | Tells operators to run a scale benchmark that is deleted | Remove; state secured-path scaling is **unmeasured** pending a replacement |
| e | `README.md:364` | `# privileged connection, used only for setup_secure()` — false | Folded into L1 |
| f | `tests/test_tenancy.py:7-8` | Contrasts with `test_database.py`'s throwaway-schema isolation, which no longer exists | Drop the clause |

**Risk: None.** Comments, docs, one CI argument.

---

### L6 — MCP's duplicate store construction (promoted into scope)

**Evidence.** `mcp_server.py:104-113` builds `SecureGraphStore` itself —
reading `rag._runtime_url`, computing `_vector_column_type(...)`, and
**assigning to `rag._secure_store`** — duplicating `core.py:136-150`'s
`for_tenant()` lazy init and reaching through the facade's private state to do
it. Its own comment admits it: *"Mirrors `for_tenant()`'s lazy store
initialization."*

**Why it is in scope now.** v1 of this plan recorded it as known-but-deferred,
then claimed "one canonical architecture" in its acceptance criteria. Both
cannot be true. Given the stated goal is zero duplicate implementations, and
the fix is small, fixing it is better than weakening the claim.

**A second, sharper defect this exposes.** `for_tenant()` validates the missing
`runtime_url` case (`core.py:141-145`, raising a specific `ValueError`). **The
MCP lifespan does not** — it passes `rag._runtime_url` straight into
`SecureGraphStore(...)`, so a misconfigured MCP server constructs a store
around `None` instead of failing with the actionable message. The duplication
is not just redundant; the two copies **behave differently on the error path**.

**Fix.** Add a facade-owned `PostgresGraphRAG._get_or_create_store()`
encapsulating the lazy init **and owning the missing-`runtime_url`
validation**, so every caller fails identically and with the same message.
Have both `for_tenant()` and the MCP lifespan call it. MCP stops touching
`_secure_store` and `_runtime_url`.

**Tests.** One test asserting `for_tenant()` and the MCP lifespan yield the
**same** store instance. `test_tenancy.py:2366` already exercises the shared
`_secure_store` side effect and must keep passing.

**Risk: Low-Medium** — touches MCP server startup. Covered by Gate 5 (MCP stdio
and HTTP startup) in Phase 6.

---

### L4 — Citation-only rule implemented three times *(DEFERRED — separate PR)*

One rule, three implementations:

1. `tenant_engine.py:1329` — production, canonical.
2. `verification.py:246-250` — `evaluate_policy`'s `citation_only` branch,
   **unreachable from production** (`answer()` returns from its self-contained
   `citation_only` branch before `evaluate_policy` is ever called; the only
   production call site at `:1408` sits in the `else` arm). Sole exerciser:
   `test_grounding_verification.py:320`.
3. `benchmarks/grounding/baseline_runner.py:32-40` — the frozen "before"
   comparator.

**Corrected approach — do not couple the instrument to what it measures.**
v1 proposed having `baseline_runner` import a shared production predicate. That
is wrong: the baseline is a *frozen measurement*, and if it imports the
predicate it measures, the baseline moves whenever production moves — hiding
precisely the drift the consolidation was meant to catch.

Instead:
- Extract a helper for production use if desired.
- **Keep `baseline_runner.citation_only_grounded()` independently implemented.**
- Add a **characterization test** asserting production and baseline agree over
  a fixed case matrix — detects divergence without coupling.
- If the semantics change intentionally, **version the baseline explicitly**.
- Leave `evaluate_policy`'s `citation_only` branch in place: it is exported
  public API and `GroundingMode` admits `citation_only`, so removing it would
  make a valid mode fall through to `verified` semantics. Add a comment noting
  `answer()` never reaches it.

**Why deferred.** This is a grounding refactor, not legacy deletion, and it is
the only item that touches live `answer()` control flow. It ships as its own PR
after L1–L3/L6, so a bisect lands on it unambiguously.

---

### L5 — `database.py` vestigial module *(DEFERRED OUTRIGHT)*

Post-removal the module is 70 lines of live leaf utilities — `normalize_entity`,
`content_hash`, `_vector_column_type`, `_as_float_list`, `_cosine_similarity`,
`_HNSW_*`, `MAX_HOPS_HARD_LIMIT`, `MAX_ROWS_PER_STATEMENT` — under a name that
no longer describes them. Two observations stand, both recorded for the future
cleanup:

1. Every symbol is re-exported through `tenancy.py:47-54`, so the codebase has
   two import paths for the same seven symbols (`tenant_engine.py:25` imports
   `content_hash` from `.tenancy`; `tests/test_database.py:10` from
   `.database`).
2. `tenancy.py:589`'s local import is annotated *"avoid cycle at module load."*
   AST analysis shows `database.py` imports only `hashlib`, `logging`, `math`,
   `re`, `typing` — **nothing from this package**. No cycle is possible; the
   comment is wrong.

**Deferred, and v1's proposed fix is withdrawn.** v1 recommended folding these
utilities into `tenancy.py`. That is rejected: `tenancy.py` is already 2,728
lines, and moving hashing, normalization, vector conversion and cosine
similarity into it would worsen the repo's largest monolith to fix a naming
problem.

`database.py` contains live utilities, not a parallel architecture. Renaming it
adds import churn without improving the safety of this removal.

**Tracked follow-up (not this change set):** rename to `_db_utils.py` and
repoint imports at it directly instead of relying on accidental re-exports
through `tenancy.py`. **Never** move the utilities into `tenancy.py`.

---

## 3. Impact on `IMPLEMENTATION_PLAN.md` and completed work

**No `IMPLEMENTATION_PLAN.md` exists in this repository.** Two artifacts carry
that role:

**a) `docs/reviews/demo-readiness-implementation-plan-2026-08-16.md`** — the
canonical in-repo equivalent (untracked, `??` in `git status`).

- §16 *"Code Removal: None proposed"* is **already superseded by its own §26**,
  which records the legacy-engine removal as a separate post-demo initiative.
  This plan continues §26; it does not contradict §16, which §26 scoped to the
  demo-day context.
- §4 *"No target architecture change is proposed"* — consistent. L1/L2 delete
  unreachable code; L6 removes a duplicate construction path without changing
  the graph's shape.
- No task (TASK-01…06a) or deferred item (TASK-D1…D9) touches `postgres_url`,
  `VerifiedAnswerResult`, or MCP store construction.
- Its §3 diagram describes `database.py` as *"leaf utilities + `DatabaseManager`
  class."* With L5 deferred, **update the `DatabaseManager` half only**.

**Verdict: no conflict.**

**b) `/Users/ayushmishra/Desktop/OpenSource copy/postgresql-graph-rag/IMPLEMENTATION_PLAN.md`**
— a **stale pre-removal snapshot outside this repository**. It still describes
`DatabaseManager` and `bench_scale.py` as live and lacks the §26/§27 addenda.
**Not authoritative; must not be used to validate this plan.**

**Completed work:** Release 2 PRs 1–6 are unaffected. L2 removes a type those
PRs declared but never wired in; L4 is deferred entirely.

---

## 4. Commit / PR structure

Change isolation is a merge requirement, not a preference.

| # | Commit / PR | Contents | Merge-blocking |
|---|---|---|---|
| **0a** | **Characterization tests** | G1–G14 written and proven green **against `HEAD`** (legacy still present), via a temporary worktree. **Must precede commit 0** | ✅ |
| **0** | Legacy engine deletion | The existing uncommitted removal, **on its own**, after 0a | ✅ |
| **0b** | Housekeeping | `docs/results/` deletion — **unrelated to architecture removal** | ❌ **independent, non-blocking PR.** It is either unrelated housekeeping or it is not; marking it blocking would re-couple the deletion this table exists to separate. May also simply be held for a later release |
| **1** | L1 — `postgres_url` | Parameter + 24 call sites + both CLI flags + README, **atomic** | ✅ |
| **2** | L3 — CI/docs repair | CI comment + job argument, architecture/operations/test docs | ✅ |
| **3** | L6 — MCP store construction | `_get_or_create_store()`; MCP stops touching private state | ✅ |
| **4a** | L2 — contract hardening | Shared validators, typed fields, derived `grounded`, 5 construction sites, **plus R1** | ✅ |
| **4b** | L2 — dead-type deletion | Delete `VerifiedAnswerResult` **only after 4a is green** | ✅ |
| **5** | Release metadata | `0.1.0` → `0.2.0`, `uv.lock`, CHANGELOG, ADR addenda | ✅ |
| **6** | L4 — grounding refactor | Characterization test + optional helper extraction | ❌ separate PR |
| **7** | L5 — `_db_utils.py` rename | Tracked follow-up | ❌ deferred |

**Precondition (Phase 1):** obtain **explicit owner confirmation** that "no
customers and no external installations" still holds. The repository asserts it
(ADR-003) but cannot prove it, and the entire premise for deleting rather than
deprecating rests on it.

---

## Phase 1 — Discovery & Dependency Verification

1. Record the baseline with Postgres reachable: `pytest -q` → **259 passed,
   0 skipped**. Any skip → stop and fix the environment.
2. `ruff check postgres_graph_rag tests benchmarks` → clean.
3. Capture pre-change benchmark output (needed by Gate 8):
   ```
   python -m benchmarks.grounding.baseline_runner      --split all > /tmp/baseline.before
   python -m benchmarks.grounding.deterministic_runner --split all > /tmp/determ.before
   ```
4. Re-confirm each finding with the checks used to derive it: `postgres_url`
   AST read-count = 0; `VerifiedAnswerResult` has no construction outside its
   test file; `evaluate_policy("citation_only")` has no production caller;
   `mcp_server.py:104-113` assigns `rag._secure_store`.
5. **Get the owner confirmation** described above, in writing.
6. **Characterization tests BEFORE the deletion commit.** The legacy removal is
   applied but **uncommitted**, so the pre-deletion tree is `HEAD`. Therefore:

   ```
   git worktree add ../pgr-head HEAD     # legacy code still present here
   # write G1–G14 against ../pgr-head
   # prove they pass THERE, against the pre-deletion behavior
   # commit the tests (commit 0a) — tests only, no deletion
   git worktree remove ../pgr-head
   ```

   Only then commit the legacy implementation/test deletion (commit 0), and the
   `docs/results/` housekeeping (0b) separately.

   **Order: 0a tests → 0 deletion → 0b housekeeping.** Writing the
   characterization tests after the deletion would codify whatever survives
   rather than what the deleted tests actually protected — which is exactly how
   the coverage hole this matrix exists to close was created.
7. `rm -rf dist/` — remove the stale artifacts before any build happens.
8. **Local workspace hygiene** (developer machine only — nothing here is
   committed, and none of it changes repository contents). Measured sizes:

   | Path | Size | Action |
   |---|---|---|
   | `/OpenSource/.venv` | **410 MB** | **Delete** — redundant parent-directory environment; the repo has its own |
   | `postgresql-graph-rag/.venv` | 199 MB | **Retain**, or recreate with `uv sync --extra dev --extra mcp` |
   | `/OpenSource/.pytest_cache` | 16 KB | Delete |
   | `postgresql-graph-rag/.pytest_cache` | 48 KB | Delete |
   | `postgresql-graph-rag/.ruff_cache` | 20 KB | Delete |
   | `__pycache__/`, `*.pyc`, `.DS_Store` | — | Delete |
   | `postgresql-graph-rag/dist` | 416 KB | Delete (step 7) |
   | `postgresql-graph-rag/.env` | — | **Retain** — the test suite loads `POSTGRES_URL` from it via `load_dotenv()`; deleting it turns Gate 1 into a silent 83-test skip |
   | `postgresql-graph-rag/.firecrawl` | 700 KB | Retain unless its cached research is no longer useful |

   Then recreate the environment with `uv sync` and **re-run step 1** — a fresh
   environment invalidates the baseline captured before it.

**Exit:** baseline recorded, findings re-confirmed, owner confirmation obtained,
base committed, `dist/` clean, workspace clean.

---

## Phase 2 — Consolidation / Migration

Behavior-preserving work that must land *before* anything is deleted.

1. **L2 hardening** — promote `_GROUNDED_STATUSES`/`_VALID_GROUNDING_STATUSES`
   into shared `validate_grounding_status()` / `is_grounded_status()`; type
   `grounding_status` as `Optional[GroundingStatus]` and `verifications` as
   `List[ClaimVerification]`; add `__post_init__` validation; **remove
   `grounded` as a field and add it as a derived property**; update all five
   `AnswerResult(...)` sites to stop passing `grounded=`; decide the `None`
   semantics.
2. **L6** — add `PostgresGraphRAG._get_or_create_store()` **owning the
   missing-`runtime_url` validation**; repoint `for_tenant()` and the MCP
   lifespan at it; remove MCP's private-state access.
3. **R1** (hardened-contract invariants) and **R3** (shared store construction)
   land with their respective changes above.

> **G1–G14 are NOT written here — they are written in Phase 1 step 6**, against
> `HEAD` in a temporary worktree where the legacy implementation still exists,
> and committed as commit 0a before the deletion. By the time Phase 2 runs they
> must already be green. v3 placed them in this phase, which contradicted its
> own "green against the pre-deletion tree" requirement, because Phase 1 had
> already committed the deletion.

**Exit:** baseline + G1–G14 + R1 + R3 pass, **0 skipped**. No behavior changed,
no symbol deleted.

---

## Phase 3 — Test Cleanup (coverage-equivalence matrix)

Every test removed by this change set **and** by the already-committed engine
removal must be classified. Do not recreate tests for deleted implementations;
do not drop product behavior silently.

**Classification:** `LEGACY-ONLY` (delete) · `COVERED` (link the secure-path
replacement) · `GAP` (write one focused secure-path test).

> 🚨 **v2 classified whole files in single rows and marked
> `test_database.py`'s tests `LEGACY-ONLY`. That was wrong, and the error was
> substantive, not clerical.** Those tests covered behaviors that are **still
> live, still shipped, and default-on in the secure path** — verified against
> `SecureGraphStore.traverse_graph()` / `find_paths()` signatures
> (`tenancy.py:2170`, `:2616`) and `RetrievalConfig` defaults
> (`models.py:36-63`). Deleting them as "legacy" would have silently created a
> coverage hole in live retrieval behavior. Classification is **per test**.

**Deleted `tests/test_database.py` — 17 tests (15 engine + 2 already survived)**

| # | Deleted test | Secure-path support | Class | Disposition |
|---|---|---|---|---|
| D1 | `test_setup_rejects_invalid_dimension` | `migrate_schema(embedding_dimension=...)` | **GAP** | → **G12**. v3 wrongly mapped this to `test_migrate_schema_rejects_embedding_dimension_mismatch` (`:1917`), which tests a **different concern**: re-migrating an *applied* schema at `DIM+1`. The deleted test asserted `ValueError` on `-1` and on the injection-shaped `"8; DROP TABLE graph_nodes;"` — **input validation before connecting**. Security-adjacent; nothing covers it |
| D2 | `test_bulk_upsert_single_round_trip_and_idempotent` | `resolve_and_upsert_nodes` | **GAP** | → **G7** |
| D3 | `test_edge_weight_increments_on_repeated_mention` | `upsert_edges` | **GAP** | → **G8** |
| D4 | `test_directed_traversal_respects_edge_direction` | `directed=` param, both APIs | **GAP** | → **G6** (4 incidental uses exist; direction *semantics* unasserted) |
| D5 | `test_relation_type_allow_and_deny_lists` | `relation_types` / `exclude_relation_types`, both APIs | **GAP** | → **G1** — **zero occurrences in the entire suite** |
| D6 | `test_max_hops_hard_limit_rejected` | 3 raise sites (`tenancy.py:2193, 2529, 2637`) | **GAP** | → **G5** — **zero test references to the limit** |
| D7 | `test_max_neighbors_per_node_caps_fanout` | `max_neighbors_per_node`, default **20** | **GAP** | → **G2** — **zero occurrences** |
| D8 | `test_min_weight_filters_weak_edges` | `min_weight`, both APIs | **GAP** | → **G3** (1 incidental use, in an unrelated test) |
| D9 | `test_hop_distance_and_score_decay` | `score_decay`, default **0.7** | **GAP** | → **G4** — **zero occurrences** |
| D10 | `test_namespace_isolation_in_traversal_and_vector_search` | namespace is a documented partition-within-tenant | **GAP** | → **G9** — **zero namespace-isolation tests**; existing coverage is *tenant* isolation, a different boundary |
| D11 | `test_entity_resolution_exact_normalization_merges` | `resolve_and_upsert_nodes` | **GAP** | → **G10** |
| D12 | `test_exact_match_resolution_merges_new_metadata` | metadata merge on exact match | **GAP** | → **G10** |
| D13 | `test_entity_resolution_fuzzy_merges_true_variants` | trigram + embedding confirmation | **GAP** | → **G11** (only the *negative* case `test_numbered_identifiers_are_never_fuzzy_merged` exists) |
| D14 | `test_entity_resolution_does_not_merge_unrelated_entities` | trigram + embedding confirmation | **GAP** | → **G13**. v3 wrongly mapped this to `test_numbered_identifiers_are_never_fuzzy_merged` (`:520`), which enforces a **different rule** (digit-bearing identifiers require exact normalized equality). The deleted test is the John Smith / John Smyth case: high trigram similarity, dissimilar embeddings, `fuzzy=True` — embedding confirmation must veto the merge. Uncovered |
| D15 | `test_chunk_hash_idempotency_tracking` | mechanism (`filter_new_chunks`/`mark_chunks_ingested`) **removed with the engine** | **COVERED** | Secure document idempotency is covered by `test_concurrent_identical_ingestion_no_duplicate_chunks` (`:1270`), `test_embedding_failure_does_not_commit_hash_and_retry_recovers` (`:1107`), `test_chunk_replacement_failure_rolls_back_hash_too` (`:1221`), `test_add_record_is_idempotent_on_unchanged_record` (`:1994`). No new test needed |
| D16-17 | `test_normalize_entity_*`, `test_content_hash_*` | live leaf utilities | **RETAINED** | Already survive in current `tests/test_database.py` |

**Deleted `tests/test_core.py` — 11 tests**

| # | Deleted test(s) | Class | Disposition |
|---|---|---|---|
| C1 | `test_setup_emits_deprecation_warning`, `test_add_texts_…`, `test_query_structured_…`, `test_query_…`, `test_for_tenant_does_not_emit_legacy_deprecation_warning` (5) | **LEGACY-ONLY** | Delete — they assert warnings on methods that no longer exist |
| C2 | `test_ingest`, `test_ingest_skips_already_ingested_chunks`, `test_ingest_does_not_skip_when_metadata_provided`, `test_ingest_skips_failed_chunk_without_aborting_batch` (4) | COVERED | `test_tenant_engine.py` `add_document`/`add_document_detailed`; retry-hole regression tests `test_tenancy.py:1108-1545` |
| C3 | `test_query`, `test_query_structured_sorts_by_score_and_passes_filters` (2) | COVERED | `test_tenant_engine.py` retrieve tests + `test_filters.py` (82 lines) + `test_tenancy.py:1722, 2222` |

**Deleted `tests/test_scenarios.py` — 5 tests**

| # | Deleted test | Class | Disposition |
|---|---|---|---|
| S1 | `test_entity_resolution_across_chunks` | **GAP** | → **G10** |
| S2 | `test_deep_path_traversal` | COVERED | `test_find_path_reconstructs_exact_chain_and_ignores_decoy_branch`, `test_find_paths_returns_ranked_paths_with_edge_evidence` |
| S3 | `test_strict_namespace_isolation` | **GAP** | → **G9** |
| S4 | `test_metadata_integrity_via_jsonb_merge` | **GAP** | → **G10** |
| S5 | `test_bidirectional_context` | **GAP** | → **G6** |

**Deleted `tests/test_integration.py` — 2 tests** (`TestIntegration.test_openai_e2e`,
`test_google_e2e`) → **GAP**, both credentialed → **R5 (release gate)**.

**`benchmarks/bench_scale.py`** → **GAP** → **R6 (claims gate)**.

**New merge-blocking tests**

| Row | Test | Origin |
|---|---|---|
| **R1** | `AnswerResult` invariants — status validation, derived `grounded`, `verifications` typing | L2; absorbs the 3 deleted `VerifiedAnswerResult` tests |
| **R2** | Legacy-symbol absence in the **installed wheel** — `DatabaseManager`, legacy facade methods, `VerifiedAnswerResult` | Engine removal + L2. **Packaging/CI assertion, not a source-tree unit test** — a test inside `tests/` cannot meaningfully assert about an installed artifact |
| **R3** | `for_tenant()` and MCP lifespan return the **same** store instance; both raise the **same** error when `runtime_url` is missing | L6 |
| **R4** | MCP and evaluation construct and start with **no admin DSN** — a **bounded smoke test or mocked-CLI test**, not `--help` | L1 |
| **G1** | `relation_types` / `exclude_relation_types` allow + deny on `traverse_graph` and `find_paths` | D5 |
| **G2** | `max_neighbors_per_node` caps fan-out | D7 |
| **G3** | `min_weight` filters weak edges | D8 |
| **G4** | `score_decay` applies per hop distance | D9 |
| **G5** | `max_hops > MAX_HOPS_HARD_LIMIT` rejected at all three raise sites | D6 |
| **G6** | Directed vs undirected traversal semantics | D4, S5 |
| **G7** | Batch node upsert is idempotent on repeat (**scope reduced** — chunk-hash dedup dropped; see D15) | D2 |
| **G8** | Edge weight increments on repeated mention | D3 |
| **G9** | Namespace isolation *within one tenant* for traversal and vector search | D10, S3 |
| **G10** | Automatic entity resolution: exact-normalization merge, metadata merge, cross-chunk resolution | D11, D12, S1, S4 |
| **G11** | Fuzzy resolution merges true variants (positive case) | D13 |
| **G12** | `migrate_schema` rejects invalid `embedding_dimension` (`-1`, non-integer, injection-shaped) **before connecting** | D1 |
| **G13** | Fuzzy resolution does **not** merge trigram-similar names with dissimilar embeddings (John Smith / John Smyth) | D14 |
| **G14** | **Facade forwarding** — `TenantGraphRAG.retrieve()` passes `directed`, `relation_types`, `exclude_relation_types`, `min_weight`, `score_decay`, `max_neighbors_per_node` and `hops` through to `SecureGraphStore.traverse_graph()` | C3 |

> **G1–G14 are merge-blocking.** Deterministic, no credentials, running against
> the existing Postgres fixture. They are not "new coverage" — they restore
> coverage of behavior that already shipped and would otherwise be silently
> lost. **`score_decay=0.7` and `max_neighbors_per_node=20` are live defaults on
> every retrieval call and currently have zero tests.**
>
> **G-rows are test *groups*, not single functions.** G1 alone spans
> `traverse_graph` and `find_paths` × allow and deny; several others cover
> multiple APIs. Implement as however many focused tests the row needs.
>
> **Why G14 matters independently of G1–G6.** Verified: `tenant_engine.py:1113-1126`
> forwards all seven values into `traverse_graph`, but the only forwarding
> assertion in the suite is `test_tenant_engine.py:70`, which checks
> `seed_scores` **only**. G1–G6 prove *storage* honors these parameters; without
> G14, storage could keep working perfectly while the public retrieval path
> silently stops passing them and **no test fails**. The existing mock-based
> `store.traverse_graph.call_args.kwargs` pattern makes this cheap.
>
> **On S1–S5:** these were `live_provider`-marked (`test_scenarios.py:17`
> `pytestmark`). Mapping their behaviors onto deterministic store-level tests is
> a deliberate **upgrade** — the behavior gets credential-free coverage. Their
> real-provider E2E aspect is separately tracked as R5.

**Deferred, tracked, non-merge-blocking**

| Row | Test | Gate |
|---|---|---|
| **R5** | One OpenAI + one Gemini secured-path E2E smoke flow | **Release gate** — blocks publishing `0.2.0` |
| **R6** | `SecureGraphStore`/`TenantGraphRAG` scale benchmark | **Claims gate** — blocks publishing performance/scalability claims; until then docs must state secured-path scaling is unmeasured |

**Explicitly retained:**
- `test_tenancy.py::test_legacy_data_migration_preserves_ids` — protects the
  live `migrate_legacy_data=True` feature; already reworked to seed via raw
  SQL. Its **name** says legacy; the feature is not.
- All 21 `test_tenancy.py` sites passing `postgres_url` — mechanical edits.
- All `evaluate_policy` tests, including `citation_only` at `:320` (sole guard
  on that branch).
- Every remaining `test_grounding_contract.py` test.

**Exit:** matrix complete; every delta against the 259 baseline accounted for;
**zero unexplained coverage loss**.

---

## Phase 4 — Legacy Deletion

Per the Section 4 commit table.

1. **L3** (commit 2) — CI comment, CI job argument, `docs/architecture.md`,
   `docs/operations.md`, `test_tenancy.py` docstring.
2. **L1** (commit 1) — **atomic**: parameter, 24 call-site arguments,
   `--postgres-url` flag, `evaluation.py`'s admin-URL requirement, 5 README
   examples. One commit; verify only after the whole thing.
3. **L2 deletion** (commit 4b, only after commit 4a's hardened contract is
   green) — `VerifiedAnswerResult`, its two `__init__.py` entries,
   `benchmarks/grounding/README.md:29`, and the 3 type-only tests.

   > **`_GROUNDED_STATUSES` and `_VALID_GROUNDING_STATUSES` are NOT deleted.**
   > They are retained and relocated into the shared validators in Phase 2 —
   > the hardened `AnswerResult` depends on them. v3 still listed them here;
   > that contradicted its own L2 section.

**Exit:** each commit independently green, 0 skipped.

---

## Phase 5 — Dependency & Configuration Cleanup

1. **Version → `0.2.0`.** `0.1.0` is authoritative (`pyproject.toml` + tag
   `v0.1.0`). The `0.7.0` in `uv.lock` has been inconsistent since the initial
   commit, never had a tag or matching project metadata — **stale generated
   metadata, not release history**. Update `pyproject.toml` **and** `uv.lock`
   together. Pre-1.0 breaking removal → minor bump.
2. **Annotate ADR-002** that "v0.7 compatibility" referred to a planned
   timeline, **not an actually released package version**.
3. **Extend ADR-003** with an addendum rather than writing ADR-004 — this is the
   same decision applied to residue the first pass missed. Record that the
   `postgres_url` removal closes a path by which an admin DSN reached processes
   that should hold only the runtime role.
4. **`CHANGELOG.md`** — one `0.2.0` entry covering L1/L2/L3/L6: what was removed,
   what was retained (`migrate_legacy_data`, `LEGACY_TENANT_ID`), and why the
   ADR-003 premise extends here. Record R5/R6 as known gaps. **Preserve
   historical entries and dated review references as history, clearly labeled —
   do not edit them.**
5. **`pyproject.toml` dependencies — no change expected.** All six runtime
   dependencies remain in use; no entry point or pytest marker is removed.
   Confirm rather than assume.
6. **`.env.example`** — **retain `POSTGRES_URL`**; demo setup and the test suite
   both need it. Do not remove it merely because two CLIs stopped demanding it.
7. `.github/workflows/ci.yml` — L3(a)(b).
8. `docs/reviews/demo-readiness-implementation-plan-2026-08-16.md` §3 — update
   the `DatabaseManager` half of the `database.py` line.
9. **No migrations affected.** Nothing touches the schema, `SCHEMA_VERSION`, or
   `migrate_schema()`.

---

## Phase 6 — Full Regression Validation

### Merge gates — correctness and packaging (all must pass)

| # | Gate | Pass criterion |
|---|---|---|
| 1 | Full suite, Postgres reachable | **0 skipped**, and every delta from 259 explained by the Phase 3 matrix. A green run with skips is a **failed gate** |
| 2 | `ruff check postgres_graph_rag tests benchmarks` | clean |
| 3 | `compileall` + `git diff --check` | clean |
| 4 | Deterministic secured-path demo E2E | `demo setup` → `ingest` → `query --mode hybrid_graph` yields the documented reasoning path |
| 5 | Entry points | `demo`, `evaluation`, **MCP stdio**, **MCP HTTP** each reach a **bounded, observable ready state** (serve one request, or a mocked-CLI test asserting the constructed config) and exit cleanly |
| 6 | **No admin DSN required (proves L1)** | MCP and evaluation reach that same ready state with only `PGR_RUNTIME_URL` set and `POSTGRES_URL` **unset**. **`--help` is NOT acceptable evidence** — verified: it returns rc=0 today with both variables unset, because `argparse` short-circuits before the credential check, so it would pass even with the bug present |
| 7 | Legacy-data migration | Raw-SQL migration test passes; verify migrated **IDs, edges, weights, and tenant isolation** |
| 8 | Benchmark parity | `baseline_runner`/`deterministic_runner --split all` **byte-identical** to `/tmp/*.before` |
| 9 | **Clean-clone build** | `rm -rf dist/`, fresh checkout, `uv build` → wheel + sdist |
| 10 | **Installed-wheel validation** | Install the fresh wheel into an **isolated venv** and exercise it — not just source imports |
| 11 | **Artifact absence assertions** | In the **installed artifact**: no `DatabaseManager`, no legacy facade methods, no `VerifiedAnswerResult`, no legacy exports |
| 12a | Executable-code assertions | Removed symbols absent from **executable code** — `postgres_graph_rag/`, `tests/`, `benchmarks/`, `.github/` — asserted per symbol, not by prose grep |
| 12b | Public-surface assertions | `__all__` and importability checked **against the installed wheel** (pairs with R2) |
| 12c | Documentation link check | Every doc path/anchor resolves — catches genuinely broken references |
| 12d | Historical allowlist | A small **explicit allowlist** of files permitted to mention removed symbols as history: `CHANGELOG.md`, `docs/decisions/002-*`, `docs/decisions/003-*` (incl. the new `postgres_url` addendum), `docs/reviews/*`, and `README.md`'s "`bench_scale.py` previously existed" note. Anything outside the allowlist is a failure |
| 13 | CI | `unit` and `postgres` jobs green on a pushed branch |
| 14 | Version metadata (merge time) | `pyproject.toml`, `uv.lock` and CHANGELOG all read `0.2.0`. **The Git tag is a release-time gate, not a merge gate** — it does not exist yet at merge |

> 🚨 **v2's Gate 12 was a single broad grep expecting matches only in
> `CHANGELOG.md`. It would fail by design.** Verified: that pattern matches six
> files with entirely legitimate historical references — `docs/architecture.md`,
> `docs/decisions/002-secure-api-primary.md`,
> `docs/decisions/003-remove-legacy-engine.md`, `README.md`, and both
> `docs/reviews/*` files — and `CHANGELOG.md` was not even inside the scope it
> searched. Replaced by 12a–12d above.

### Release gate — external-provider verification

| # | Gate | Blocks |
|---|---|---|
| R5 | One OpenAI + one Gemini secured-path E2E smoke flow | **Publishing the `0.2.0` package.** Deterministic/mocked tests already validate the code paths; this validates real API compatibility |

### Claims gate — performance evidence

| # | Gate | Blocks |
|---|---|---|
| R6 | Secured-path scale benchmark | **Publishing any performance or scalability claim.** Until it exists, documentation must state secured-path scaling is unmeasured |

### Release and rollback

Tag the pre-removal revision. This deletion **does not alter the secure database
schema**, so rollback is a code/package rollback, not a database rollback.
**Publish only CI-built artifacts. Never upload local `dist/` contents** — the
current ones contain the complete legacy engine.

---

## Phase 7 — Architecture Re-verification

1. **Re-derive the import graph from source**, including function-local imports
   (`core.py:138`, `mcp_server.py:109`, `tenancy.py:589`,
   `tenant_engine.py:1258`). A top-of-file scan misses all four and produces a
   wrong graph.
2. **One construction path:** every entry point reaches storage through
   `PostgresGraphRAG._get_or_create_store()` → `SecureGraphStore` →
   `TenantGraphRAG`. With L6 fixed, no component touches facade private state.
3. **One result type** on the answer path: `AnswerResult`, now validated.
4. **Zero unused public exports:** every name in `__init__.__all__` has a
   producer or a documented consumer.
5. Update `docs/architecture.md` to the post-removal shape; record the ADR-002
   and ADR-003 addenda.
6. **Record the two knowingly-open items** — L4's triple implementation and
   L5's module naming — as tracked follow-ups, so the next audit does not
   rediscover them as findings.

---

## Final acceptance

| Requirement | Status after this change set |
|---|---|
| One canonical architecture | ✅ **Achieved, not claimed-around.** Single engine, single validated result type, single store-construction path (L6 fixed rather than deferred) |
| Zero unnecessary duplicate implementations | ⚠️ **One knowingly retained:** the citation-only rule's three implementations. Deliberate — the frozen benchmark must stay independent of what it measures — and handled in a separate PR with a characterization test. **Recorded, not overlooked** |
| Zero unnecessary legacy tests | ✅ **Exact accounting:** the four deleted files held 35 test functions; 2 utility tests survived, so **33 were removed — 26 routine and 7 credentialed `live_provider`** (5 in `test_scenarios.py`, 2 in `test_integration.py`). Plus 3 `VerifiedAnswerResult` tests planned for removal here. Each is classified **individually**; 5 deprecation-warning tests are genuinely `LEGACY-ONLY`; 14 groups restored as G1–G14 because the behavior is still live. Nothing kept for coverage optics — and nothing dropped by mistaking a deleted *implementation* for a deleted *behavior* |
| No broken references | ✅ Gates 11, 12, 13 |
| No broken current functionality | ✅ Gates 1, 4–8, with the skip-count trap an explicit failure condition |
| No conflict with `IMPLEMENTATION_PLAN.md` | ✅ §3 — the only conflicting section is superseded by its own §26; the sibling-directory copy is flagged non-authoritative |
| Removal, not deprecation | ✅ L1/L2/L3 are deletions. L4 is a deferred refactor; L5 is deferred with its v1 fix withdrawn. Neither is a deprecation of anything |

---

## Appendix A — v3 → v4 correction log

| # | v3 said | Verified problem | v4 |
|---|---|---|---|
| 1 | G1–G14 written in Phase 2; Phase 1 step 6 commits the deletion first | **Executable contradiction.** v3 required them green "against the pre-deletion tree" that its own earlier step had already destroyed | **Phase 1 step 6:** worktree from `HEAD`, prove, commit as **0a**, *then* commit the deletion |
| 2 | Phase 4 deletes `_GROUNDED_STATUSES` + `_VALID_GROUNDING_STATUSES` | Contradicted v3's own L2 section, which retains them | **Removed from the deletion list**; only `VerifiedAnswerResult` is deleted |
| 3 | D1 COVERED by `test_migrate_schema_rejects_embedding_dimension_mismatch` | Different concern — that test re-migrates an *applied* schema at `DIM+1`; the deleted one rejected `-1` and `"8; DROP TABLE graph_nodes;"` **before connecting** | **GAP → G12** (security-adjacent) |
| 4 | D14 COVERED by `test_numbered_identifiers_are_never_fuzzy_merged` | Different rule — that enforces exact equality for digit-bearing identifiers; the deleted one is John Smith/John Smyth, where embedding confirmation must veto a trigram-similar merge | **GAP → G13** |
| 5 | No facade-forwarding coverage | Verified: `tenant_engine.py:1113-1126` forwards 7 values; the only forwarding assertion (`test_tenant_engine.py:70`) checks `seed_scores` alone. Storage could work while retrieval silently stops passing them | **G14 added** |
| 6 | D15 GAP → G7 | The `filter_new_chunks`/`mark_chunks_ingested` mechanism died with the engine; secure idempotency is covered 4× | **COVERED**; G7 rescoped to batch node-upsert idempotency (D2) only |
| 7 | R1–R6 tables appeared twice in Phase 3 | Editing artifact | **Duplicate removed** |
| 8 | "35 removed tests" | 35 functions existed; 2 survived | **33 removed — 26 routine + 7 credentialed `live_provider`**, verified via `pytestmark` at `test_scenarios.py:17` and the `test_integration.py` class marker |
| 9 | G-rows implied one test each | G1 alone spans 2 APIs × allow/deny | **Explicitly test *groups*** |
| 10 | L2 as one commit | Hardening and deletion have different risk profiles | **Split into 4a / 4b** |

## Appendix B — v2 → v3 correction log

| # | v2 said | Verified problem | v3 |
|---|---|---|---|
| 1 | Phase 3 matrix classified whole files; `test_database.py`'s tests were `LEGACY-ONLY` | **Wrong, and substantive.** `traverse_graph()`/`find_paths()` still accept `relation_types`, `exclude_relation_types`, `min_weight`, `score_decay`, `max_neighbors_per_node`; `score_decay=0.7` and `max_neighbors_per_node=20` are live defaults. The suite has **zero** occurrences of `relation_types`, `max_neighbors`, `score_decay`, or the `max_hops` hard limit, and **zero** namespace-isolation tests | **Per-test matrix (35 tests); 11 GAPs restored as merge-blocking G1–G11** |
| 2 | Phase 4 deletes `_GROUNDED_STATUSES` + `_VALID_GROUNDING_STATUSES` | Self-contradictory — Phase 2 hardening depends on both | **Retained and relocated** into shared validators; deletion limited to the dead class |
| 3 | Gate 12: one broad grep expecting matches only in `CHANGELOG.md` | **Fails by design.** Matches 6 files with legitimate history; `CHANGELOG.md` was not even in the searched scope | **Split into 12a–12d** with an explicit historical allowlist |
| 4 | Commit 0b (`docs/results/`) marked merge-blocking | Re-couples the very deletion the table separates | **Non-blocking independent PR** |
| 5 | No workspace hygiene | 410 MB redundant parent `.venv`, two `.pytest_cache` dirs, stale caches | **Phase 1 step 8**, with `.env` explicitly retained |
| 6 | Gates 5/6 satisfied by process start (v1 named `--help`) | **Verified:** `--help` returns rc=0 with both DSNs unset — passes even with the bug | **Bounded smoke / mocked-CLI test**; `--help` explicitly disallowed |
| 7 | Gate 14 included the Git tag | Tag does not exist at merge time | **Metadata at merge; tag at release** |
| 8 | L6 fix described as deduplication only | `for_tenant()` validates missing `runtime_url`; the MCP lifespan does not — the copies diverge on the error path | **`_get_or_create_store()` owns the validation** |
| 9 | R2 written as an ordinary test | A `tests/` unit test cannot assert about an installed artifact | **Packaging/CI assertion** |
| 10 | v2 did not say where G-tests run | Characterization tests written after deletion codify the survivor, not the intent | **G1–G11 land in Phase 2, green pre-deletion** |
| 11 | Status: "no changes implemented" | The prerequisite removal exists uncommitted | **Status corrected** |

## Appendix C — v1 → v2 correction log

Recorded so the reasoning is auditable rather than silently overwritten.

| # | v1 said | Why it was wrong | v2 |
|---|---|---|---|
| 1 | L1 in two commits: call sites, then parameter | **Impossible.** The intermediate raises `TypeError: missing required argument: postgres_url`. v1 also asserted the reverse order was the breaking one — both break | **Atomic single commit** |
| 2 | `baseline_runner` should import a shared production predicate | Couples a frozen measuring instrument to what it measures; the baseline would move with production and hide drift | **Keep independent + characterization test**; deferred to its own PR |
| 3 | Fold `database.py` utilities into `tenancy.py` | `tenancy.py` is already 2,728 lines; this worsens the largest monolith to fix a naming problem | **Deferred outright**; future rename to `_db_utils.py`, never a fold |
| 4 | Delete `VerifiedAnswerResult`; "accept the validation loss explicitly" as an option | It is the *stricter* contract (frozen, validated, derived `grounded`). Deleting it first leaves the unvalidated `AnswerResult` as sole survivor | **Harden `AnswerResult` first**, then delete |
| 5 | MCP duplicate construction out of scope, yet claimed "one canonical architecture" | Internal contradiction | **L6 promoted into scope and fixed** |
| 6 | No `dist/`, version, installed-wheel, or commit-split gates | Missed. Both local artifacts still contain the full legacy engine | **Gates 9–11, 14; Section 4 commit table; `rm -rf dist/` in Phase 1** |
| 7 | Test cleanup listed retained tests only | Never mapped *removed* tests to replacements | **Phase 3 coverage-equivalence matrix** |
