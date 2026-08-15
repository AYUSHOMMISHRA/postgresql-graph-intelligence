# Grounding benchmark — before/after: citation-only vs. DeterministicVerifier (PR 3)

Run date: 2026-08-15. Dataset: `grounding-benchmark-v1-candidate` (still not
independently human-reviewed — see `benchmarks/grounding/LABELING.md`).
Compares the citation-only validator currently live in `tenant_engine.py`
against `postgres_graph_rag.verification.DeterministicVerifier` — the
non-model deterministic layers from Release 2 PR 3. No LLM call is
involved in either side of this comparison; PR 4 (batched model
entailment) is what closes the remaining gap below.

## Sealed split (28 cases)

| Gate | Citation-only (before) | DeterministicVerifier (after) | Threshold |
|---|---:|---:|---:|
| Unsupported-claim escape rate | 81.8% FAIL | **0.0% PASS** | ≤ 5% |
| Contradicted-claim escape rate | 100.0% FAIL | **0.0% PASS** | ≤ 2% |
| Supported-claim retention rate | 100.0% PASS | **66.7% FAIL** | ≥ 85% |
| Incorrect-abstention rate | 0.0% PASS | **33.3% FAIL** | ≤ 15% |
| Unknown-citation rejection rate | 100.0% PASS | 100.0% PASS | ≥ 100% |
| Fabricated-quote rejection rate | N/A | N/A | ≥ 100% |
| Verifier-failure safely represented | N/A | N/A | ≥ 100% |
| Human/verifier agreement rate | N/A | N/A | ≥ 90% |

## What this shows

**The trust problem this whole effort exists to fix is closed at the
deterministic layer alone, with no model call.** Every category that
escaped as "grounded" under citation-only checking (reversed
relationships, wrong predicates, wrong numbers/dates, unsupported causal
claims, conflicting documents, valid-but-irrelevant citations,
pending/partial-extraction evidence, two-hop and partially-supported
compound claims) now correctly returns something other than "supported" —
escape rate 100% → 0% for contradicted claims specifically, 81.8% → 0%
overall.

**That trust improvement was not free — it cost recall, and the trade
shows up exactly where the design says it should.** Supported-claim
retention dropped from 100% to 66.7% (2 of 3 "supported" categories, or
20/30 instances across the full 140-case dataset). Breaking down *which*
supported claims got dropped and why is the useful part:

- `explicit_supported_relationship` — retained. Direct literal match.
- `incorrect_abstention_pressure` — retained. The separator-normalized
  quote-matching fallback (see `_locate_quote` in `verification.py`)
  handles the hyphen-vs-space formatting difference correctly.
- `multi_source_supported_claim` — **dropped, and expected to be.** A
  compound claim spanning two citations (e.g. "X depends on Y, which is
  owned by Team Z") doesn't appear verbatim (or normalized) in either
  single cited chunk, so the deterministic layer correctly can't confirm
  it end-to-end — recovering this specific pattern requires actual
  semantic understanding across citations, which is precisely PR 4's job,
  not a gap in this layer's logic.

This is the intended shape of PR 3's result, not a surprise: the plan
explicitly scoped this layer to never guess "supported" from a weak
signal, accepting a documented, bounded recall cost in exchange for
closing 100% of the contradicted-escape gate. A verifier that hit
one gate table's numbers immediately, with no visible trade-off anywhere,
would be more suspicious than this result, not less.

## Per-category escape rate, before vs. after

| Category | Citation-only | DeterministicVerifier |
|---|---:|---:|
| conflicting_documents | 100% | **0%** |
| evidence_requiring_two_hops | 100% | **0%** |
| multi_source_partially_supported_claim | 100% | **0%** |
| pending_or_partial_extraction | 100% | **0%** |
| reversed_relationship | 100% | **0%** |
| same_entities_different_predicate | 100% | **0%** |
| unsupported_causal_statement | 100% | **0%** |
| unsupported_number_or_date | 100% | **0%** |
| valid_but_irrelevant_citation | 100% | **0%** |
| correct_abstention | 0% | 0% |
| superseded_document_evidence | 0% | 0% |

## Limitations (in addition to the ones in the citation-only baseline doc)

- The three gates still N/A require, respectively: a model that can
  fabricate a quote to reject (PR 4), a verifier with a failure mode of
  its own to represent safely (deterministic pure functions don't have
  one), and completed independent human review of the sealed split.
- `supported_claim_retention_rate` failing here is not this PR's bug to
  fix — closing it without a model would mean guessing at entailment from
  weak signals, which is the exact failure mode citation-only checking
  already has. PR 4's job is recovering this recall *without*
  reintroducing that risk.

## Reproduce

```bash
python -m benchmarks.grounding.baseline_runner --split sealed       # before
python -m benchmarks.grounding.deterministic_runner --split sealed  # after
```
