# Grounding benchmark — baseline result (citation-only validator)

Run date: 2026-08-15. Dataset: `grounding-benchmark-v1-candidate`, 140 cases
across 14 categories (see `benchmarks/grounding/`). **This dataset is a
review candidate, not yet independently human-reviewed** — see
`benchmarks/grounding/LABELING.md` and `benchmarks/grounding/README.md` for
what that means and what's still required (`reviewer_packets.py` +
`reconcile_reviews.py`) before this is a real gating benchmark. This result
establishes the "before" number Release 2's verifier is measured against;
it is not itself a release gate.

Measures the check currently live in `tenant_engine.py`: a citation is
"grounded" iff its marker names a chunk actually present in the retrieved
evidence pool — no check of whether that chunk's text supports the claim.

## Sealed split (28 cases)

| Gate | Result | Threshold | Verdict |
|---|---:|---:|---|
| Unsupported-claim escape rate | 81.8% | ≤ 5% | **FAIL** |
| Contradicted-claim escape rate | 100.0% | ≤ 2% | **FAIL** |
| Supported-claim retention rate | 100.0% | ≥ 85% | PASS |
| Incorrect-abstention rate | 0.0% | ≤ 15% | PASS |
| Unknown-citation rejection rate | 100.0% | ≥ 100% | PASS |
| Fabricated-quote rejection rate | N/A | ≥ 100% | N/A — no quote checking exists yet |
| Verifier-failure safely represented | N/A | ≥ 100% | N/A — no verifier exists yet |
| Human/verifier agreement rate | N/A | ≥ 90% | N/A — sealed split not yet independently reviewed |

Full dataset (all 140 cases, all splits) reproduces the same rates —
expected, since the dataset is deterministically constructed rather than
sampled, so `dev`/`calibration`/`sealed` aren't different populations here.

## What this shows

The gates the current implementation already satisfies are exactly the
ones citation-existence checking is actually designed for:

- **Supported-claim retention (100%) / incorrect-abstention rate (0%)**:
  it never rejects a genuinely entailed claim — these two are numeric
  complements of each other, reported separately because over-caution and
  under-caution are different failure modes a real verifier can trade off
  against one another, even though they move together here.
- **Unknown-citation rejection (100%)**: a citation naming a chunk outside
  the retrieved pool (superseded document, model hallucinating an ID) is
  correctly caught — this is real, working defense, not nothing.

The two gates it fails are exactly the gap Release 2 exists to close:

- **100% of contradicted claims escape** — reversed relationships, wrong
  predicates, wrong numbers/dates, unsupported causal claims: every one of
  these cites a chunk that *is* in the retrieved pool, so the existence
  check passes every time regardless of what the chunk actually says.
- **81.8% of all unsupported claims escape** (contradicted + insufficient
  combined) — driven down slightly from 100% only by `correct_abstention`
  and `superseded_document_evidence`, the two categories where the citation
  itself is absent/empty and existence-checking alone is sufficient.

Per-category escape rate (contradicted/insufficient cases only):

| Category | Escape rate |
|---|---:|
| conflicting_documents | 100% |
| evidence_requiring_two_hops | 100% |
| multi_source_partially_supported_claim | 100% |
| pending_or_partial_extraction | 100% |
| reversed_relationship | 100% |
| same_entities_different_predicate | 100% |
| unsupported_causal_statement | 100% |
| unsupported_number_or_date | 100% |
| valid_but_irrelevant_citation | 100% |
| correct_abstention | 0% |
| superseded_document_evidence | 0% |

Not measured by this static baseline (require a live provider call and/or
completed human review, not applicable to a pure citation-existence
function): human/verifier agreement rate, p50/p95 latency, added cost per
answer.

## Limitations

- Synthetic, template-generated cases in a single incident-response domain
  — not yet reviewed by independent humans, and not sourced from real
  incident postmortems. See "Status" in `benchmarks/grounding/README.md`.
- 10 instances per category is enough to get a directionally reliable rate
  per category but not a tight confidence interval; treat per-category
  percentages as indicative, not precise.
- This baseline necessarily can't report the three verifier-specific gates
  (fabricated-quote rejection, safe failure representation, human/verifier
  agreement) since the current system doesn't attempt any of those checks
  and no human review has happened yet — they'll get their first real
  number once a verifier exists and the sealed split has been reviewed.

## Reproduce

```bash
python -m benchmarks.grounding.baseline_runner --split sealed
python -m benchmarks.grounding.baseline_runner --split all --json
```
