# Grounding benchmark

Measures whether a citation actually entails the claim it's attached to —
the gap the citation-marker-only validator in `tenant_engine.py` cannot
close (`grounded = bool(markers) and not invalid` checks that a citation ID
exists in the retrieved evidence pool, never that the cited text supports
what the claim says). This is Release 2's frozen evaluation harness, built
*before* any verifier — see `../../CHANGELOG.md` and the CTO release notes
in this project's history for why that ordering matters: a benchmark built
alongside the thing it measures tends to get shaped around that thing's
strengths.

## Files

| File | Purpose |
|---|---|
| `dataset_schema.json` | JSON Schema for one case (claim + citations + evidence pool + label). |
| `LABELING.md` | How `label` is assigned, split discipline, and the current dataset's status. |
| `generate_dataset.py` | Builds `cases.json` from versioned templates, one per category. |
| `cases.json` | The frozen dataset (generated, not hand-edited — see below). 140 cases, 14 categories, 84/28/28 dev/calibration/sealed. |
| `gates.py` | Release 2 acceptance gates, frozen before any sealed-split run. |
| `baseline_runner.py` | Runs the *current* citation-only validator against the dataset and reports gate metrics. |
| `deterministic_runner.py` | Runs `postgres_graph_rag.verification.DeterministicVerifier` (PR 3's non-model layers) against the dataset for a before/after comparison. |
| `reviewer_packets.py` | Produces a blinded (answer-key-stripped) worksheet for one human reviewer. |
| `reconcile_reviews.py` | Compares two completed reviewer worksheets, reports agreement, writes agreed labels. |
| `fabrication_fixtures.py` / `fabrication_fixtures.json` | Real-quote/fabricated-quote pairs for a future verifier's quote-validation layer. |
| `verifier_fixtures.py` | Stub verifiers (raising, timing out, malformed response) for a future verifier's failure-handling tests (PR 4/5). |
| `../../postgres_graph_rag/grounding.py` | PR 2: the verification contract types (`AnswerClaim`, `ClaimVerification`, `VerifiedAnswerResult`, `Verifier` protocol, `GroundingMode`). |
| `../../postgres_graph_rag/verification.py` | PR 3: the deterministic (non-model) verification layers, including `DeterministicVerifier`. |
| `../../tests/test_grounding_benchmark.py` | Committed integrity tests for `cases.json` (schema shape, unique ids, exact counts, literal-substring quotes, reproducibility). |
| `../../tests/test_grounding_fixtures.py` | Committed integrity tests for the fabrication/verifier fixtures. |
| `../../tests/test_grounding_contract.py` | Tests for the PR 2 contract types. |
| `../../tests/test_grounding_verification.py` | Tests for the PR 3 deterministic layers (stale evidence, invalid citations, reversed relationships, conflicts, partial support, fabricated quotes, policy evaluation, rendering). |

## Regenerating `cases.json`

```bash
python benchmarks/grounding/generate_dataset.py
```

Only do this to add new categories/instances as a deliberate new dataset
version (bump `DATASET_VERSION` in `generate_dataset.py`), never to silently
edit existing sealed-split cases — see `LABELING.md` rule 5.

## Running the baseline

```bash
python -m benchmarks.grounding.baseline_runner --split sealed
python -m benchmarks.grounding.baseline_runner --split all --json
```

See `../../docs/results/grounding-benchmark-baseline.md` for the recorded
result and what it means.

## Running the PR 3 before/after comparison

```bash
python -m benchmarks.grounding.deterministic_runner --split sealed
python -m benchmarks.grounding.deterministic_runner --split all --json
```

See `../../docs/results/grounding-benchmark-deterministic-verifier.md` for
the recorded result: contradicted/unsupported-claim escape closes to 0%
with no model call, at a documented, expected cost to supported-claim
retention that PR 4 (batched model entailment) is scoped to recover.

## Getting the sealed split independently reviewed

```bash
python -m benchmarks.grounding.reviewer_packets --reviewer <name-1>
python -m benchmarks.grounding.reviewer_packets --reviewer <name-2>
# each reviewer independently fills in reviewer_label/reviewer_reasoning
# in their own packets/sealed_review_<name>.json, without seeing the
# other's answers or the dataset's constructed label
python -m benchmarks.grounding.reconcile_reviews \
    benchmarks/grounding/packets/sealed_review_<name-1>.json \
    benchmarks/grounding/packets/sealed_review_<name-2>.json
```

`packets/` is gitignored — it holds working files for an in-progress
review, not benchmark artifacts to commit. Once reviewers agree (or
disagreements are discussed and resolved per `LABELING.md` rule 4), splice
the reconciled labels into `cases.json`'s `label` field as its own
reviewed, deliberate commit, and only then rename `DATASET_VERSION` from
`...-v1-candidate` to a plain `...-v1` in `generate_dataset.py`.

## Status: `v1-candidate`, integrity-tested, not yet independently reviewed

The 140 cases here (10 per category) were constructed with a
mechanically-derivable label per category, not independently labeled by
two human reviewers as `LABELING.md` requires for the `sealed` split.
Compared to the earlier v0 seed, this dataset now has:

- exact case/category/split counts and dataset-generator reproducibility
  enforced by committed tests (`tests/test_grounding_benchmark.py`),
- tooling to actually run the independent-review process
  (`reviewer_packets.py`, `reconcile_reviews.py`) — smoke-tested against
  itself (agreement math and disagreement reporting both verified), not
  yet run against real reviewers,
- fixtures ready for the quote-validation and failure-handling layers a
  future verifier will need (`fabrication_fixtures.py`, `verifier_fixtures.py`).

**It is still not a validated gating benchmark.** No independent human has
labeled the `sealed` split — the constructed labels are the only labels
that exist. Before treating any `sealed`-split result as a real go/no-go
signal, run the review process above, and don't rename the dataset to
plain `v1` until it's actually done.

Expanding the dataset further (real incident postmortems instead of
synthetic templates, adversarial cases contributed by red-teaming an
actual verifier) is a reasonable next step but a separate one from this PR.
