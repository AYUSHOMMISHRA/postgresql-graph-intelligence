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
| `model_verifier_runner.py` | Runs `postgres_graph_rag.model_verifier.ModelEntailmentVerifier` (PR 4) against the dataset; `--simulate-failure` measures failure-handling without real credentials, JSON/Markdown output, `--fail-on-gate` exit code. |
| `reviewer_packets.py` | Produces a blinded (answer-key-stripped) worksheet for one human reviewer. |
| `reconcile_reviews.py` | Compares two completed reviewer worksheets, reports agreement, writes agreed labels. |
| `fabrication_fixtures.py` / `fabrication_fixtures.json` | Real-quote/fabricated-quote pairs for a future verifier's quote-validation layer. |
| `verifier_fixtures.py` | Stub verifiers (raising, timing out, malformed response) for a verifier's failure-handling tests -- used by both `test_grounding_fixtures.py` and `test_model_verifier.py`. |
| `../../postgres_graph_rag/grounding.py` | PR 2: the verification contract types (`AnswerClaim`, `ClaimVerification`, `VerifiedAnswerResult`, `Verifier` protocol, `VerifierUnavailableError`, `GroundingMode`). |
| `../../postgres_graph_rag/verification.py` | PR 3: the deterministic (non-model) verification layers, including `DeterministicVerifier`. |
| `../../postgres_graph_rag/model_verifier.py` | PR 4: `ModelEntailmentVerifier` -- batched model entailment layered on top of PR 3, one bounded provider call per answer, server-side quote validation, `VerifierUnavailableError` on any provider failure. |
| `../../postgres_graph_rag/extractor.py` | `LLMExtractor.verify_claims()` -- the provider-specific (OpenAI/Google) structured-output plumbing `ModelEntailmentVerifier` calls, alongside the pre-existing `extract_triplets()`. |
| `../../tests/test_grounding_benchmark.py` | Committed integrity tests for `cases.json` (schema shape, unique ids, exact counts, literal-substring quotes, reproducibility). |
| `../../tests/test_grounding_fixtures.py` | Committed integrity tests for the fabrication/verifier fixtures. |
| `../../tests/test_grounding_contract.py` | Tests for the PR 2 contract types. |
| `../../tests/test_grounding_verification.py` | Tests for the PR 3 deterministic layers (stale evidence, invalid citations, reversed relationships, conflicts, partial support, fabricated quotes, policy evaluation, rendering). |
| `../../tests/test_model_verifier.py` | Tests for PR 4 (`ModelEntailmentVerifier`), all against a mocked extractor -- deterministic-first short-circuiting, fabricated-quote rejection, provider-failure handling, telemetry. |
| `../../tests/test_extractor.py` | Includes tests for `LLMExtractor.verify_claims()`'s OpenAI/Google structured-output plumbing. |
| `../../tests/test_model_verifier_runner.py` | Tests for `model_verifier_runner.py` against `AlwaysFailingExtractor` -- gate computation, exit-code behavior, Markdown/JSON output, no real provider anywhere. |

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

See `../../CHANGELOG.md` for the recorded result and what it means.

## Running the PR 3 before/after comparison

```bash
python -m benchmarks.grounding.deterministic_runner --split sealed
python -m benchmarks.grounding.deterministic_runner --split all --json
```

See `../../CHANGELOG.md` for the recorded result: contradicted/unsupported-claim
escape closes to 0% with no model call, at a documented, expected cost to
supported-claim retention that PR 4 (batched model entailment) is scoped
to recover.

## PR 4/6 status: `model_verifier_runner.py` built and tested, no live-provider run yet

`ModelEntailmentVerifier` (`../../postgres_graph_rag/model_verifier.py`) and
its runner (`model_verifier_runner.py`) are implemented and tested — 9 + 2
tests in `test_model_verifier.py`, 7 provider-plumbing tests in
`test_extractor.py`, and runner tests in `test_model_verifier_runner.py` —
all against a mocked, `AlwaysFailingExtractor`, or fabrication-check stub,
never a real provider:

```bash
# No credentials needed. --simulate-failure runs the *primary* pass
# through an always-failing extractor too, so every quality gate that
# depends on a resolved case correctly reports N/A instead of a
# misleading number.
python -m benchmarks.grounding.model_verifier_runner --simulate-failure --split sealed
```

`verifier_failure_safely_represented_rate` and
`fabricated_quote_rejection_rate` are measured on *every* invocation
(live or simulated), via `run_failure_safety_check()` and
`run_fabrication_check()` -- both run a small, dedicated offline exercise
unconditionally, independent of `--simulate-failure`, so a genuine live
release run populates these gates instead of leaving them N/A.

A `verification_failed` outcome (provider error/timeout/malformed response)
is treated as an *incomplete* result for that case — excluded from every
escape/retention/rejection rate's denominator, never counted as either an
acceptance or a rejection, and disclosed via `verification_failure_count`.
Folding it into a rate (the original bug here) let a provider outage look
identical to, or better than, a correct rejection —
`test_outage_cannot_improve_escape_or_rejection_rates` and
`test_outage_cannot_improve_unknown_citation_rejection_rate` lock this in.

Two gate-checking modes: `--fail-on-gate` (permissive — only an actual FAIL
blocks, for `dev`/`calibration` iteration) and `--strict-release-gate` (an
N/A verdict blocks too, and so does any `verification_failed` case — an
incomplete run is never a valid release signal, regardless of which gates
it happens to touch — required for the actual release go/no-go decision).

What's still missing before this PR's own before/after result can be
recorded the way PR 3's was: an actual run against a real OpenAI or Gemini
API key, which costs real money and needs credentials this environment
doesn't have, plus the sealed split's reconciled human answer key:

```bash
OPENAI_API_KEY=... python -m benchmarks.grounding.model_verifier_runner \
    --provider openai --split sealed --strict-release-gate \
    --reviewed-answer-key benchmarks/grounding/packets/sealed_reconciled.json \
    --cost-per-1k-prompt-tokens <rate> --cost-per-1k-completion-tokens <rate> \
    --markdown-out grounding-benchmark-model-verifier.md
```

`--reviewed-answer-key` is validated, not merely parsed: it must be for
the `sealed` split, list exactly two distinct reviewers, have no
duplicate ids or invalid labels, and cover *exactly* the sealed split's
28 case ids -- a partial key (even one that happens to agree on every
case it does cover) is rejected outright, since a key covering 1 of 28
cases could otherwise report a misleading 100% agreement rate. Any
violation exits with status 2 and a message on stderr, not a silent
wrong number or an unhandled traceback.

That run — `deterministic_runner.py`'s result vs. this one against the
sealed split, with a real `LLMExtractor` — is the natural follow-up once
credentials are available, and should be recorded in `CHANGELOG.md`
alongside real latency/cost/token numbers, not estimated ones.
`human_verifier_agreement_rate`
stays N/A without `--reviewed-answer-key` — it needs the sealed split's
independent human review (see below) reconciled into the file that flag
points at, which is orthogonal to having API credentials, so
`--strict-release-gate` will correctly still block the actual release
decision until that review has happened and been supplied.

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
