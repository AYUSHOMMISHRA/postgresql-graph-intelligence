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
| `cases.json` | The frozen dataset (generated, not hand-edited — see below). |
| `gates.py` | Release 2 acceptance gates, frozen before any sealed-split run. |
| `baseline_runner.py` | Runs the *current* citation-only validator against the dataset and reports gate metrics. |

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

## Status: v0 seed, not yet a validated gating benchmark

The 98 cases here were constructed with a mechanically-derivable label per
category, not independently labeled by two human reviewers as
`LABELING.md` requires for the `sealed` split. That makes this dataset
useful for:

- exercising a verifier's mechanics end-to-end before spending real review
  time,
- producing a first, directionally-correct baseline number, and
- giving human reviewers a concrete starting point to *revise* rather than
  author from a blank page,

but **not yet** useful as the actual sealed gate for a shipped verifier.
Before treating any `sealed`-split result as a real go/no-go signal:

1. Recruit two independent reviewers.
2. Have each label every `sealed` case from `retrieved_evidence` +
   `asserted_citation_ids` alone (blind to the constructed `label`).
3. Reconcile disagreements per `LABELING.md`; drop or fix cases that don't
   converge.
4. Only then treat `gates.py`'s thresholds as binding.

Expanding the dataset (more categories, real incident postmortems instead
of synthetic templates, adversarial cases contributed by red-teaming an
actual verifier) is a reasonable next step but a separate one from this
scaffold.
