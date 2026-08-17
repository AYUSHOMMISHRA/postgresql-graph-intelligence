# Grounding benchmark: labeling instructions

## What you're labeling

Each case is one **claim** plus the **evidence pool** it was checked against
(`retrieved_evidence`) plus the specific **citation(s)** attached to it
(`asserted_citation_ids`). Your job is to assign `label`: does the cited
evidence actually entail the claim?

```text
supported     — the cited evidence, taken at face value, entails the claim.
contradicted  — the cited evidence addresses the same entities/relationship
                but asserts something different (reversed direction, wrong
                predicate, wrong number/date, etc.)
insufficient  — the cited evidence doesn't address the claim at all: it's
                irrelevant, missing, invalid, or incomplete.
```

Label `contradicted` only when the evidence positively asserts something
inconsistent with the claim. If the evidence is merely silent on the claim,
that's `insufficient`, not `contradicted` — these are different failure
modes for a verifier to catch (entailment-checking vs. hallucination-of-fact),
and conflating them would hide which one a verifier is actually good at.

## Rules

1. **Judge only what's written.** Don't use outside domain knowledge about
   whether a claim is plausible — a claim can be true in the real world and
   still `insufficient` here, if the cited evidence doesn't say so.
2. **`asserted_citation_ids` referencing evidence outside `retrieved_evidence`
   is always `insufficient`** (or `contradicted` if another, valid citation
   in the same list is also present and contradicts — rare; when in doubt,
   `insufficient`). A citation to something never retrieved is not evidence.
3. **Partial multi-source support is `insufficient`, not `supported`.** If a
   claim combines two sub-facts and only one citation actually supports its
   half, the claim as a whole is not fully entailed.
4. **Two independent reviewers must agree exactly on every `sealed`-split
   case** before it's usable for a gating measurement. Disagreements get
   discussed and either resolved or the case dropped — never broken by a
   single tie-breaking vote, which just hides genuine ambiguity in the case
   itself.
5. **Never relabel a `sealed` case after a runner has been evaluated against
   it.** If a labeling mistake is found post-hoc, retire the case's `id`
   permanently (don't reuse the id for a corrected version) and add a new
   one — otherwise a later "improvement" in a reported metric could be an
   artifact of a changed answer key, not the verifier.

## Split discipline

- **`dev`** — inspect freely while building a verifier. Expect to overfit to
  it a little; that's what it's for.
- **`calibration`** — for tuning thresholds (e.g. a confidence cutoff) only.
  Look at aggregate metrics on it, not individual cases, once a specific
  threshold-search is underway.
- **`sealed`** — do not look at these cases' outcomes while iterating.
  Run against them exactly once per candidate design, record the result,
  move on. If a result on `sealed` motivates a code change, that case's
  informational value for judging *this* change is spent — treat any
  further iteration as needing a fresh sealed set, in principle, even if in
  practice the existing one gets reused pragmatically. The point is
  discipline, not superstition: the gates in `README.md` are frozen before
  the sealed run, and are not renegotiated after seeing the number.

## Status of the current dataset: `grounding-benchmark-v1-candidate`

The cases currently in `cases.json` (140, 10 per category) were constructed
(not sourced from real incidents), with `label` assigned at construction
time from an unambiguous, mechanically-checkable rule per category (e.g.
"the relationship direction in the claim is swapped from the cited text"
-> `contradicted`) — not by independent human judgment.

This is further along than a raw seed: the dataset now has committed
integrity tests (`tests/test_grounding_benchmark.py` — exact counts, unique
ids, valid enums, literal-substring supporting quotes, reproducibility from
the generator) and `reviewer_packets.py` can produce blinded worksheets
(labels/quotes/notes stripped) for the `sealed` split. That makes it
genuinely usable for exercising a verifier's mechanics, producing a first
baseline number, and handing reviewers a concrete starting point.

**It still does not satisfy rule 4 above.** No independent human reviewer
has labeled the `sealed` split yet — the constructed labels are the only
labels that exist. Until two independent reviewers have gone through
`reviewer_packets.py`'s output and `reconcile_reviews.py` shows agreement
(see that script's docstring for the reconciliation process), this is a
**release candidate for the sealed answer key, not the sealed answer key
itself**. Treat every current metric derived from this dataset (see
`CHANGELOG.md` for recorded measurements) as provisional, and do not
promote it to plain `grounding-benchmark-v1` by renaming until that review
has actually happened — the version string itself is part of what
"sealed" is supposed to mean.
