"""Integrity tests for the grounding benchmark dataset
(benchmarks/grounding/cases.json). These don't touch Postgres or an LLM --
they check the dataset itself is internally consistent, so a future edit to
generate_dataset.py (adding a category, tweaking a template) can't silently
corrupt the frozen dataset without a test noticing.

Deliberately dependency-free: no `jsonschema` package requirement, even
though dataset_schema.json exists for external tooling/documentation --
these checks re-implement the same constraints in plain Python so this test
module has no dependency beyond the standard library.
"""
import json
from pathlib import Path

import pytest

from benchmarks.grounding.generate_dataset import (
    CATEGORY_BUILDERS,
    DATASET_VERSION,
    INSTANCES_PER_CATEGORY,
    build_dataset,
)

CASES_PATH = Path(__file__).parent.parent / "benchmarks" / "grounding" / "cases.json"

REQUIRED_FIELDS = {
    "id", "category", "split", "claim", "asserted_citation_ids",
    "retrieved_evidence", "label",
}
VALID_LABELS = {"supported", "contradicted", "insufficient"}
VALID_SPLITS = {"dev", "calibration", "sealed"}
VALID_CATEGORIES = {builder.__name__ for builder in CATEGORY_BUILDERS}


@pytest.fixture(scope="module")
def dataset():
    return json.loads(CASES_PATH.read_text())


@pytest.fixture(scope="module")
def cases(dataset):
    return dataset["cases"]


def test_dataset_version_matches_generator(dataset):
    assert dataset["dataset_version"] == DATASET_VERSION, (
        "cases.json's dataset_version doesn't match generate_dataset.DATASET_VERSION -- "
        "the committed file is stale relative to the generator, or DATASET_VERSION "
        "was bumped without regenerating."
    )


def test_dataset_is_reproducible_from_generator(cases):
    """The committed cases.json must be exactly what generate_dataset.py
    produces right now -- if this fails, either the generator changed
    without regenerating cases.json, or cases.json was hand-edited."""
    assert cases == build_dataset(), "cases.json does not match a fresh run of generate_dataset.build_dataset()"


def test_exact_case_and_category_counts(cases):
    assert len(cases) == len(CATEGORY_BUILDERS) * INSTANCES_PER_CATEGORY
    counts_by_category = {}
    for case in cases:
        counts_by_category[case["category"]] = counts_by_category.get(case["category"], 0) + 1
    assert set(counts_by_category) == VALID_CATEGORIES
    for category, count in counts_by_category.items():
        assert count == INSTANCES_PER_CATEGORY, f"{category} has {count} cases, expected {INSTANCES_PER_CATEGORY}"


def test_exact_split_counts(cases):
    counts_by_split = {}
    for case in cases:
        counts_by_split[case["split"]] = counts_by_split.get(case["split"], 0) + 1
    total = len(cases)
    # 60/20/20 nominal -- assert the exact counts this dataset size produces
    # (84/28/28 of 140), not just approximate percentages, so a future
    # instance-count change is forced to update this test deliberately.
    assert counts_by_split == {"dev": 84, "calibration": 28, "sealed": 28}, counts_by_split
    assert sum(counts_by_split.values()) == total


def test_every_category_represented_in_every_split(cases):
    """A category landing entirely in one split (e.g. all its cases happen
    to be 'sealed') would make that split's per-category escape rate
    meaningless -- confirm the split cycle actually distributes every
    category across all three splits."""
    by_category_split = {}
    for case in cases:
        by_category_split.setdefault(case["category"], set()).add(case["split"])
    for category, splits in by_category_split.items():
        assert splits == VALID_SPLITS, f"{category} only appears in splits {splits}, expected all of {VALID_SPLITS}"


def test_ids_are_unique_and_stable_format(cases):
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for case in cases:
        expected_prefix = case["category"].replace("_", "-")
        assert case["id"].startswith(expected_prefix), (
            f"id {case['id']!r} doesn't start with its category's expected prefix {expected_prefix!r}"
        )


def test_required_fields_present(cases):
    for case in cases:
        missing = REQUIRED_FIELDS - set(case)
        assert not missing, f"{case['id']} is missing required fields: {missing}"


def test_labels_and_splits_and_categories_are_valid_enum_values(cases):
    for case in cases:
        assert case["label"] in VALID_LABELS, f"{case['id']} has invalid label {case['label']!r}"
        assert case["split"] in VALID_SPLITS, f"{case['id']} has invalid split {case['split']!r}"
        assert case["category"] in VALID_CATEGORIES, f"{case['id']} has invalid category {case['category']!r}"


def test_retrieved_evidence_shape(cases):
    for case in cases:
        for ev in case["retrieved_evidence"]:
            assert set(ev) >= {"source_id", "ordinal", "content"}, (
                f"{case['id']} has a malformed retrieved_evidence entry: {ev}"
            )
            assert isinstance(ev["ordinal"], int)
            assert ev["content"], f"{case['id']} has an empty-content evidence entry"


def test_asserted_citation_ids_reference_source_hash_ordinal_format(cases):
    for case in cases:
        for cid in case["asserted_citation_ids"]:
            assert "#" in cid, f"{case['id']} has a malformed citation id {cid!r} (expected 'source_id#ordinal')"


def test_supported_cases_have_supporting_quotes(cases):
    for case in cases:
        if case["label"] == "supported":
            assert case["supporting_quotes"], f"{case['id']} is labeled supported but has no supporting_quotes"


def test_supporting_quotes_are_literal_substrings_of_retrieved_evidence(cases):
    """A quote that isn't an exact substring of some retrieved chunk would
    itself be a fabricated quote -- the dataset must not contain the exact
    failure mode (PR 4's fabricated-quote check) in its own ground truth."""
    for case in cases:
        evidence_texts = [ev["content"] for ev in case["retrieved_evidence"]]
        for quote in case["supporting_quotes"]:
            assert any(quote in text for text in evidence_texts), (
                f"{case['id']}'s supporting_quote {quote!r} is not a literal substring "
                f"of any of its retrieved_evidence"
            )


def test_asserted_citations_outside_retrieved_evidence_are_never_labeled_supported(cases):
    """A citation naming a (source_id, ordinal) pair absent from
    retrieved_evidence cannot possibly support a claim -- catches a
    contradictory case (label='supported' with a dangling citation) that
    would otherwise silently corrupt the baseline's supported-claim
    retention metric."""
    for case in cases:
        allowed = {f"{ev['source_id']}#{ev['ordinal']}" for ev in case["retrieved_evidence"]}
        has_unknown_citation = any(cid not in allowed for cid in case["asserted_citation_ids"])
        if has_unknown_citation:
            assert case["label"] != "supported", (
                f"{case['id']} cites evidence outside its retrieved_evidence pool but is labeled 'supported'"
            )
