"""Runs the *current* citation-marker validator (tenant_engine.py's
`grounded = bool(markers) and not invalid`) against the grounding benchmark
and reports the gate metrics from gates.py.

This intentionally does not call the live retrieve()/answer() pipeline --
each case already specifies its own retrieved_evidence pool and the citation
IDs a model attached to its claim, so the check under test can run as a pure
function. What's reproduced here is exactly the semantic check tenant_engine
performs: a citation marker is "valid" iff it names a (source_id, ordinal)
pair that's actually in the retrieved evidence pool. tenant_engine.py's
_citation_marker() formats that pair as "[source_id#ordinal]" (with the
brackets a regex later strips out of generated text); this runner uses the
same "source_id#ordinal" identity without the parsing-layer brackets, since
those are an implementation detail of extracting markers from free text, not
part of the validity check itself.

Usage:
    python -m benchmarks.grounding.baseline_runner [--split sealed|dev|calibration|all]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .gates import DISCLOSED_METRICS, RELEASE_2_GATES

CASES_PATH = Path(__file__).parent / "cases.json"


def citation_only_grounded(case: Dict[str, Any]) -> bool:
    """Reimplementation of tenant_engine.py's current check:
    `grounded = bool(markers) and not invalid`, where `invalid` is any
    marker absent from the retrieved evidence pool."""
    asserted = case["asserted_citation_ids"]
    if not asserted:
        return False
    allowed = {f"{ev['source_id']}#{ev['ordinal']}" for ev in case["retrieved_evidence"]}
    return all(cid in allowed for cid in asserted)


def load_cases(split: Optional[str]) -> List[Dict[str, Any]]:
    data = json.loads(CASES_PATH.read_text())
    cases = data["cases"]
    if split and split != "all":
        cases = [c for c in cases if c["split"] == split]
    return cases


def compute_metrics(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    predictions = [(c, citation_only_grounded(c)) for c in cases]

    unsupported = [(c, p) for c, p in predictions if c["label"] in ("contradicted", "insufficient")]
    contradicted = [(c, p) for c, p in predictions if c["label"] == "contradicted"]
    supported = [(c, p) for c, p in predictions if c["label"] == "supported"]
    # Cases specifically constructed so the asserted citation is absent from
    # the retrieved pool (superseded_document_evidence is the clearest
    # example of this shape; correct_abstention has no citations at all).
    unknown_citation_cases = [
        (c, p) for c, p in predictions
        if any(
            cid not in {f"{ev['source_id']}#{ev['ordinal']}" for ev in c["retrieved_evidence"]}
            for cid in c["asserted_citation_ids"]
        )
    ]

    def rate(pairs, predicate):
        return (sum(1 for _, p in pairs if predicate(p)) / len(pairs)) if pairs else None

    metrics = {
        "case_count": len(cases),
        "unsupported_claim_escape_rate": rate(unsupported, lambda p: p is True),
        "contradicted_claim_escape_rate": rate(contradicted, lambda p: p is True),
        "supported_claim_retention_rate": rate(supported, lambda p: p is True),
        # The complement of supported_claim_retention_rate, reported under
        # its own name (see gates.py) since "over-cautious" and
        # "under-cautious" are different failure modes to disclose
        # separately even though they're numerically tied together here.
        "incorrect_abstention_rate": rate(supported, lambda p: p is False),
        "unknown_citation_rejection_rate": rate(unknown_citation_cases, lambda p: p is False),
        # Not applicable to a citation-existence-only check: it never looks
        # at quote text, represents its own failure as anything but a
        # binary grounded/not-grounded, or has been compared against a
        # reconciled human answer key yet -- reported as None (not 0 or 1)
        # so they aren't mistaken for a pass or fail.
        "fabricated_quote_rejection_rate": None,
        "verifier_failure_safely_represented_rate": None,
        "human_verifier_agreement_rate": None,
    }

    by_category: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "escaped": 0})
    for c, p in unsupported:
        by_category[c["category"]]["n"] += 1
        if p is True:
            by_category[c["category"]]["escaped"] += 1
    metrics["escape_rate_by_category"] = {
        cat: (v["escaped"] / v["n"] if v["n"] else None) for cat, v in sorted(by_category.items())
    }

    return metrics


def print_report(split_label: str, metrics: Dict[str, Any]) -> None:
    print(f"\n=== Baseline: current citation-marker validator -- split={split_label} ===")
    print(f"cases: {metrics['case_count']}")
    print("\nGates:")
    for gate in RELEASE_2_GATES:
        value = metrics.get(gate.metric)
        verdict = gate.passes(value)
        verdict_str = "N/A" if verdict is None else ("PASS" if verdict else "FAIL")
        value_str = "N/A" if value is None else f"{value:.1%}"
        print(f"  [{verdict_str:4}] {gate.metric:42} = {value_str:>6}  (gate: {gate.comparison} {gate.threshold:.0%})  -- {gate.description}")

    print("\nDisclosed (not gated):")
    for name in DISCLOSED_METRICS:
        print(f"  {name}: not measured by this static baseline (requires a live provider call / human review pass)")

    print("\nEscape rate by category (contradicted/insufficient cases only):")
    for cat, r in metrics["escape_rate_by_category"].items():
        r_str = "N/A" if r is None else f"{r:.0%}"
        print(f"  {cat:42} {r_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="sealed", choices=["dev", "calibration", "sealed", "all"])
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text report.")
    args = parser.parse_args()

    cases = load_cases(args.split)
    metrics = compute_metrics(cases)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_report(args.split, metrics)


if __name__ == "__main__":
    main()
