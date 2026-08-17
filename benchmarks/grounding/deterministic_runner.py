"""Runs postgres_graph_rag.verification.DeterministicVerifier (Release 2
PR 3's non-model layers) against the grounding benchmark and reports the
same gate metrics baseline_runner.py does, for a direct before/after
comparison against the current citation-only validator.

Each case's `retrieved_evidence` and `asserted_citation_ids` (plain
"source_id#ordinal" strings) are adapted into the real shapes
DeterministicVerifier expects (TenantRetrievedChunk evidence, bracketed
"[source_id#ordinal]" citation markers -- tenant_engine.py's own
_citation_marker() format) so this exercises the actual production code
path, not a reimplementation of it (contrast with baseline_runner.py,
which deliberately reimplements the simple citation-existence check
inline since there's no standalone function for it to import).

Usage:
    python -m benchmarks.grounding.deterministic_runner [--split sealed|dev|calibration|all]
"""
import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from postgres_graph_rag.grounding import AnswerClaim
from postgres_graph_rag.tenant_engine import TenantRetrievedChunk
from postgres_graph_rag.verification import DeterministicVerifier

from .gates import DISCLOSED_METRICS, RELEASE_2_GATES

CASES_PATH = Path(__file__).parent / "cases.json"


def _to_chunk(ev: Dict[str, Any]) -> TenantRetrievedChunk:
    return TenantRetrievedChunk(
        id=f"{ev['source_id']}-{ev['ordinal']}", document_id=ev["source_id"], source_id=ev["source_id"],
        ordinal=ev["ordinal"], content=ev["content"], rrf_score=1.0, lexical_rank=1, semantic_rank=1,
    )


def _to_bracketed(cid: str) -> str:
    return f"[{cid}]"


def deterministic_verdict(case: Dict[str, Any]) -> str:
    evidence = [_to_chunk(ev) for ev in case["retrieved_evidence"]]
    claim = AnswerClaim(
        id=case["id"], text=case["claim"],
        citation_ids=[_to_bracketed(cid) for cid in case["asserted_citation_ids"]],
    )
    verifier = DeterministicVerifier(evidence=evidence)
    [result] = asyncio.run(verifier.verify([claim]))
    return result.verdict


def load_cases(split: Optional[str]) -> List[Dict[str, Any]]:
    data = json.loads(CASES_PATH.read_text())
    cases = data["cases"]
    if split and split != "all":
        cases = [c for c in cases if c["split"] == split]
    return cases


def compute_metrics(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    predictions = [(c, deterministic_verdict(c)) for c in cases]

    unsupported = [(c, v) for c, v in predictions if c["label"] in ("contradicted", "insufficient")]
    contradicted = [(c, v) for c, v in predictions if c["label"] == "contradicted"]
    supported = [(c, v) for c, v in predictions if c["label"] == "supported"]
    unknown_citation_cases = [
        (c, v) for c, v in predictions
        if any(
            cid not in {f"{ev['source_id']}#{ev['ordinal']}" for ev in c["retrieved_evidence"]}
            for cid in c["asserted_citation_ids"]
        )
    ]

    def rate(pairs, predicate):
        return (sum(1 for _, v in pairs if predicate(v)) / len(pairs)) if pairs else None

    metrics = {
        "case_count": len(cases),
        "unsupported_claim_escape_rate": rate(unsupported, lambda v: v == "supported"),
        "contradicted_claim_escape_rate": rate(contradicted, lambda v: v == "supported"),
        "supported_claim_retention_rate": rate(supported, lambda v: v == "supported"),
        "incorrect_abstention_rate": rate(supported, lambda v: v != "supported"),
        "unknown_citation_rejection_rate": rate(unknown_citation_cases, lambda v: v != "supported"),
        # Still N/A: this layer doesn't check quotes against a model-
        # supplied offset (there's no model yet, PR 4), doesn't have a
        # failure-representation concept of its own (it's a pure function,
        # nothing to time out or error), and the sealed split still hasn't
        # been through independent human review.
        "fabricated_quote_rejection_rate": None,
        "verifier_failure_safely_represented_rate": None,
        "human_verifier_agreement_rate": None,
    }

    by_category: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "escaped": 0})
    for c, v in unsupported:
        by_category[c["category"]]["n"] += 1
        if v == "supported":
            by_category[c["category"]]["escaped"] += 1
    metrics["escape_rate_by_category"] = {
        cat: (val["escaped"] / val["n"] if val["n"] else None) for cat, val in sorted(by_category.items())
    }

    return metrics


def print_report(split_label: str, metrics: Dict[str, Any]) -> None:
    print(f"\n=== DeterministicVerifier (PR 3, non-model layers) -- split={split_label} ===")
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
        print(f"  {name}: not measured (deterministic layers make no provider calls)")

    print("\nEscape rate by category (contradicted/insufficient cases only):")
    for cat, r in metrics["escape_rate_by_category"].items():
        r_str = "N/A" if r is None else f"{r:.0%}"
        print(f"  {cat:42} {r_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="sealed", choices=["dev", "calibration", "sealed", "all"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.split)
    metrics = compute_metrics(cases)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_report(args.split, metrics)


if __name__ == "__main__":
    main()
