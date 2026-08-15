"""Generates the grounding benchmark's frozen `cases.json` from versioned
templates -- one function per category, parameterized by an instance number,
so the dataset is reproducible without committing hand-written prose for
every single case (the same approach postgres_graph_rag/evaluation.py uses
for the retrieval benchmark).

Re-running this script regenerates `cases.json` byte-for-byte from the same
seed. Once a version of `cases.json` has been used to record a `sealed`-split
result, do not regenerate it in place -- see LABELING.md rule 5. Any future
edit to a template function should bump DATASET_VERSION and be treated as
producing a *new* dataset, not a silent edit of the old one.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

DATASET_VERSION = "grounding-benchmark-v0-seed"
INSTANCES_PER_CATEGORY = 7
OUTPUT_PATH = Path(__file__).parent / "cases.json"

# 60/20/20 dev/calibration/sealed, applied per category so every category is
# represented in every split rather than risking a category landing entirely
# in one split by chance.
SPLIT_CYCLE = ["dev", "dev", "dev", "dev", "calibration", "sealed", "sealed"]


def _evidence(source_id: str, ordinal: int, content: str) -> Dict[str, Any]:
    return {"source_id": source_id, "ordinal": ordinal, "content": content}


def _cid(source_id: str, ordinal: int) -> str:
    return f"{source_id}#{ordinal}"


def explicit_supported_relationship(n: int) -> Dict[str, Any]:
    checkout, auth = f"checkout-service-{n:03d}", f"auth-service-{n:03d}"
    src = f"incident-{n:03d}-deploy"
    quote = f"{checkout} depends on {auth} for session validation."
    return {
        "category": "explicit_supported_relationship",
        "question": f"What does {checkout} depend on?",
        "claim": f"{checkout} depends on {auth}.",
        "asserted_citation_ids": [_cid(src, 0)],
        "retrieved_evidence": [_evidence(src, 0, quote)],
        "supporting_quotes": [quote],
        "label": "supported",
        "notes": "Direct, unambiguous entailment -- the baseline case a verifier must not regress on.",
    }


def valid_but_irrelevant_citation(n: int) -> Dict[str, Any]:
    checkout, auth = f"checkout-service-{n:03d}", f"auth-service-{n:03d}"
    src = f"incident-{n:03d}-oncall"
    return {
        "category": "valid_but_irrelevant_citation",
        "question": f"What does {checkout} depend on?",
        "claim": f"{checkout} depends on {auth}.",
        "asserted_citation_ids": [_cid(src, 0)],
        "retrieved_evidence": [
            _evidence(src, 0, f"The on-call rotation for {checkout} changes every Monday at 09:00 UTC."),
        ],
        "supporting_quotes": [],
        "label": "insufficient",
        "notes": "Citation ID exists in the retrieved pool (so a citation-only checker calls this grounded), but the cited text says nothing about the claimed dependency.",
    }


def reversed_relationship(n: int) -> Dict[str, Any]:
    checkout, auth = f"checkout-service-{n:03d}", f"auth-service-{n:03d}"
    src = f"incident-{n:03d}-deploy"
    quote = f"{auth} depends on {checkout} for rate-limit telemetry."
    return {
        "category": "reversed_relationship",
        "question": f"What does {checkout} depend on?",
        "claim": f"{checkout} depends on {auth}.",
        "asserted_citation_ids": [_cid(src, 0)],
        "retrieved_evidence": [_evidence(src, 0, quote)],
        "supporting_quotes": [],
        "label": "contradicted",
        "notes": "The cited text asserts the dependency in the opposite direction.",
    }


def same_entities_different_predicate(n: int) -> Dict[str, Any]:
    checkout, auth = f"checkout-service-{n:03d}", f"auth-service-{n:03d}"
    src = f"incident-{n:03d}-topology"
    quote = f"{checkout} monitors {auth}'s health-check endpoint every 30 seconds."
    return {
        "category": "same_entities_different_predicate",
        "question": f"What does {checkout} depend on?",
        "claim": f"{checkout} depends on {auth}.",
        "asserted_citation_ids": [_cid(src, 0)],
        "retrieved_evidence": [_evidence(src, 0, quote)],
        "supporting_quotes": [],
        "label": "contradicted",
        "notes": "Same two entities, but the cited relationship is monitoring, not dependency -- a verifier keying on entity co-occurrence alone would wrongly accept this.",
    }


def conflicting_documents(n: int) -> Dict[str, Any]:
    checkout, auth = f"checkout-service-{n:03d}", f"auth-service-{n:03d}"
    src_a, src_b = f"incident-{n:03d}-postmortem", f"incident-{n:03d}-runbook"
    quote_a = f"{checkout} depends on {auth} for session validation."
    quote_b = f"{checkout} does not depend on {auth}; it validates sessions locally."
    return {
        "category": "conflicting_documents",
        "question": f"What does {checkout} depend on?",
        "claim": f"{checkout} depends on {auth}.",
        "asserted_citation_ids": [_cid(src_a, 0)],
        "retrieved_evidence": [_evidence(src_a, 0, quote_a), _evidence(src_b, 0, quote_b)],
        "supporting_quotes": [],
        "label": "insufficient",
        "notes": "The cited source alone supports the claim, but the retrieved pool contains a second source directly contradicting it -- an unresolved conflict a verifier should surface, not silently pick a side on.",
    }


def unsupported_number_or_date(n: int) -> Dict[str, Any]:
    checkout = f"checkout-service-{n:03d}"
    src = f"incident-{n:03d}-timeline"
    quote = f"{checkout} experienced a 12-minute outage starting at 03:14 UTC."
    return {
        "category": "unsupported_number_or_date",
        "question": f"How long was the {checkout} outage?",
        "claim": f"{checkout} experienced a 45-minute outage.",
        "asserted_citation_ids": [_cid(src, 0)],
        "retrieved_evidence": [_evidence(src, 0, quote)],
        "supporting_quotes": [],
        "label": "contradicted",
        "notes": "The cited text gives a specific, different duration -- the claim's number was not derived from this evidence.",
    }


def unsupported_causal_statement(n: int) -> Dict[str, Any]:
    checkout, auth = f"checkout-service-{n:03d}", f"auth-service-{n:03d}"
    src = f"incident-{n:03d}-timeline"
    quote = f"{auth} latency rose at 03:10 UTC. {checkout} error rate rose at 03:12 UTC."
    return {
        "category": "unsupported_causal_statement",
        "question": f"Why did {checkout} error rate rise?",
        "claim": f"{checkout}'s error rate rose because {auth} latency increased.",
        "asserted_citation_ids": [_cid(src, 0)],
        "retrieved_evidence": [_evidence(src, 0, quote)],
        "supporting_quotes": [],
        "label": "insufficient",
        "notes": "The evidence establishes temporal sequence, not causation -- a common LLM over-reach the current citation-only check cannot catch at all.",
    }


def multi_source_supported_claim(n: int) -> Dict[str, Any]:
    checkout, auth, team = f"checkout-service-{n:03d}", f"auth-service-{n:03d}", f"Identity Team {n:03d}"
    src_a, src_b = f"incident-{n:03d}-deploy", f"incident-{n:03d}-ownership"
    quote_a = f"{checkout} depends on {auth} for session validation."
    quote_b = f"{auth} is owned by {team}."
    return {
        "category": "multi_source_supported_claim",
        "question": f"Which team owns the service {checkout} depends on?",
        "claim": f"{checkout} depends on {auth}, which is owned by {team}.",
        "asserted_citation_ids": [_cid(src_a, 0), _cid(src_b, 0)],
        "retrieved_evidence": [_evidence(src_a, 0, quote_a), _evidence(src_b, 0, quote_b)],
        "supporting_quotes": [quote_a, quote_b],
        "label": "supported",
        "notes": "Each half of the compound claim is entailed by a different, correctly-cited source.",
    }


def multi_source_partially_supported_claim(n: int) -> Dict[str, Any]:
    checkout, auth, team = f"checkout-service-{n:03d}", f"auth-service-{n:03d}", f"Identity Team {n:03d}"
    src_a, src_b = f"incident-{n:03d}-deploy", f"incident-{n:03d}-oncall"
    quote_a = f"{checkout} depends on {auth} for session validation."
    quote_b = f"The on-call rotation for {auth} changes every Monday at 09:00 UTC."
    return {
        "category": "multi_source_partially_supported_claim",
        "question": f"Which team owns the service {checkout} depends on?",
        "claim": f"{checkout} depends on {auth}, which is owned by {team}.",
        "asserted_citation_ids": [_cid(src_a, 0), _cid(src_b, 0)],
        "retrieved_evidence": [_evidence(src_a, 0, quote_a), _evidence(src_b, 0, quote_b)],
        "supporting_quotes": [quote_a],
        "label": "insufficient",
        "notes": "Only the dependency half is supported; the ownership half's citation is valid but doesn't mention ownership at all -- the compound claim as a whole is not fully entailed.",
    }


def evidence_requiring_two_hops(n: int) -> Dict[str, Any]:
    checkout, auth, db = f"checkout-service-{n:03d}", f"auth-service-{n:03d}", f"identity-db-{n:03d}"
    src_a, src_b = f"incident-{n:03d}-deploy", f"incident-{n:03d}-topology"
    quote_a = f"{checkout} depends on {auth} for session validation."
    quote_b = f"{auth} depends on {db} for credential storage."
    return {
        "category": "evidence_requiring_two_hops",
        "question": f"Does {checkout} depend on {db}?",
        "claim": f"{checkout} depends on {db}.",
        "asserted_citation_ids": [_cid(src_a, 0), _cid(src_b, 0)],
        "retrieved_evidence": [_evidence(src_a, 0, quote_a), _evidence(src_b, 0, quote_b)],
        "supporting_quotes": [],
        "label": "insufficient",
        "notes": "Both hops are individually true, but neither cited passage states the transitive relationship directly -- text entailment should not accept an unstated inference, even one a graph traversal would confirm structurally.",
    }


def superseded_document_evidence(n: int) -> Dict[str, Any]:
    checkout, auth = f"checkout-service-{n:03d}", f"auth-service-{n:03d}"
    stale_src = f"incident-{n:03d}-deploy-v1"
    current_src = f"incident-{n:03d}-deploy-v2"
    return {
        "category": "superseded_document_evidence",
        "question": f"What does {checkout} depend on?",
        "claim": f"{checkout} depends on {auth}.",
        "asserted_citation_ids": [_cid(stale_src, 0)],
        "retrieved_evidence": [
            _evidence(current_src, 0, f"{checkout} depends on {auth} for session validation (revision 2)."),
        ],
        "supporting_quotes": [],
        "label": "insufficient",
        "notes": "The citation references a document revision that a newer publication has replaced, so it's absent from what was actually retrieved -- the same shape as citing a chunk id that was deleted by a concurrent update.",
    }


def pending_or_partial_extraction(n: int) -> Dict[str, Any]:
    checkout, auth = f"checkout-service-{n:03d}", f"auth-service-{n:03d}"
    src = f"incident-{n:03d}-postmortem"
    quote = (
        f"During the incident, engineers noted that {checkout} calls out to several internal "
        f"services, including {auth}, while diagnosing the root cause."
    )
    return {
        "category": "pending_or_partial_extraction",
        "question": f"What does {checkout} depend on?",
        "claim": f"{checkout} depends on {auth}.",
        "asserted_citation_ids": [_cid(src, 0)],
        "retrieved_evidence": [_evidence(src, 0, quote)],
        "supporting_quotes": [],
        "label": "insufficient",
        "notes": "The chunk is text-searchable (extraction hasn't produced a confirmed graph relationship for it yet, or only partially), and the prose alone only mentions the services in passing without asserting an actual dependency.",
    }


def correct_abstention(n: int) -> Dict[str, Any]:
    checkout = f"checkout-service-{n:03d}"
    unrelated = f"incident-{n:03d}-billing"
    return {
        "category": "correct_abstention",
        "question": f"What does {checkout} depend on?",
        "claim": "Insufficient evidence to answer from the indexed sources.",
        "asserted_citation_ids": [],
        "retrieved_evidence": [
            _evidence(unrelated, 0, "The billing reconciliation job runs nightly at 02:00 UTC."),
        ],
        "supporting_quotes": [],
        "label": "insufficient",
        "notes": "Nothing retrieved addresses the question; abstaining (no citations, no claim of a relationship) is the correct behavior, and the current citation-only check already handles this correctly (bool([]) is False).",
    }


def incorrect_abstention_pressure(n: int) -> Dict[str, Any]:
    checkout, auth = f"checkout-service-{n:03d}", f"auth-service-{n:03d}"
    src = f"incident-{n:03d}-deploy"
    # Deliberately phrased with punctuation/casing that differs from the
    # question's identifier formatting, the exact real failure mode
    # documented in tenant_engine.py's _normalize_identifier.
    quote = f"{checkout.replace('-', ' ')} depends on {auth.replace('-', ' ')} for session validation."
    return {
        "category": "incorrect_abstention_pressure",
        "question": f"What does {checkout} depend on?",
        "claim": f"{checkout} depends on {auth}.",
        "asserted_citation_ids": [_cid(src, 0)],
        "retrieved_evidence": [_evidence(src, 0, quote)],
        "supporting_quotes": [quote],
        "label": "supported",
        "notes": "The evidence genuinely supports the claim; only the surface formatting (hyphens vs. spaces) differs from the question. A verifier under pressure to be conservative must not abstain here just because of superficial phrasing.",
    }


CATEGORY_BUILDERS = [
    explicit_supported_relationship,
    valid_but_irrelevant_citation,
    reversed_relationship,
    same_entities_different_predicate,
    conflicting_documents,
    unsupported_number_or_date,
    unsupported_causal_statement,
    multi_source_supported_claim,
    multi_source_partially_supported_claim,
    evidence_requiring_two_hops,
    superseded_document_evidence,
    pending_or_partial_extraction,
    correct_abstention,
    incorrect_abstention_pressure,
]


def build_dataset() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for builder in CATEGORY_BUILDERS:
        category = builder.__name__
        for i in range(1, INSTANCES_PER_CATEGORY + 1):
            case = builder(i)
            case["id"] = f"{category.replace('_', '-')}-{i:03d}"
            case["split"] = SPLIT_CYCLE[(i - 1) % len(SPLIT_CYCLE)]
            cases.append(case)
    return cases


def main() -> None:
    cases = build_dataset()
    payload = {
        "dataset_version": DATASET_VERSION,
        "case_count": len(cases),
        "cases": cases,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    by_split: Dict[str, int] = {}
    for c in cases:
        by_split[c["split"]] = by_split.get(c["split"], 0) + 1
    print(f"Wrote {len(cases)} cases ({len(CATEGORY_BUILDERS)} categories x {INSTANCES_PER_CATEGORY} instances) "
          f"to {OUTPUT_PATH}")
    print(f"Split sizes: {by_split}")


if __name__ == "__main__":
    main()
