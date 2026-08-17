"""Release 2 acceptance gates for the grounding benchmark, frozen *before*
any verifier is implemented or run against the sealed split (see
../../CHANGELOG.md for recorded runs and README.md for how these are used).

Do not edit a gate's value after inspecting a sealed-split result it was
meant to judge. If a gate turns out to be miscalibrated, that's a finding to
report alongside the result, not something to quietly adjust and re-report.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Gate:
    metric: str
    comparison: str  # "<=" or ">="
    threshold: float
    description: str

    def passes(self, value: Optional[float]) -> Optional[bool]:
        if value is None:
            return None
        if self.comparison == "<=":
            return value <= self.threshold
        if self.comparison == ">=":
            return value >= self.threshold
        raise ValueError(f"unknown comparison {self.comparison!r}")


RELEASE_2_GATES = [
    Gate("unsupported_claim_escape_rate", "<=", 0.05,
         "Fraction of contradicted/insufficient claims a verifier still calls grounded."),
    Gate("contradicted_claim_escape_rate", "<=", 0.02,
         "Fraction of specifically *contradicted* claims a verifier still calls grounded -- "
         "the more dangerous failure mode (confidently wrong) than merely insufficient."),
    Gate("supported_claim_retention_rate", ">=", 0.85,
         "Fraction of genuinely supported claims a verifier still accepts (not over-conservative)."),
    Gate("incorrect_abstention_rate", "<=", 0.15,
         "Fraction of genuinely supported claims a verifier incorrectly abstains on -- the "
         "complement of supported_claim_retention_rate, reported as its own gate since "
         "over-caution (abstaining on good evidence) and under-caution (accepting bad evidence) "
         "are different failure modes a verifier can trade off against each other."),
    Gate("unknown_citation_rejection_rate", ">=", 1.00,
         "Fraction of claims citing evidence outside the retrieved pool that are correctly rejected."),
    Gate("fabricated_quote_rejection_rate", ">=", 1.00,
         "Fraction of claims with a supporting_quotes entry not a literal substring of its cited "
         "chunk that are correctly rejected. N/A for the citation-only baseline (it doesn't check "
         "quotes at all); applies once a verifier that checks quotes exists."),
    Gate("verifier_failure_safely_represented_rate", ">=", 1.00,
         "Fraction of verifier-unavailable/error cases that report a distinct failure status "
         "rather than silently falling back to 'grounded'. N/A for the citation-only baseline."),
    Gate("human_verifier_agreement_rate", ">=", 0.90,
         "Fraction of sealed-split cases where the verifier's verdict matches the reconciled "
         "human answer key. N/A until the sealed split has been through independent human "
         "review per LABELING.md -- see reconcile_reviews.py."),
]

# Metrics that are measured and disclosed, not gated with a pass/fail
# threshold (cost and latency trade off against the gates above; a gate here
# would be an arbitrary target, not evidence of a bug).
DISCLOSED_METRICS = [
    "p50_latency_ms",
    "p95_latency_ms",
    "added_cost_per_answer_usd",
]
