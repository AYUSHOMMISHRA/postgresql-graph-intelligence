"""Release 2 PR 2: the verification contract -- types and interfaces only.

Deliberately no provider-based entailment behavior lives here (see
benchmarks/grounding/README.md for the PR sequencing this follows). What's
defined:

- `AnswerClaim` / `ClaimVerification` / `VerifiedAnswerResult`: the typed
  result shape a verifier produces, replacing an ever-more-overloaded plain
  `grounded: bool` with an explicit `grounding_status` and a `grounded`
  property derived from it for backward compatibility with `AnswerResult`
  (tenant_engine.py) call sites that only check `.grounded`.
- `Verifier`: the protocol a verification backend implements. Structurally
  compatible with `benchmarks/grounding/verifier_fixtures.py`'s stub
  verifiers (`RaisingVerifier`, `TimingOutVerifier`,
  `MalformedResponseVerifier`) without those fixtures needing to import
  from this module -- they were written against the same
  `async def verify(self, claims) -> ...` shape ahead of this contract
  existing, and still satisfy it.
- `GroundingMode`: the three policies PR 5 will expose on the answer path
  (`citation_only`, `verified`, `verified_strict`).

Nothing in `tenant_engine.py`'s actual `answer()` method changes yet --
that wiring is PR 3 (deterministic layers) and PR 4 (batched model
entailment).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, runtime_checkable

# ----------------------------------------------------------------------
# Claims and verdicts
# ----------------------------------------------------------------------

Verdict = Literal["supported", "contradicted", "insufficient"]

# The specific reason a claim's citation/quote was rejected or accepted --
# distinct from `Verdict` (the outcome) so a caller/reviewer can tell *why*
# without parsing free text. Deliberately validated (see
# ClaimVerification.__post_init__) the same way tenancy.py's
# set_extraction_status() validates its own status column: catch an invalid
# value here, in the type that owns it, not wherever it happens to get used.
ReasonCode = Literal[
    "explicit_relation_confirmed",
    "citation_missing",
    "citation_invalid",
    "citation_outside_retrieved_evidence",
    "citation_references_superseded_revision",
    "relationship_reversed",
    "different_predicate_same_entities",
    "numeric_or_date_mismatch",
    "unsupported_causal_claim",
    "quote_not_found_verbatim",
    "evidence_conflicting",
    "evidence_irrelevant",
    "evidence_insufficient",
    "partial_multi_source_support",
    "requires_unstated_transitive_inference",
    "model_entailment_supported",
    "model_entailment_contradicted",
]

_VALID_REASON_CODES = frozenset(ReasonCode.__args__)  # type: ignore[attr-defined]
_VALID_VERDICTS = frozenset(Verdict.__args__)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class AnswerClaim:
    """One atomic factual sentence extracted from a generated answer, plus
    the citation ids the answer attached to it. The server renders the
    final answer text from claims that survive verification (PR 3) rather
    than trusting an unconstrained `final_answer` string the model could
    have padded with unverified facts -- see the "Do not accept an
    unconstrained final_answer" constraint in the Release 2 plan."""

    id: str
    text: str
    citation_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ClaimVerification:
    """One verifier's verdict on one `AnswerClaim`."""

    claim_id: str
    verdict: Verdict
    supporting_quotes: List[str] = field(default_factory=list)
    reason_code: ReasonCode = "evidence_insufficient"
    # Ranking metadata only -- explicitly NOT a calibrated probability (see
    # the Release 2 plan's "treat model confidence as uncalibrated metadata"
    # constraint). None when a verifier doesn't produce one at all (e.g. the
    # deterministic-only layers in PR 3, before any model call is involved).
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(_VALID_VERDICTS)}, got {self.verdict!r}")
        if self.reason_code not in _VALID_REASON_CODES:
            raise ValueError(f"reason_code must be one of {sorted(_VALID_REASON_CODES)}, got {self.reason_code!r}")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence!r}")


# ----------------------------------------------------------------------
# Overall result
# ----------------------------------------------------------------------

# Every distinct state a verified answer can end up in. Deliberately more
# granular than a bool: "citation_valid_only" (no entailment check ran at
# all -- today's actual behavior) is a genuinely different claim than
# "verified" (every material claim was checked and supported), and
# "verification_failed" (the verifier itself errored) must never be
# silently reported as either -- see the Release 2 plan's "never silently
# claim verification" constraint.
GroundingStatus = Literal[
    "verified",             # every material claim checked and supported
    "partially_verified",   # some claims verified and kept; unsupported ones removed
    "citation_valid_only",  # citation-only mode: markers checked for existence only, no entailment
    "contradicted",         # at least one claim was checked and found to contradict the evidence
    "insufficient",         # not enough evidence for the claim(s); answer withheld or emptied
    "verification_failed",  # the verifier was invoked but errored, timed out, or returned garbage
    "abstained",            # no evidence retrieved, or the model itself declined to answer
]

_VALID_GROUNDING_STATUSES = frozenset(GroundingStatus.__args__)  # type: ignore[attr-defined]

# grounding_status values a caller checking only the plain `grounded: bool`
# field should still see as "yes, use this answer" -- the rest (contradicted,
# insufficient, verification_failed, abstained) are all "no" for a caller
# that hasn't been updated to look at the richer status.
_GROUNDED_STATUSES = frozenset({"verified", "partially_verified", "citation_valid_only"})


def validate_grounding_status(status: str) -> GroundingStatus:
    """Shared validator for any type storing a `GroundingStatus` value.

    Originally private to `VerifiedAnswerResult.__post_init__`; promoted to
    a module-level function so `tenant_engine.AnswerResult` -- the one
    production result type on the answer path -- can apply the same
    validation instead of accepting an unvalidated `Optional[str]`.
    """
    if status not in _VALID_GROUNDING_STATUSES:
        raise ValueError(f"grounding_status must be one of {sorted(_VALID_GROUNDING_STATUSES)}, got {status!r}")
    return status  # type: ignore[return-value]


def is_grounded_status(status: Optional[str]) -> bool:
    """Whether `status` counts as "grounded" for a caller that only checks
    the plain `grounded: bool` compatibility field. `None` (not yet
    computed) is never grounded."""
    return status in _GROUNDED_STATUSES


@dataclass(frozen=True)
class VerifiedAnswerResult:
    """Successor to tenant_engine.py's `AnswerResult`, once PR 3/4 wire a
    real verifier into the answer path. `grounded` is kept as a computed
    property (not a stored field) specifically so it can never drift out of
    sync with `grounding_status` -- there is exactly one source of truth."""

    claims: List[AnswerClaim]
    verifications: List[ClaimVerification]
    answer: str
    grounding_status: GroundingStatus
    citations: List[Any] = field(default_factory=list)  # tenant_engine.Citation, kept as Any to avoid an import cycle
    usage: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_grounding_status(self.grounding_status)

    @property
    def grounded(self) -> bool:
        """Compatibility with AnswerResult.grounded (a plain bool). New
        code should check `grounding_status` directly -- this collapses
        seven distinct states down to the same two everything already
        knows how to handle, which is exactly the loss of information the
        richer field exists to avoid."""
        return is_grounded_status(self.grounding_status)

    def verification_for(self, claim_id: str) -> Optional[ClaimVerification]:
        for v in self.verifications:
            if v.claim_id == claim_id:
                return v
        return None


# ----------------------------------------------------------------------
# Verifier interface
# ----------------------------------------------------------------------


class VerifierUnavailableError(Exception):
    """A verifier's own failure -- provider error, timeout, malformed
    response -- as opposed to the verifier running successfully and
    finding a claim unsupported. Callers must map this to
    `grounding_status = "verification_failed"`, never silently treat it as
    any other status: see the Release 2 plan's "never silently claim
    verification" constraint. Was originally sketched as a placeholder in
    benchmarks/grounding/verifier_fixtures.py ahead of PR 4 implementing a
    real verifier; this is now the one definition both the fixtures and
    postgres_graph_rag.model_verifier.ModelEntailmentVerifier raise/catch.
    """


@runtime_checkable
class Verifier(Protocol):
    """What a verification backend implements. Structurally (not
    nominally) typed -- benchmarks/grounding/verifier_fixtures.py's stub
    verifiers satisfy this without importing from postgres_graph_rag at
    all, which is deliberate: those fixtures predate this contract and are
    meant to be usable by any future verifier implementation's tests,
    inside or outside this package.

    A real implementation is expected to batch every claim for one answer
    into a single provider call (see the Release 2 plan's "one bounded
    verification call per answer, not one provider call per claim"
    constraint) rather than iterating `claims` and calling out per item.
    """

    async def verify(self, claims: Sequence[AnswerClaim]) -> List[ClaimVerification]:
        ...


# ----------------------------------------------------------------------
# Grounding modes (PR 5 wires these into the answer path; defined here so
# the type exists ahead of that wiring)
# ----------------------------------------------------------------------

GroundingMode = Literal["citation_only", "verified", "verified_strict"]
_VALID_GROUNDING_MODES = frozenset(GroundingMode.__args__)  # type: ignore[attr-defined]


def validate_grounding_mode(mode: str) -> GroundingMode:
    if mode not in _VALID_GROUNDING_MODES:
        raise ValueError(f"grounding_mode must be one of {sorted(_VALID_GROUNDING_MODES)}, got {mode!r}")
    return mode  # type: ignore[return-value]
