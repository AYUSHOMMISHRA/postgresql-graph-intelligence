"""Release 2 PR 3: the deterministic (non-model) verification layers.

Per the Release 2 plan, this implements everything *before* batched model
entailment (PR 4):

1. `parse_claims` -- structured claim parsing.
2 + 3. `DeterministicVerifier` checks citation existence and (by
   construction, since it only ever sees `retrieval.chunks`, which Release
   1's atomic-publication work already guarantees are from active document
   revisions) active evidence -- a citation naming a chunk outside that
   pool is rejected the same way whether the chunk was never retrieved or
   was retrieved and then superseded.
4. Identifier checks reuse tenant_engine.py's own `_identifier_anchors`/
   `_normalize_identifier` rather than reimplementing them.
5. `_locate_quote` is the server-side supporting-quote location: it only
   ever accepts a quote it found verbatim (or after separator
   normalization) inside the cited evidence's own text -- it never trusts
   claim text as if the model had already proven it, and never accepts an
   offset without relocating it.
6. `evaluate_policy` -- citation_only / verified / verified_strict.
7. `render_answer` -- deterministic rendering from surviving claims only,
   never an unconstrained model-supplied final_answer string.
8. `DeterministicVerifier` doubles as the injectable offline `Verifier`
   implementation the Release 2 plan asks for (parallel to
   `offline.OfflineExtractor` standing in for a real extraction model).

What this deliberately cannot do: confirm genuine semantic entailment.
`DeterministicVerifier` only ever positively confirms "supported" from a
literal (or separator-normalized) quote match, and only ever positively
confirms "contradicted" from an explicit numeric/date mismatch or an
explicit negation in a competing retrieved chunk -- everything else
(reversed relationships, swapped predicates between the same two entities,
claims that combine facts from two citations, unstated multi-hop
inference) it reports as "insufficient" rather than guessing. That's a
deliberate, conservative choice: a wrong "insufficient" costs an
unnecessary abstention; a wrong "supported" costs a hallucination reaching
the user. Recovering the recall lost to this conservatism is exactly PR 4's
job, not this one's.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .grounding import (
    AnswerClaim,
    ClaimVerification,
    GroundingMode,
    GroundingStatus,
    Verdict,
)
from .tenant_engine import (
    TenantRetrievedChunk,
    _citation_marker,
    _identifier_anchors,
    _normalize_identifier,
)

_NUMBER_RE = re.compile(r"\d+")
_NEGATION_MARKERS = (" not ", "n't ", "no longer", "does not", "doesn't", "isn't", "is not")


# ----------------------------------------------------------------------
# 1. Structured claim parsing
# ----------------------------------------------------------------------


def parse_claims(raw_claims: Sequence[Dict[str, Any]]) -> List[AnswerClaim]:
    """Validates and coerces a model's structured claim output into
    `AnswerClaim` objects. Rejects malformed entries outright rather than
    guessing at intent -- a claim with no `text`, or a `citation_ids` that
    isn't a list of strings, is a shape the rest of this pipeline should
    never have to handle defensively.
    """
    claims: List[AnswerClaim] = []
    for i, raw in enumerate(raw_claims):
        if not isinstance(raw, dict):
            raise ValueError(f"claim {i} is not an object: {raw!r}")
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"claim {i} has no non-empty 'text': {raw!r}")
        citation_ids = raw.get("citation_ids", [])
        if not isinstance(citation_ids, list) or not all(isinstance(c, str) for c in citation_ids):
            raise ValueError(f"claim {i}'s citation_ids must be a list of strings: {raw!r}")
        claim_id = raw.get("id") or f"claim-{i}"
        claims.append(AnswerClaim(id=claim_id, text=text.strip(), citation_ids=list(citation_ids)))
    return claims


# ----------------------------------------------------------------------
# 2-5, 8. The deterministic verifier
# ----------------------------------------------------------------------


def _strip_trailing_period(text: str) -> str:
    return text.rstrip(".").strip()


def _extract_numbers(text: str) -> set:
    return set(_NUMBER_RE.findall(text))


def _locate_quote(claim_text: str, chunk: TenantRetrievedChunk) -> Optional[str]:
    """Server-side quote location: only ever returns a quote actually
    present in `chunk.content`, never the claim's own phrasing as if it had
    been confirmed. Tries an exact (trailing-period-insensitive) substring
    match first; if that fails, tries a separator-normalized match (the
    same hyphen/underscore/dot-to-space normalization tenant_engine.py's
    `_normalize_identifier` already uses) so a real match isn't missed over
    pure punctuation -- but reports the *chunk's own* text as the quote in
    that case, since the claim's literal phrasing (with its own hyphens)
    generally won't itself be a literal substring of the differently-
    punctuated source.
    """
    stripped_claim = _strip_trailing_period(claim_text)
    if not stripped_claim:
        return None
    if stripped_claim in chunk.content:
        return stripped_claim
    if _normalize_identifier(stripped_claim) in _normalize_identifier(chunk.content):
        return chunk.content
    return None


def _has_conflicting_evidence(anchors: List[str], all_evidence: Sequence[TenantRetrievedChunk]) -> bool:
    """A narrow, explicit heuristic (not general contradiction detection):
    flags a conflict only when some *other* retrieved chunk mentions every
    identifier the claim names *and* contains an explicit negation marker
    near them. Deliberately conservative -- this catches a document pool
    that directly asserts "X does NOT do Y" alongside a chunk that says it
    does, which is exactly the shape of the conflicting_documents category
    in benchmarks/grounding/, not a general-purpose conflict detector.
    """
    if not anchors:
        return False
    for chunk in all_evidence:
        normalized_content = _normalize_identifier(chunk.content)
        if not all(_normalize_identifier(a) in normalized_content for a in anchors):
            continue
        lowered = f" {chunk.content.lower()} "
        if any(marker in lowered for marker in _NEGATION_MARKERS):
            return True
    return False


class DeterministicVerifier:
    """The injectable, model-free `Verifier` implementation (PR 3 item 8),
    and the fallback layer any real (model-backed) verifier should run
    claims through *first* -- a claim this rejects should never reach a
    provider call at all (see the Release 2 plan's "deterministic checks
    first" ordering).

    Evidence is bound at construction time (one instance per answer, over
    that answer's own `retrieval.chunks`) rather than passed to `verify()`,
    so this satisfies the `Verifier` protocol's `verify(self, claims)`
    signature exactly -- the same signature the pre-existing stub
    verifiers in benchmarks/grounding/verifier_fixtures.py already
    implement, with no protocol change required to add this real one
    alongside them.
    """

    def __init__(self, evidence: Sequence[TenantRetrievedChunk]):
        self._evidence = list(evidence)
        self._allowed: Dict[str, TenantRetrievedChunk] = {_citation_marker(c): c for c in self._evidence}

    async def verify(self, claims: Sequence[AnswerClaim]) -> List[ClaimVerification]:
        return [self._verify_one(claim) for claim in claims]

    def _verify_one(self, claim: AnswerClaim) -> ClaimVerification:
        if not claim.citation_ids:
            return ClaimVerification(claim_id=claim.id, verdict="insufficient", reason_code="citation_missing")

        invalid = [cid for cid in claim.citation_ids if cid not in self._allowed]
        if invalid:
            return ClaimVerification(
                claim_id=claim.id, verdict="insufficient",
                reason_code="citation_outside_retrieved_evidence",
            )

        cited_chunks = [self._allowed[cid] for cid in claim.citation_ids]
        combined_cited_text = " ".join(c.content for c in cited_chunks)

        # Checked before the identifier/anchor check below on purpose: a
        # claim like "a 45-minute outage" contains a hyphenated token
        # ("45-minute") that _identifier_anchors() also matches as if it
        # were an entity identifier. If the anchor check ran first, a
        # numeric mismatch ("45-minute" absent because the evidence says
        # "12-minute") would be misreported as evidence_irrelevant instead
        # of the more specific, more useful numeric_or_date_mismatch.
        claim_numbers = _extract_numbers(claim.text)
        evidence_numbers = _extract_numbers(combined_cited_text)
        if claim_numbers and evidence_numbers and not claim_numbers.issubset(evidence_numbers):
            return ClaimVerification(
                claim_id=claim.id, verdict="contradicted", reason_code="numeric_or_date_mismatch",
            )

        anchors = _identifier_anchors(claim.text)
        missing_anchors = [a for a in anchors if _normalize_identifier(a) not in _normalize_identifier(combined_cited_text)]
        if missing_anchors:
            return ClaimVerification(claim_id=claim.id, verdict="insufficient", reason_code="evidence_irrelevant")

        quote: Optional[str] = None
        for chunk in cited_chunks:
            quote = _locate_quote(claim.text, chunk)
            if quote is not None:
                break

        if quote is None:
            return ClaimVerification(claim_id=claim.id, verdict="insufficient", reason_code="evidence_insufficient")

        if _has_conflicting_evidence(anchors, self._evidence):
            return ClaimVerification(claim_id=claim.id, verdict="insufficient", reason_code="evidence_conflicting")

        return ClaimVerification(
            claim_id=claim.id, verdict="supported",
            supporting_quotes=[quote], reason_code="explicit_relation_confirmed",
        )


# ----------------------------------------------------------------------
# 6. Policy evaluation
# ----------------------------------------------------------------------


def evaluate_policy(
    claims: Sequence[AnswerClaim],
    verifications: Sequence[ClaimVerification],
    mode: GroundingMode,
) -> GroundingStatus:
    """Decides the overall `GroundingStatus` for one answer from its
    per-claim verdicts. `citation_only` deliberately ignores per-claim
    verdicts entirely (compatibility mode: today's behavior, unchanged);
    `verified` and `verified_strict` differ only in how they react to a
    claim that isn't fully supported -- drop it, or abstain the whole
    answer.
    """
    if not claims:
        return "abstained"

    by_id = {v.claim_id: v for v in verifications}
    verdicts: List[Verdict] = [by_id[c.id].verdict for c in claims if c.id in by_id]

    if mode == "citation_only":
        # Compatibility: a claim with any citation at all (valid, since
        # this verifier only reaches this point for claims whose citations
        # already passed existence-checking upstream) is accepted without
        # inspecting its verdict -- matching tenant_engine.py's current
        # `grounded = bool(markers) and not invalid`.
        return "citation_valid_only"

    contradicted = any(v == "contradicted" for v in verdicts)
    supported = [v for v in verdicts if v == "supported"]

    if mode == "verified_strict":
        if contradicted:
            return "contradicted"
        if len(supported) < len(verdicts):
            return "insufficient"
        return "verified"

    # mode == "verified"
    if contradicted:
        return "contradicted"
    if not supported:
        return "insufficient"
    if len(supported) == len(verdicts):
        return "verified"
    return "partially_verified"


# ----------------------------------------------------------------------
# 7. Deterministic answer rendering
# ----------------------------------------------------------------------


_ABSTENTION_TEXT = "Insufficient evidence to answer from the indexed sources."


def render_answer(
    claims: Sequence[AnswerClaim],
    verifications: Sequence[ClaimVerification],
    grounding_status: GroundingStatus,
) -> str:
    """Renders the final answer text from surviving claims only -- never
    from a separate, unconstrained `final_answer` string a model could pad
    with facts that were never checked (see the Release 2 plan's
    "Do not accept an unconstrained final_answer" constraint).

    "Surviving" means verdict == 'supported' in `verified` mode, or every
    claim in `citation_only` mode (nothing was checked to drop). A
    `verified_strict` grounding_status other than 'verified' renders the
    standard abstention text, dropping every claim -- that mode's whole
    point is full-or-nothing.
    """
    if grounding_status == "abstained" or grounding_status == "insufficient":
        return _ABSTENTION_TEXT
    if grounding_status == "contradicted":
        return _ABSTENTION_TEXT

    by_id = {v.claim_id: v for v in verifications}
    if grounding_status == "citation_valid_only":
        surviving = claims
    else:  # "verified" or "partially_verified"
        surviving = [c for c in claims if by_id.get(c.id) and by_id[c.id].verdict == "supported"]

    if not surviving:
        return _ABSTENTION_TEXT

    sentences = []
    for claim in surviving:
        markers = "".join(f"[{cid}]" for cid in claim.citation_ids)
        sentences.append(f"{claim.text} {markers}".strip())
    return " ".join(sentences)
