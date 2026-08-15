"""Release 2 PR 4: batched model entailment -- the first verifier that
makes a real provider call, layered strictly on top of PR 3's deterministic
checks rather than replacing them.

Design constraints from the Release 2 plan, and where each is enforced:

- "One bounded verification call per answer, not one provider call per
  claim" -> `ModelEntailmentVerifier.verify()` makes exactly one
  `extractor.verify_claims()` call covering every claim this instance was
  asked to check (never one call per claim), and only for the claims
  DeterministicVerifier couldn't already decide (see below) -- both a cost
  control and a literal reading of "bounded".
- "Deterministic checks first" -> every claim is run through
  `DeterministicVerifier` before this even considers a model call; a claim
  the deterministic layer could already confirm/reject (citation missing,
  citation invalid, irrelevant evidence, numeric mismatch, a located quote)
  keeps that verdict untouched and never reaches the model at all.
- "Do not trust model-supplied character offsets; request quotes and
  locate them server-side" -> the model is asked for `supporting_quote`
  text, never an offset, and `_locate_quote()` (imported from PR 3's
  `verification.py`, not reimplemented) re-locates it in the actual cited
  evidence; a quote the model claims but that isn't literally (or
  separator-normalized) present is downgraded, never trusted.
- "Do not accept an unconstrained final_answer" -> this module has no
  concept of a final answer at all; it only ever returns per-claim
  `ClaimVerification`s for `verification.render_answer()` to build from.
- "Treat model confidence as uncalibrated metadata, not probability" ->
  `ModelVerdict.confidence` is passed through into `ClaimVerification`
  unmodified and unranked against anything; nothing here thresholds on it.
- "Verifier unavailable -> return verification_failed; never silently
  claim verification" -> any exception from the provider call (including
  `ExtractionRefusedError`, a malformed/refused response) is re-raised as
  `VerifierUnavailableError` for the whole batch, rather than partially
  substituting a default verdict for the claims that were mid-flight.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

from .extractor import LLMExtractor, ModelVerdict
from .grounding import AnswerClaim, ClaimVerification, VerifierUnavailableError
from .tenant_engine import TenantRetrievedChunk, _citation_marker
from .verification import DeterministicVerifier, _locate_quote


class VerificationTelemetry:
    """Cost/latency/token disclosure for one `verify()` call -- the
    "measured and disclosed" (not gated) metrics from gates.py. `None`
    fields mean "not applicable to this call" (e.g. no model call was made
    because every claim was resolved deterministically), not "zero"."""

    def __init__(self) -> None:
        self.latency_ms: Optional[float] = None
        self.usage: Optional[Dict[str, int]] = None
        self.model_call_made: bool = False

    def estimated_cost_usd(
        self, cost_per_1k_prompt_tokens: float, cost_per_1k_completion_tokens: float,
    ) -> Optional[float]:
        """Computed on demand from actual token usage and caller-supplied
        rates, rather than a hardcoded price table baked into this
        library -- provider prices change, and a stale hardcoded number
        would be worse than making the caller supply a current one."""
        if not self.usage:
            return None
        return (
            self.usage.get("prompt_tokens", 0) / 1000 * cost_per_1k_prompt_tokens
            + self.usage.get("completion_tokens", 0) / 1000 * cost_per_1k_completion_tokens
        )


def _format_claims_prompt(
    claims: Sequence[AnswerClaim], evidence_by_marker: Dict[str, TenantRetrievedChunk],
) -> str:
    lines = []
    for claim in claims:
        lines.append(f"Claim {claim.id}: {claim.text}")
        for cid in claim.citation_ids:
            chunk = evidence_by_marker.get(cid)
            if chunk is not None:
                lines.append(f"  Evidence {cid}: {chunk.content}")
        lines.append("")
    return "\n".join(lines)


class ModelEntailmentVerifier:
    """The injectable, provider-backed `Verifier` implementation. Evidence
    is bound at construction (one instance per answer, same convention as
    `DeterministicVerifier`), so `verify(claims)` matches the `Verifier`
    protocol's signature exactly.
    """

    def __init__(self, extractor: LLMExtractor, evidence: Sequence[TenantRetrievedChunk]):
        self._extractor = extractor
        self._evidence = list(evidence)
        self._deterministic = DeterministicVerifier(evidence=evidence)
        self._evidence_by_marker: Dict[str, TenantRetrievedChunk] = {
            _citation_marker(c): c for c in self._evidence
        }
        self.last_telemetry = VerificationTelemetry()

    async def verify(self, claims: Sequence[AnswerClaim]) -> List[ClaimVerification]:
        self.last_telemetry = VerificationTelemetry()
        claims = list(claims)

        deterministic_results = await self._deterministic.verify(claims)
        by_id: Dict[str, ClaimVerification] = {v.claim_id: v for v in deterministic_results}

        # Only claims the deterministic layer genuinely couldn't decide
        # (not "citation missing", not "irrelevant", not a positively
        # confirmed number mismatch or quote match) go to the model --
        # this is both the cost control and the "deterministic checks
        # first" ordering from the plan.
        needs_model = [c for c in claims if by_id[c.id].reason_code == "evidence_insufficient"]

        if not needs_model:
            return [by_id[c.id] for c in claims]

        prompt = _format_claims_prompt(needs_model, self._evidence_by_marker)
        started = time.perf_counter()
        try:
            model_verdicts = await self._extractor.verify_claims(prompt)
        except Exception as exc:
            raise VerifierUnavailableError(
                f"model entailment verification failed: {exc}"
            ) from exc
        self.last_telemetry.latency_ms = (time.perf_counter() - started) * 1000
        self.last_telemetry.usage = self._extractor.last_usage
        self.last_telemetry.model_call_made = True

        # A model verdict naming a claim_id we never asked about is simply
        # ignored -- it can't be attached to anything this caller is
        # rendering an answer from.
        model_verdicts_by_id = {v.claim_id: v for v in model_verdicts}
        for claim in needs_model:
            model_verdict = model_verdicts_by_id.get(claim.id)
            by_id[claim.id] = self._resolve_model_verdict(claim, model_verdict)

        return [by_id[c.id] for c in claims]

    def _resolve_model_verdict(
        self, claim: AnswerClaim, model_verdict: Optional[ModelVerdict],
    ) -> ClaimVerification:
        if model_verdict is None:
            # The model didn't return a verdict for this claim at all --
            # safe default, not a crash, but also not silently "supported".
            return ClaimVerification(claim_id=claim.id, verdict="insufficient", reason_code="evidence_insufficient")

        if model_verdict.verdict == "insufficient":
            return ClaimVerification(claim_id=claim.id, verdict="insufficient", reason_code="evidence_insufficient")

        # 'supported' and 'contradicted' both require a quote actually
        # located in the cited evidence -- an unverified model assertion
        # of either is downgraded, not trusted.
        cited_chunks = [
            self._evidence_by_marker[cid] for cid in claim.citation_ids if cid in self._evidence_by_marker
        ]
        located = None
        if model_verdict.supporting_quote:
            for chunk in cited_chunks:
                if model_verdict.supporting_quote in chunk.content:
                    located = model_verdict.supporting_quote
                    break
                if _locate_quote(model_verdict.supporting_quote, chunk) is not None:
                    located = chunk.content
                    break

        if located is None:
            return ClaimVerification(
                claim_id=claim.id, verdict="insufficient", reason_code="quote_not_found_verbatim",
            )

        # Distinct reason codes from the deterministic layer's
        # "explicit_relation_confirmed"/"numeric_or_date_mismatch": those
        # imply a *specific* mechanical check found the answer, whereas
        # this verdict came from the model's own judgment (server-side
        # quote validation only confirms the cited text exists, not which
        # semantic check the model actually applied).
        return ClaimVerification(
            claim_id=claim.id,
            verdict=model_verdict.verdict,
            supporting_quotes=[located],
            reason_code=(
                "model_entailment_supported" if model_verdict.verdict == "supported"
                else "model_entailment_contradicted"
            ),
            confidence=model_verdict.confidence,
        )
