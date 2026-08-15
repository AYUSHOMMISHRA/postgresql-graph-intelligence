"""Fixtures for a future verifier's failure-handling path (PR 4/5 in the
Release 2 plan: "verifier unavailable -> return verification_failed; never
silently claim verification"). No verifier interface exists yet (that's
PR 2's job -- see the CTO plan's ClaimVerification/VerifiedAnswerResult
types), so this is deliberately minimal: a stub that always fails in each
of the ways a real provider call can fail, for that future verifier's test
suite to import and confirm against, rather than each test reinventing its
own mock.

Usage once a verifier exists (illustrative -- the verifier type doesn't
exist yet, so this can't be exercised end-to-end today):

    from benchmarks.grounding.verifier_fixtures import RaisingVerifier, TimingOutVerifier

    async def test_provider_error_reports_verification_failed():
        result = await answer_with_verifier(question, verifier=RaisingVerifier())
        assert result.grounding_status == "verification_failed"
        assert result.grounding_status != "verified"  # never silently fall back to success
"""
from __future__ import annotations

import asyncio
from typing import Any, List


class VerifierUnavailableError(Exception):
    """Placeholder for the real exception type PR 2's verifier protocol
    should define. Using a distinct exception (not a bare RuntimeError)
    matters: it's what lets calling code distinguish "the verifier itself
    is broken/unreachable" from "the verifier ran and found a problem",
    which must be represented as two different grounding_status values,
    not conflated into one catch-all failure."""


class RaisingVerifier:
    """Simulates a provider call that fails immediately (auth error, 4xx,
    malformed request) -- the simplest failure mode a verifier's caller
    must not swallow into a false "verified" result."""

    async def verify(self, claims: List[Any]) -> List[Any]:
        raise VerifierUnavailableError("simulated: verifier provider rejected the request")


class TimingOutVerifier:
    """Simulates a provider call that never returns -- exercises whatever
    timeout/cancellation handling the real verifier call site needs, as
    opposed to RaisingVerifier's immediate failure."""

    def __init__(self, delay_seconds: float = 3600.0):
        self.delay_seconds = delay_seconds

    async def verify(self, claims: List[Any]) -> List[Any]:
        await asyncio.sleep(self.delay_seconds)
        raise AssertionError("TimingOutVerifier should have been cancelled/timed out before this point")


class MalformedResponseVerifier:
    """Simulates a provider call that succeeds at the transport level but
    returns something that doesn't parse as the expected structured
    verdict shape -- distinct from RaisingVerifier because this failure
    mode is often silent (a caller that doesn't validate the response
    shape can mistake a garbage response for a real verdict)."""

    async def verify(self, claims: List[Any]) -> List[Any]:
        return "not a list of ClaimVerification objects"  # type: ignore[return-value]
