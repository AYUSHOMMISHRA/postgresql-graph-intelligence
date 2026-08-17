"""Fixtures for a verifier's failure-handling path (the Release 2 plan's
"verifier unavailable -> return verification_failed; never silently claim
verification"). `postgres_graph_rag.model_verifier.ModelEntailmentVerifier`
(PR 4) is the real verifier these exist to test against; this module
provides stubs that always fail in each of the ways a real provider call
can fail, for any verifier's test suite to import and confirm against
rather than each test reinventing its own mock.

Usage:

    from benchmarks.grounding.verifier_fixtures import RaisingVerifier
    from postgres_graph_rag.grounding import VerifierUnavailableError

    async def test_provider_error_reports_verification_failed():
        with pytest.raises(VerifierUnavailableError):
            await RaisingVerifier().verify([])
"""
from __future__ import annotations

import asyncio
from typing import Any, List

from postgres_graph_rag.grounding import VerifierUnavailableError

__all__ = [
    "VerifierUnavailableError",  # re-exported for callers written against this module before PR 2/4 landed
    "RaisingVerifier",
    "TimingOutVerifier",
    "MalformedResponseVerifier",
]


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
