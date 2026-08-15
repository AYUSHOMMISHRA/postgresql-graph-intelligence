"""Integrity tests for the fabrication and verifier-unavailable fixtures
used by a future verifier's test suite (PR 3-5 of the Release 2 plan).
These fixtures don't have a verifier to run against yet -- what's checked
here is that the fixture data itself is well-formed, so it's ready the
moment a verifier exists to consume it.
"""
import asyncio
import json
from pathlib import Path

import pytest

from benchmarks.grounding.fabrication_fixtures import build_fixtures
from benchmarks.grounding.verifier_fixtures import (
    MalformedResponseVerifier,
    RaisingVerifier,
    TimingOutVerifier,
    VerifierUnavailableError,
)

FIXTURES_PATH = Path(__file__).parent.parent / "benchmarks" / "grounding" / "fabrication_fixtures.json"


@pytest.fixture(scope="module")
def fixtures():
    return json.loads(FIXTURES_PATH.read_text())["fixtures"]


def test_fabrication_fixtures_are_reproducible(fixtures):
    assert fixtures == build_fixtures()


def test_fabrication_fixture_ids_are_unique(fixtures):
    ids = [f["id"] for f in fixtures]
    assert len(ids) == len(set(ids))


def test_real_quote_is_a_literal_substring(fixtures):
    for f in fixtures:
        assert f["real_quote"] in f["chunk_content"], (
            f"{f['id']}: real_quote is not a literal substring of chunk_content"
        )


def test_fabricated_quote_is_not_a_literal_substring(fixtures):
    """The whole point of these fixtures: a fabricated quote must NOT be
    findable verbatim, or it wouldn't exercise a verifier's rejection path
    at all."""
    for f in fixtures:
        assert f["fabricated_quote"] not in f["chunk_content"], (
            f"{f['id']}: fabricated_quote is (accidentally) a literal substring of chunk_content"
        )


def test_real_and_fabricated_quotes_differ(fixtures):
    for f in fixtures:
        assert f["real_quote"] != f["fabricated_quote"], f"{f['id']}: real and fabricated quotes are identical"


@pytest.mark.asyncio
async def test_raising_verifier_raises_verifier_unavailable_error():
    with pytest.raises(VerifierUnavailableError):
        await RaisingVerifier().verify([])


@pytest.mark.asyncio
async def test_timing_out_verifier_can_be_cancelled():
    task = asyncio.ensure_future(TimingOutVerifier(delay_seconds=3600).verify([]))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_malformed_response_verifier_returns_non_list_shape():
    result = await MalformedResponseVerifier().verify([])
    assert not isinstance(result, list)
