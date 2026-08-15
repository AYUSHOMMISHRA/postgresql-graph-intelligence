"""Unit tests for LLMExtractor's failure-mode handling — mocked, no real
API keys/network calls, no POSTGRES_URL dependency."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types

from postgres_graph_rag.extractor import (
    ExtractionRefusedError,
    ExtractionResult,
    LLMExtractor,
    ModelVerdict,
    VerificationBatchResult,
)
from postgres_graph_rag.models import GOOGLE_DEFAULT_CONFIG, OPENAI_DEFAULT_CONFIG


def _extractor_openai() -> LLMExtractor:
    ext = LLMExtractor(config=OPENAI_DEFAULT_CONFIG, openai_api_key="test")
    ext.openai_client = AsyncMock()
    return ext


def _extractor_google() -> LLMExtractor:
    ext = LLMExtractor(config=GOOGLE_DEFAULT_CONFIG, google_api_key="test")
    ext.google_client = AsyncMock()
    return ext


@pytest.mark.asyncio
async def test_openai_parse_none_raises():
    ext = _extractor_openai()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(parsed=None))]
    ext.openai_client.beta.chat.completions.parse = AsyncMock(return_value=completion)

    with pytest.raises(AttributeError):
        await ext._extract_openai("text", "prompt")


@pytest.mark.asyncio
async def test_google_safety_block_raises_extraction_refused():
    ext = _extractor_google()
    response = MagicMock()
    response.parsed = None
    response.usage_metadata = None
    response.prompt_feedback = None
    response.candidates = [MagicMock(finish_reason=types.FinishReason.SAFETY)]
    ext.google_client.models.generate_content = AsyncMock(return_value=response)

    with pytest.raises(ExtractionRefusedError):
        await ext._extract_google("text", "prompt")


@pytest.mark.asyncio
async def test_google_whole_prompt_block_raises_extraction_refused():
    ext = _extractor_google()
    response = MagicMock()
    response.parsed = None
    response.usage_metadata = None
    response.prompt_feedback = MagicMock(block_reason="SAFETY")
    response.candidates = []
    ext.google_client.models.generate_content = AsyncMock(return_value=response)

    with pytest.raises(ExtractionRefusedError):
        await ext._extract_google("text", "prompt")


@pytest.mark.asyncio
async def test_google_clean_stop_but_unparsed_raises_extraction_refused():
    ext = _extractor_google()
    response = MagicMock()
    response.parsed = None
    response.usage_metadata = None
    response.prompt_feedback = None
    response.candidates = [MagicMock(finish_reason=types.FinishReason.STOP)]
    ext.google_client.models.generate_content = AsyncMock(return_value=response)

    with pytest.raises(ExtractionRefusedError):
        await ext._extract_google("text", "prompt")


@pytest.mark.asyncio
async def test_google_clean_stop_empty_triplets_returns_empty_list():
    ext = _extractor_google()
    response = MagicMock()
    response.parsed = ExtractionResult(triplets=[])
    response.usage_metadata = None
    response.prompt_feedback = None
    response.candidates = [MagicMock(finish_reason=types.FinishReason.STOP)]
    ext.google_client.models.generate_content = AsyncMock(return_value=response)

    result = await ext._extract_google("text", "prompt")
    assert result == []


@pytest.mark.asyncio
async def test_verify_openai_parses_verdicts():
    ext = _extractor_openai()
    completion = MagicMock()
    completion.usage = None
    completion.choices = [MagicMock(message=MagicMock(parsed=VerificationBatchResult(
        verdicts=[ModelVerdict(claim_id="c1", verdict="supported", supporting_quote="X depends on Y", confidence=0.9)],
    )))]
    ext.openai_client.beta.chat.completions.parse = AsyncMock(return_value=completion)

    result = await ext._verify_openai("claims prompt")
    assert result == [ModelVerdict(claim_id="c1", verdict="supported", supporting_quote="X depends on Y", confidence=0.9)]


@pytest.mark.asyncio
async def test_verify_google_safety_block_raises_extraction_refused():
    ext = _extractor_google()
    response = MagicMock()
    response.parsed = None
    response.usage_metadata = None
    response.prompt_feedback = None
    response.candidates = [MagicMock(finish_reason=types.FinishReason.SAFETY)]
    ext.google_client.models.generate_content = AsyncMock(return_value=response)

    with pytest.raises(ExtractionRefusedError):
        await ext._verify_google("claims prompt")


@pytest.mark.asyncio
async def test_verify_google_whole_prompt_block_raises_extraction_refused():
    ext = _extractor_google()
    response = MagicMock()
    response.parsed = None
    response.usage_metadata = None
    response.prompt_feedback = MagicMock(block_reason="SAFETY")
    response.candidates = []
    ext.google_client.models.generate_content = AsyncMock(return_value=response)

    with pytest.raises(ExtractionRefusedError):
        await ext._verify_google("claims prompt")


@pytest.mark.asyncio
async def test_verify_google_clean_stop_but_unparsed_raises_extraction_refused():
    ext = _extractor_google()
    response = MagicMock()
    response.parsed = None
    response.usage_metadata = None
    response.prompt_feedback = None
    response.candidates = [MagicMock(finish_reason=types.FinishReason.STOP)]
    ext.google_client.models.generate_content = AsyncMock(return_value=response)

    with pytest.raises(ExtractionRefusedError):
        await ext._verify_google("claims prompt")


@pytest.mark.asyncio
async def test_verify_google_parses_verdicts():
    ext = _extractor_google()
    response = MagicMock()
    response.parsed = VerificationBatchResult(
        verdicts=[ModelVerdict(claim_id="c1", verdict="contradicted", supporting_quote="Y depends on X", confidence=0.7)],
    )
    response.usage_metadata = None
    response.prompt_feedback = None
    response.candidates = [MagicMock(finish_reason=types.FinishReason.STOP)]
    ext.google_client.models.generate_content = AsyncMock(return_value=response)

    result = await ext._verify_google("claims prompt")
    assert result == [ModelVerdict(claim_id="c1", verdict="contradicted", supporting_quote="Y depends on X", confidence=0.7)]


@pytest.mark.asyncio
async def test_verify_claims_dispatches_by_model_name():
    ext = _extractor_openai()
    ext._verify_openai = AsyncMock(return_value=[])
    await ext.verify_claims("prompt")
    ext._verify_openai.assert_awaited_once_with("prompt")


@pytest.mark.asyncio
async def test_verify_claims_raises_for_unsupported_model():
    ext = LLMExtractor(config={**OPENAI_DEFAULT_CONFIG, "extraction_model": "some-other-model"})
    with pytest.raises(ValueError, match="not supported or API key missing"):
        await ext.verify_claims("prompt")
