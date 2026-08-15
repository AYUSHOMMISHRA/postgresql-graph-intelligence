"""Unit tests for LLMExtractor's failure-mode handling — mocked, no real
API keys/network calls, no POSTGRES_URL dependency."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import types

from postgres_graph_rag.extractor import (
    ExtractionRefusedError,
    ExtractionResult,
    LLMExtractor,
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
