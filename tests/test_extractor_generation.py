from unittest.mock import AsyncMock, MagicMock

import pytest

from postgres_graph_rag.extractor import LLMExtractor
from postgres_graph_rag.models import OPENAI_DEFAULT_CONFIG


@pytest.mark.asyncio
async def test_gpt5_text_generation_uses_completion_token_parameter():
    extractor = LLMExtractor(config=OPENAI_DEFAULT_CONFIG, openai_api_key="test")
    completion = MagicMock(usage=None, choices=[MagicMock(message=MagicMock(content="ok"))])
    extractor.openai_client = MagicMock()
    extractor.openai_client.chat.completions.create = AsyncMock(return_value=completion)

    result = await extractor.generate_text("summarize", max_tokens=123)

    assert result == "ok"
    kwargs = extractor.openai_client.chat.completions.create.call_args.kwargs
    assert kwargs["max_completion_tokens"] == 123
    assert "max_tokens" not in kwargs
