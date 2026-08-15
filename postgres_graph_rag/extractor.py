import contextvars
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import openai
from google import genai
from google.genai import types
from .models import ProviderConfig

# Token usage from the most recently completed call on *this concurrent
# task*, normalized across providers to {"prompt_tokens",
# "completion_tokens", "total_tokens"} (ints), or None if the last call
# didn't report usage.
#
# This is a contextvar, not a plain instance attribute, on purpose:
# add_document() runs multiple extraction calls concurrently (bounded by
# max_concurrent_extractions), all against the same shared LLMExtractor
# instance. A plain `self.last_usage = ...` would be clobbered by whichever
# concurrent call finishes last — verified empirically before choosing this
# fix, not assumed. asyncio.Task copies the current context at creation
# (asyncio.gather wraps each coroutine in one), so each concurrent
# extraction call gets its own isolated value here.
_last_usage_var: contextvars.ContextVar[Optional[Dict[str, int]]] = contextvars.ContextVar(
    "postgres_graph_rag_last_usage", default=None
)


class ExtractionRefusedError(RuntimeError):
    """The LLM did not produce a valid structured extraction (refusal, safety
    block, or malformed response) — distinct from a genuine "no facts found"."""


class Triplet(BaseModel):
    subject: str = Field(
        ..., description="The entity that is the subject of the relationship"
    )
    predicate: str = Field(
        ..., description="The relationship between the subject and the object"
    )
    object: str = Field(
        ..., description="The entity that is the object of the relationship"
    )

class ExtractionResult(BaseModel):
    triplets: List[Triplet]

class LLMExtractor:
    def __init__(
        self,
        config: ProviderConfig,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        self.config = config
        self.openai_client = None
        self.google_client = None

        if openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        if google_api_key:
            # Use the .aio attribute for async operations
            self.google_client = genai.Client(api_key=google_api_key).aio

    @property
    def last_usage(self) -> Optional[Dict[str, int]]:
        """Token usage for the most recently completed call *on the
        current asyncio task* — see the module-level contextvar comment for
        why this isn't a plain instance attribute. Read it immediately
        after awaiting a call on the same task."""
        return _last_usage_var.get()

    async def extract_triplets(self, text: str) -> List[Triplet]:
        """Extracts entities and relationships from text using the configured LLM."""
        prompt = (
            "You are an expert knowledge graph extractor. Your task is to decompose the given text "
            "into atomic subject-predicate-object triplets.\n\n"
            "Guidelines:\n"
            "1. Entities (Subject/Object): Use proper nouns or specific concepts. Avoid pronouns (he, she, it, they).\n"
            "2. Predicates: Use short, active verbs or clear relationship terms (e.g., 'works_at', 'developed', 'is_located_in').\n"
            "3. Atomicity: Each triplet must represent a single, distinct fact.\n"
            "4. Normalization: Clean up entity names (e.g., 'Apple Inc.' and 'Apple' should be 'Apple').\n"
            "5. Context: Only extract facts explicitly stated in the text.\n\n"
            "Return a JSON list of triplets with the keys: 'subject', 'predicate', 'object'."
        )

        model = self.config["extraction_model"]
        if "gpt" in model and self.openai_client:
            return await self._extract_openai(text, prompt)
        elif "gemini" in model and self.google_client:
            return await self._extract_google(text, prompt)
        else:
            raise ValueError(f"Model {model} not supported or API key missing.")

    def _set_openai_usage(self, usage) -> None:
        if usage is None:
            _last_usage_var.set(None)
            return
        _last_usage_var.set({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": usage.total_tokens,
        })

    def _set_google_usage(self, usage_metadata) -> None:
        if usage_metadata is None:
            _last_usage_var.set(None)
            return
        _last_usage_var.set({
            "prompt_tokens": usage_metadata.prompt_token_count or 0,
            "completion_tokens": usage_metadata.candidates_token_count or 0,
            "total_tokens": usage_metadata.total_token_count or 0,
        })

    async def _extract_openai(self, text: str, prompt: str) -> List[Triplet]:
        completion = await self.openai_client.beta.chat.completions.parse(
            model=self.config["extraction_model"],
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            response_format=ExtractionResult,
        )
        self._set_openai_usage(completion.usage)
        return completion.choices[0].message.parsed.triplets

    async def _extract_google(self, text: str, prompt: str) -> List[Triplet]:
        response = await self.google_client.models.generate_content(
            model=self.config["extraction_model"],
            contents=[prompt, text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
            ),
        )
        self._set_google_usage(response.usage_metadata)

        # A refusal/safety-block/malformed response must raise (so the
        # caller's retry logic and lease/cache bookkeeping treat it as a
        # failed extraction), never be silently treated as "the model
        # looked and found zero facts" — a legitimate empty result is
        # ExtractionResult(triplets=[]), a truthy parsed object, which
        # still reaches the `return response.parsed.triplets` below.
        block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else None
        if block_reason:
            raise ExtractionRefusedError(
                f"Gemini blocked the entire prompt before generation: block_reason={block_reason!r}"
            )

        if not response.candidates:
            raise ExtractionRefusedError("Gemini returned no candidates (empty response).")

        finish_reason = response.candidates[0].finish_reason
        if finish_reason != types.FinishReason.STOP:
            raise ExtractionRefusedError(
                f"Gemini did not cleanly stop: finish_reason={finish_reason!r}"
            )

        if not response.parsed:
            raise ExtractionRefusedError(
                "Gemini finished cleanly but produced no parsed structured output "
                "(schema-conforming JSON missing)."
            )

        return response.parsed.triplets

    async def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """Plain text completion, reusing the same extraction_model/clients
        already configured — used for community summarization (v0.4),
        which needs free-text generation rather than structured triplet
        extraction."""
        model = self.config["extraction_model"]
        if "gpt" in model and self.openai_client:
            request = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            # GPT-5 models reject the legacy `max_tokens` parameter. Keep
            # compatibility with older OpenAI chat models while using the
            # current completion-token parameter for GPT-5+.
            if model.startswith("gpt-5"):
                request["max_completion_tokens"] = max_tokens
            else:
                request["max_tokens"] = max_tokens
            completion = await self.openai_client.chat.completions.create(**request)
            self._set_openai_usage(completion.usage)
            return completion.choices[0].message.content or ""
        elif "gemini" in model and self.google_client:
            response = await self.google_client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(max_output_tokens=max_tokens),
            )
            self._set_google_usage(response.usage_metadata)
            return response.text or ""
        raise ValueError(f"Model {model} not supported or API key missing.")

    async def get_embedding(
        self, text: str | List[str]
    ) -> List[float] | List[List[float]]:
        """
        Retrieves embeddings for one or more texts.
        Returns a single list of floats if a single string is provided,
        or a list of lists if a list of strings is provided.
        """
        model = self.config["embedding_model"]
        is_list = isinstance(text, list)
        input_texts = text if is_list else [text]

        if self.openai_client and "text-embedding" in model:
            # We use a separate variable to help the type checker
            openai_resp = await self.openai_client.embeddings.create(
                input=input_texts, model=model
            )
            self._set_openai_usage(openai_resp.usage)
            embeddings = [item.embedding for item in openai_resp.data]
            return embeddings if is_list else embeddings[0]

        elif self.google_client and (
            "text-embedding" in model or "embedding" in model
        ):
            # We use a separate variable to help the type checker
            google_resp = await self.google_client.models.embed_content(
                model=model, contents=input_texts
            )
            # Google's EmbedContentResponse carries no usage/token field at
            # all (verified against the installed SDK's response type) —
            # this is a real provider/SDK limitation, not something skipped.
            _last_usage_var.set(None)
            # Google's embed_content returns an object with an 'embeddings' list
            embeddings = [e.values for e in google_resp.embeddings]
            return embeddings if is_list else embeddings[0]

        raise ValueError("API key missing or provider mismatch for embeddings.")
