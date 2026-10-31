"""
client.py — LLM API client wrapper for Featherless AI.

Provides an abstraction over the Featherless AI API using the
OpenAI-compatible AsyncOpenAI client.
"""

from __future__ import annotations

import logging
from typing import Any
from openai import AsyncOpenAI, OpenAIError

try:
    from backend.llm.config import LLMSettings, get_settings
except ModuleNotFoundError:
    from llm.config import LLMSettings, get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around the Featherless AI API (OpenAI-compatible)."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self._settings = settings or get_settings()

        if not self._settings.llm_api_key or not self._settings.llm_api_key.strip():
            raise ValueError(
                "LLM API key is missing. Please set LLM_API_KEY in your .env file or environment variables."
            )

        if not self._settings.llm_model_name or not self._settings.llm_model_name.strip():
            raise ValueError(
                "LLM model name is missing. Please set LLM_MODEL_NAME in your .env file or environment variables."
            )

        self._client = AsyncOpenAI(
            api_key=self._settings.llm_api_key,
            base_url=self._settings.llm_base_url,
        )
        self._last_usage = None
        self._last_metadata: dict[str, Any] = {}

    async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send *prompt* to the LLM and return the generated text.

        Args:
            prompt: Text prompt to send to the model.
            system_prompt: Optional custom system prompt.

        Returns:
            The generated text string from the model.

        Raises:
            ValueError: If the prompt is empty.
            OpenAIError: If the API request fails.
            RuntimeError: If the model response contains no choices.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        sys_content = system_prompt or (
            "You are a precise scientific research information extraction engine. "
            "Output ONLY the final complete, valid JSON object. "
            "Do NOT include any reasoning, explanation, or commentary."
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._settings.llm_model_name,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self._settings.llm_max_tokens,
                temperature=self._settings.llm_temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            if not response.choices:
                raise RuntimeError("Received an empty response with no choices from LLM provider.")

            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason

            # Check for separate reasoning-related fields (boolean only; do NOT log content)
            msg = choice.message
            has_reasoning = bool(
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or (
                    getattr(msg, "model_extra", None)
                    and isinstance(msg.model_extra, dict)
                    and any(
                        "reason" in k.lower() or "think" in k.lower()
                        for k in msg.model_extra
                        if msg.model_extra[k]
                    )
                )
            )

            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else 0

            self._last_usage = response.usage
            self._last_metadata = {
                "model_name": self._settings.llm_model_name,
                "max_tokens": self._settings.llm_max_tokens,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "visible_content_length": len(content),
                "has_reasoning_fields": has_reasoning,
            }

            # Safe diagnostic logging: NO api keys, NO private reasoning content
            logger.info(
                "LLM call diagnostics: model=%s, max_tokens=%d, finish_reason=%s, "
                "prompt_tokens=%d, completion_tokens=%d, visible_content_length=%d, "
                "has_reasoning_fields=%s",
                self._settings.llm_model_name,
                self._settings.llm_max_tokens,
                finish_reason,
                prompt_tokens,
                completion_tokens,
                len(content),
                has_reasoning,
            )

            return content

        except OpenAIError as exc:
            logger.error("Featherless AI API error during generation: %s", exc)
            raise

    @property
    def last_usage(self) -> Any:
        """Return the token usage from the most recent completion, if available."""
        return getattr(self, "_last_usage", None)

    @property
    def last_metadata(self) -> dict[str, Any]:
        """Return diagnostic metadata from the most recent completion."""
        return getattr(self, "_last_metadata", {})

