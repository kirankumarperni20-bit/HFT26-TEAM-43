"""
config.py — LLM configuration via environment variables.

Uses pydantic-settings so that every setting can be overridden by an
environment variable (or a .env file) without touching code.

Default values are tuned for Featherless AI, which exposes an
OpenAI-compatible REST API at https://api.featherless.ai/v1.
Point ``llm_base_url`` at any other OpenAI-compatible provider
(e.g. vLLM, Ollama, OpenRouter) to switch without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class LLMSettings(BaseSettings):
    """Configuration for the LLM client.

    All fields map 1-to-1 to environment variables.
    See .env.example for documentation.
    """

    llm_api_key: str = ""
    llm_base_url: str = "https://api.featherless.ai/v1"
    llm_model_name: str = "Qwen/Qwen3.8-27B"
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.3

    model_config = SettingsConfigDict(
        env_file=(ENV_PATH, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> LLMSettings:
    """Return a cached LLMSettings instance (reads env once)."""
    return LLMSettings()
