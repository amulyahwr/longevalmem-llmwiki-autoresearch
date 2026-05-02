"""Ollama client for the llmwiki eval harness."""

from __future__ import annotations

import os
import re

from openai import AsyncOpenAI
from agents.models.openai_responses import OpenAIResponsesModel
from agents import ModelSettings
from agents.model_settings import Reasoning
from pydantic_settings import BaseSettings


class _Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "gemma4:e2b"
    ollama_timeout: float = 120.0

    class Config:
        env_prefix = "OLLAMA_"
        env_file = ".env"
        extra = "ignore"


settings = _Settings()

# The openai-agents SDK checks os.environ for OPENAI_API_KEY at import time.
# Ollama doesn't need a real key; set a placeholder so the SDK doesn't warn.
os.environ.setdefault("OPENAI_API_KEY", "ollama")

_client = AsyncOpenAI(
    base_url=settings.ollama_base_url,
    api_key="ollama",
    timeout=settings.ollama_timeout,
)


def get_model() -> OpenAIResponsesModel:
    return OpenAIResponsesModel(model=settings.ollama_model, openai_client=_client)


def get_model_settings(reasoning: bool = False) -> ModelSettings:
    return ModelSettings(
        temperature=0.0,
        reasoning=Reasoning(effort="medium" if reasoning else "none"),
    )


def clean_json(text: str) -> str:
    """Extract JSON from model output — strips code fences and leading prose."""
    text = text.strip()
    # Strip code fences first
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # Strip any prose preceding the JSON object/array
    for i, ch in enumerate(text):
        if ch in "{[":
            return text[i:]
    return text
