"""LM Studio client for the llmwiki eval harness."""

from __future__ import annotations

import re

from openai import AsyncOpenAI
from agents.models.openai_responses import OpenAIResponsesModel
from agents import ModelSettings
from agents.model_settings import Reasoning
from pydantic_settings import BaseSettings


class _Settings(BaseSettings):
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "google/gemma-4-e4b"
    lm_studio_timeout: float = 120.0

    class Config:
        env_prefix = "LM_STUDIO_"
        env_file = ".env"
        extra = "ignore"


settings = _Settings()

_client = AsyncOpenAI(
    base_url=settings.lm_studio_base_url,
    api_key="lm-studio",
    timeout=settings.lm_studio_timeout,
)


def get_model() -> OpenAIResponsesModel:
    return OpenAIResponsesModel(model=settings.lm_studio_model, openai_client=_client)


def get_model_settings(reasoning: bool = False) -> ModelSettings:
    return ModelSettings(
        # temperature=0.1,
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
