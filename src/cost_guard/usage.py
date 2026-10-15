"""Usage records: one observed LLM request."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .pricing import Pricing


class Usage(BaseModel):
    """Token usage of a single LLM request, as reported by the provider."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    project: str = "default"
    api_key_id: str = ""  # a stable key label, never the secret itself
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost(self, pricing: Pricing) -> float:
        """Cost in USD; cached prompt tokens are billed at the cache rate."""
        fresh_prompt = max(0, self.prompt_tokens - self.cached_prompt_tokens)
        input_cost = (fresh_prompt * pricing.input_per_mtok) / 1e6
        cache_cost = (self.cached_prompt_tokens * pricing.cache_rate) / 1e6
        output_cost = (self.completion_tokens * pricing.output_per_mtok) / 1e6
        return input_cost + cache_cost + output_cost


def utc_now() -> datetime:
    return datetime.now(UTC)
