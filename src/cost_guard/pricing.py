"""Model pricing: per-million-token rates and catalog lookup."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Pricing(BaseModel):
    """Price per 1M tokens. Cached prompt tokens are billed at a discount."""

    model: str
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cache_read_per_mtok: float = Field(default=-1.0)  # -1 = derive (1/10 of input)

    @property
    def cache_rate(self) -> float:
        if self.cache_read_per_mtok >= 0:
            return self.cache_read_per_mtok
        return self.input_per_mtok / 10.0


class UnknownModelError(KeyError):
    """Raised when a model has no pricing entry."""


#: Built-in catalog. Rates are US$ per 1M tokens (2026 public list prices,
#: approximate — override with your own catalog in production).
PRICING: dict[str, Pricing] = {
    "gpt-4o": Pricing(
        model="gpt-4o", input_per_mtok=2.50, output_per_mtok=10.00, cache_read_per_mtok=1.25
    ),
    "gpt-4o-mini": Pricing(
        model="gpt-4o-mini", input_per_mtok=0.15, output_per_mtok=0.60, cache_read_per_mtok=0.075
    ),
    "gpt-4.1": Pricing(
        model="gpt-4.1", input_per_mtok=2.00, output_per_mtok=8.00, cache_read_per_mtok=0.50
    ),
    "gpt-4.1-mini": Pricing(
        model="gpt-4.1-mini", input_per_mtok=0.40, output_per_mtok=1.60, cache_read_per_mtok=0.10
    ),
    "claude-3-5-sonnet": Pricing(
        model="claude-3-5-sonnet",
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cache_read_per_mtok=0.30,
    ),
    "claude-3-5-haiku": Pricing(
        model="claude-3-5-haiku",
        input_per_mtok=0.80,
        output_per_mtok=4.00,
        cache_read_per_mtok=0.08,
    ),
    "llama-3.1-70b": Pricing(model="llama-3.1-70b", input_per_mtok=0.59, output_per_mtok=0.79),
    "mistral-large": Pricing(
        model="mistral-large", input_per_mtok=2.00, output_per_mtok=6.00
    ),
}


def get_pricing(
    model: str,
    catalog: dict[str, Pricing] | None = None,
    *,
    default: Pricing | None = None,
) -> Pricing:
    """Look up pricing by model id, supporting dated-suffix matching.

    ``gpt-4o-mini-2024-07-18`` resolves to ``gpt-4o-mini``. Longest catalog
    key that is a prefix of the model id wins. Unknown models raise
    :class:`UnknownModelError` unless ``default`` is given (commonly a
    zero-priced entry for self-hosted models).
    """
    cat = catalog if catalog is not None else PRICING
    if model in cat:
        return cat[model]
    best: str | None = None
    for key in cat:
        if model.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    if best is not None:
        return cat[best]
    if default is not None:
        return default
    raise UnknownModelError(f"no pricing for model: {model!r}")
