"""Parse provider responses into Usage records.

Supports the OpenAI-compatible wire format (chat/completions and
embeddings) so the guard can sit between your code and any compatible
endpoint — OpenAI, Azure, vLLM, Ollama, LiteLLM — without changing call
sites.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .usage import Usage


class UnparseableResponseError(ValueError):
    """Raised when a response has no usable ``usage`` block."""


def parse_openai_response(
    data: dict,
    *,
    project: str = "default",
    api_key_id: str = "",
    timestamp: datetime | None = None,
) -> Usage:
    """Extract a :class:`Usage` from a chat/completions or embeddings body.

    Tolerates missing ``usage`` (returns zero usage) so partial or
    mocked responses do not crash the pipeline.
    """
    usage = data.get("usage") or {}
    model = str(data.get("model") or "unknown")
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    cached = 0
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens", 0) or 0)
    if cached > prompt_tokens:
        cached = prompt_tokens  # defensive: cache is a subset of prompt
    return Usage(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_prompt_tokens=cached,
        project=project,
        api_key_id=api_key_id,
        timestamp=timestamp if timestamp is not None else datetime.now(UTC),
    )


def parse_response_json(
    body: str,
    *,
    project: str = "default",
    api_key_id: str = "",
    timestamp: datetime | None = None,
) -> Usage:
    """Same as :func:`parse_openai_response` but from a raw JSON string."""
    import json

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise UnparseableResponseError(f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise UnparseableResponseError("response body is not a JSON object")
    return parse_openai_response(
        data, project=project, api_key_id=api_key_id, timestamp=timestamp
    )
