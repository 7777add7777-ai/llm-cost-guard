"""Parser tests: chat completions, embeddings, malformed input."""

import json
from datetime import UTC

import pytest

from cost_guard.parser import (
    UnparseableResponseError,
    parse_openai_response,
    parse_response_json,
)


def test_parse_chat_full():
    body = {
        "id": "chatcmpl-1",
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {
            "prompt_tokens": 250,
            "completion_tokens": 120,
            "total_tokens": 370,
            "prompt_tokens_details": {"cached_tokens": 100},
        },
    }
    u = parse_openai_response(body, project="p")
    assert u.model == "gpt-4o-mini-2024-07-18"
    assert u.prompt_tokens == 250
    assert u.completion_tokens == 120
    assert u.cached_prompt_tokens == 100
    assert u.project == "p"


def test_parse_chat_no_usage_block():
    u = parse_openai_response({"model": "gpt-4o", "choices": []})
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0
    assert u.model == "gpt-4o"


def test_parse_embeddings():
    body = {
        "model": "text-embedding-3-small",
        "data": [{"embedding": [0.1]}],
        "usage": {"prompt_tokens": 80, "total_tokens": 80},
    }
    u = parse_openai_response(body)
    assert u.prompt_tokens == 80
    assert u.completion_tokens == 0


def test_cached_capped_at_prompt():
    body = {
        "model": "m",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 999},
        },
    }
    u = parse_openai_response(body)
    assert u.cached_prompt_tokens == 100


def test_missing_details_ignored():
    body = {"model": "m", "usage": {"prompt_tokens": 5, "completion_tokens": 5}}
    u = parse_openai_response(body)
    assert u.cached_prompt_tokens == 0


def test_explicit_timestamp_wins():
    from datetime import datetime

    ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    u = parse_openai_response({"model": "m"}, timestamp=ts)
    assert u.timestamp == ts


def test_default_timestamp_is_utc_aware():
    u = parse_openai_response({"model": "m"})
    assert u.timestamp.tzinfo is not None


def test_parse_json_string_roundtrip():
    body = {"model": "gpt-4o", "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    u = parse_response_json(json.dumps(body), api_key_id="key-1")
    assert u.prompt_tokens == 7
    assert u.api_key_id == "key-1"


def test_parse_json_invalid():
    with pytest.raises(UnparseableResponseError, match="not valid JSON"):
        parse_response_json("{nope")


def test_parse_json_non_object():
    with pytest.raises(UnparseableResponseError, match="not a JSON object"):
        parse_response_json("[1, 2, 3]")
