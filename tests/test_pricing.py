"""Pricing tests: exact math, cache discounts, catalog lookup."""

import pytest

from cost_guard.pricing import (
    PRICING,
    Pricing,
    UnknownModelError,
    get_pricing,
)
from cost_guard.usage import Usage


def make_usage(prompt=1000, completion=500, cached=0, model="gpt-4o-mini"):
    return Usage(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_prompt_tokens=cached,
    )


def test_cost_math_no_cache():
    u = make_usage(prompt=100, completion=100)
    p = PRICING["gpt-4o-mini"]
    # 100 * 0.15/1e6 + 100 * 0.60/1e6
    assert u.cost(p) == pytest.approx(100 * 0.15 / 1e6 + 100 * 0.60 / 1e6)


def test_cost_math_with_cache():
    u = make_usage(prompt=1000, completion=0, cached=400)
    p = PRICING["gpt-4o-mini"]
    # 600 fresh @ 0.15 + 400 cached @ 0.075
    expected = 600 * 0.15 / 1e6 + 400 * 0.075 / 1e6
    assert u.cost(p) == pytest.approx(expected)


def test_cache_discount_saves_money():
    p = PRICING["gpt-4o-mini"]
    no_cache = make_usage(prompt=1000, completion=0, cached=0).cost(p)
    all_cached = make_usage(prompt=1000, completion=0, cached=1000).cost(p)
    assert all_cached < no_cache
    assert all_cached == pytest.approx(1000 * 0.075 / 1e6)


def test_cache_rate_derived_when_unset():
    p = Pricing(model="m", input_per_mtok=2.0, output_per_mtok=4.0)
    assert p.cache_rate == pytest.approx(0.2)  # 2.0 / 10


def test_cache_rate_explicit_wins():
    p = Pricing(model="m", input_per_mtok=2.0, output_per_mtok=4.0, cache_read_per_mtok=0.5)
    assert p.cache_rate == pytest.approx(0.5)


def test_get_pricing_exact():
    assert get_pricing("gpt-4o-mini").model == "gpt-4o-mini"


def test_get_pricing_dated_suffix():
    assert get_pricing("gpt-4o-mini-2024-07-18").model == "gpt-4o-mini"
    assert get_pricing("gpt-4o-2024-11-20").model == "gpt-4o"


def test_get_pricing_longest_prefix_wins():
    # both gpt-4o and gpt-4o-mini are prefixes of gpt-4o-mini; the longer must win
    p = get_pricing("gpt-4o-mini-2024-07-18")
    assert p.input_per_mtok == pytest.approx(0.15)


def test_get_pricing_unknown_raises():
    with pytest.raises(UnknownModelError):
        get_pricing("totally-unknown-model")


def test_get_pricing_default_fallback():
    zero = Pricing(model="self-hosted", input_per_mtok=0.0, output_per_mtok=0.0)
    p = get_pricing("my-local-model", default=zero)
    assert p.input_per_mtok == 0.0


def test_zero_pricing_costs_zero():
    zero = Pricing(model="local")
    u = make_usage(prompt=10_000_000, completion=10_000_000, model="local")
    assert u.cost(zero) == 0.0


def test_usage_total_tokens():
    assert make_usage(prompt=10, completion=5).total_tokens == 15
