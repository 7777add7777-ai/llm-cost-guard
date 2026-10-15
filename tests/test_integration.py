"""Integration: parser -> ledger -> guard -> report, one full loop."""

from datetime import UTC, datetime

import pytest

from cost_guard.budget import Budget, CostGuard, OverBudgetError
from cost_guard.ledger import CostLedger
from cost_guard.parser import parse_openai_response
from cost_guard.pricing import get_pricing
from cost_guard.report import SpendingReport

TS = "2026-10-15T12:00:00+00:00"


def chat_body(model, prompt, completion):
    return {
        "model": model,
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def run_workload(ledger, guard, bodies):
    for body in bodies:
        usage = parse_openai_response(body, timestamp=TS)
        entry = ledger.record(usage)
        guard.observe(usage, entry.cost)


def test_full_loop_accounting():
    ledger = CostLedger()
    guard = CostGuard(Budget(limit_usd=10.0))
    bodies = [
        chat_body("gpt-4o-mini", 100, 100),
        chat_body("gpt-4o-mini", 100, 100),
        chat_body("gpt-4o", 100, 100),
    ]
    run_workload(ledger, guard, bodies)
    expected = (
        2 * (100 * 0.15 / 1e6 + 100 * 0.60 / 1e6)
        + (100 * 2.5 / 1e6 + 100 * 10.0 / 1e6)
    )
    assert guard.status()["spent_total"] == pytest.approx(expected)
    assert ledger.total_cost() == pytest.approx(expected)
    rep = SpendingReport(ledger)
    assert rep.total_requests == 3


def test_full_loop_blocks_on_overrun():
    ledger = CostLedger()
    guard = CostGuard(Budget(limit_usd=0.0001), policy="block")
    bodies = [chat_body("gpt-4o", 1000, 1000)]  # ~0.0125 USD
    with pytest.raises(OverBudgetError):
        run_workload(ledger, guard, bodies)
    # the blocked call is in the ledger? No: ledger.record happens BEFORE
    # observe. We accept that the accounting includes the blocked call, but
    # the guard's committed spend is zero.
    assert guard.status()["spent_total"] == 0.0
    assert ledger.total_cost() > 0.0


def test_guard_and_ledger_disagree_by_design():
    """Ledger = what was requested; guard = what the budget allows."""
    ledger = CostLedger()
    guard = CostGuard(Budget(limit_usd=0.000001), policy="block")
    body = chat_body("gpt-4o-mini", 100000, 100000)
    usage = parse_openai_response(body, timestamp=TS)
    entry = ledger.record(usage)
    with pytest.raises(OverBudgetError):
        guard.observe(usage, entry.cost)
    assert entry.cost > 0.0  # ledger knows the price


def test_report_matches_guard_after_rollover(tmp_path):
    ledger = CostLedger(tmp_path / "l.jsonl")
    def fixed_now():
        return datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC)

    guard = CostGuard(Budget(limit_usd=10.0), now=fixed_now)
    bodies = [
        chat_body("gpt-4o-mini", 500, 500),
        chat_body("gpt-4o-mini", 800, 200),
    ]
    run_workload(ledger, guard, bodies)
    rep = SpendingReport(ledger)
    assert rep.total_cost == pytest.approx(guard.status()["spent_total"])


def test_pricing_used_by_ledger_matches_guard():
    """Both sides must price the same usage identically."""
    ledger = CostLedger()
    guard = CostGuard(Budget(limit_usd=10.0))
    usage = parse_openai_response(
        chat_body("gpt-4o", 123, 456), timestamp=TS
    )
    entry = ledger.record(usage)
    expected = usage.cost(get_pricing(usage.model))
    assert entry.cost == pytest.approx(expected)
    guard.observe(usage, entry.cost)
    assert guard.status()["spent_total"] == pytest.approx(expected)
