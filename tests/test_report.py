"""Report tests: group-by, shares, CSV."""

import csv
import io

import pytest

from cost_guard.ledger import CostLedger
from cost_guard.report import SpendingReport
from cost_guard.usage import Usage

TS1 = "2026-10-15T10:00:00+00:00"
TS2 = "2026-10-16T10:00:00+00:00"


def make_ledger() -> CostLedger:
    ledger = CostLedger()
    ledger.record(
        Usage(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=100,
              project="a", timestamp=TS1)
    )
    ledger.record(
        Usage(model="gpt-4o", prompt_tokens=1000, completion_tokens=1000,
              project="a", timestamp=TS1)
    )
    ledger.record(
        Usage(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=100,
              project="b", timestamp=TS2)
    )
    return ledger


def test_totals():
    rep = SpendingReport(make_ledger())
    assert rep.total_requests == 3
    expected = (
        2 * (100 * 0.15 / 1e6 + 100 * 0.60 / 1e6)
        + (1000 * 2.5 / 1e6 + 1000 * 10.0 / 1e6)
    )
    assert rep.total_cost == pytest.approx(expected)
    assert rep.total_tokens == 2 * 200 + 2000


def test_by_model_ordering():
    rep = SpendingReport(make_ledger())
    rows = rep.by_model()
    assert [r.key for r in rows] == ["gpt-4o", "gpt-4o-mini"]
    assert rows[1].requests == 2


def test_by_project():
    rep = SpendingReport(make_ledger())
    rows = rep.by_project()
    by_key = {r.key: r for r in rows}
    assert by_key["a"].requests == 2
    assert by_key["b"].requests == 1


def test_by_day():
    rep = SpendingReport(make_ledger())
    rows = rep.by_day()
    assert [r.key for r in rows] == ["2026-10-15", "2026-10-16"]


def test_top_models_shares_sum_to_one():
    rep = SpendingReport(make_ledger())
    rows = rep.top_models(10)
    assert sum(r.share for r in rows) == pytest.approx(1.0)
    # gpt-4o is the expensive one
    assert rows[0].key == "gpt-4o"
    assert rows[0].share > rows[1].share


def test_filters_propagate():
    rep = SpendingReport(make_ledger(), project="b")
    assert rep.total_requests == 1
    assert rep.by_model()[0].key == "gpt-4o-mini"


def test_csv_roundtrip():
    rep = SpendingReport(make_ledger())
    rows = list(csv.reader(io.StringIO(rep.to_csv())))
    assert rows[0] == [
        "timestamp", "model", "project", "api_key_id",
        "prompt_tokens", "completion_tokens", "cached_prompt_tokens", "cost_usd",
    ]
    assert len(rows) == 4  # header + 3 entries
    assert rows[1][1] == "gpt-4o-mini"
    float(rows[1][7])  # cost parses as float


def test_empty_report():
    rep = SpendingReport(CostLedger())
    assert rep.total_cost == 0.0
    assert rep.by_model() == []
    assert rep.top_models(5) == []
    assert "requests=0" in rep.summary()


def test_summary_contains_totals():
    rep = SpendingReport(make_ledger())
    s = rep.summary()
    assert "cost=$" in s
    assert "gpt-4o" in s
