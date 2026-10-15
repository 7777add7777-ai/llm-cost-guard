"""Ledger tests: append, persist, load, filter."""

import json
from datetime import UTC

import pytest

from cost_guard.ledger import CostLedger
from cost_guard.pricing import Pricing, UnknownModelError
from cost_guard.usage import Usage

TS = "2026-10-15T10:00:00+00:00"


def make_usage(model="gpt-4o-mini", project="p1", ts=TS, prompt=100, completion=50):
    return Usage(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        project=project,
        timestamp=ts,
    )


def test_record_appends_and_prices():
    ledger = CostLedger()
    entry = ledger.record(make_usage(prompt=100, completion=100))
    assert len(ledger) == 1
    expected = 100 * 0.15 / 1e6 + 100 * 0.60 / 1e6
    assert entry.cost == pytest.approx(expected)
    assert entry.model_pricing == "gpt-4o-mini"


def test_total_cost_and_tokens():
    ledger = CostLedger()
    ledger.record(make_usage(prompt=100, completion=100))
    ledger.record(make_usage(prompt=200, completion=0))
    assert ledger.total_cost() == pytest.approx(
        100 * 0.15 / 1e6 + 100 * 0.60 / 1e6 + 200 * 0.15 / 1e6
    )
    assert ledger.total_tokens() == 100 + 100 + 200


def test_filter_by_project():
    ledger = CostLedger()
    ledger.record(make_usage(project="a"))
    ledger.record(make_usage(project="b"))
    a_cost = ledger.total_cost(project="a")
    assert a_cost == pytest.approx(100 * 0.15 / 1e6 + 50 * 0.60 / 1e6)
    assert ledger.total_cost(project="zzz") == 0.0
    assert len(list(ledger)) == 2


def test_filter_by_model():
    ledger = CostLedger()
    ledger.record(make_usage(model="gpt-4o-mini"))
    ledger.record(make_usage(model="gpt-4o", prompt=1000, completion=1000))
    c = ledger.total_cost(model="gpt-4o")
    assert c == pytest.approx(1000 * 2.5 / 1e6 + 1000 * 10.0 / 1e6)


def test_filter_by_time_window():
    ledger = CostLedger()
    from datetime import datetime

    ledger.record(make_usage(ts="2026-10-14T09:00:00+00:00"))
    ledger.record(make_usage(ts="2026-10-15T09:00:00+00:00"))
    since = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    assert ledger.total_cost(since=since) == pytest.approx(
        100 * 0.15 / 1e6 + 50 * 0.60 / 1e6
    )
    until = datetime(2026, 10, 14, 23, 59, tzinfo=UTC)
    assert ledger.total_cost(until=until) == pytest.approx(
        100 * 0.15 / 1e6 + 50 * 0.60 / 1e6
    )


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = CostLedger(path)
    ledger.record(make_usage(prompt=100, completion=100))
    ledger.record(make_usage(model="gpt-4o", project="other"))
    assert path.exists()

    reloaded = CostLedger(path)
    assert len(reloaded) == 2
    assert reloaded.total_cost() == pytest.approx(ledger.total_cost())
    models = [e.usage.model for e in reloaded]
    assert models == ["gpt-4o-mini", "gpt-4o"]


def test_record_appends_to_existing_file(tmp_path):
    path = tmp_path / "ledger.jsonl"
    l1 = CostLedger(path)
    l1.record(make_usage())
    l2 = CostLedger(path)
    l2.record(make_usage())
    l3 = CostLedger(path)
    assert len(l3) == 2


def test_save_compaction(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = CostLedger(path)
    ledger.record(make_usage())
    ledger.record(make_usage(project="x"))
    # append a blank line, then compact
    with path.open("a") as f:
        f.write("\n")
    ledger.save()
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    reloaded = CostLedger(path)
    assert len(reloaded) == 2


def test_load_missing_file_is_empty(tmp_path):
    ledger = CostLedger(tmp_path / "nope.jsonl")
    assert len(ledger) == 0


def test_load_skips_blank_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    u = make_usage()
    line = json.dumps({"usage": u.model_dump(mode="json"), "cost": 0.0001,
                       "model_pricing": "gpt-4o-mini"})
    path.write_text(f"{line}\n\n{line}\n", encoding="utf-8")
    ledger = CostLedger(path)
    assert len(ledger) == 2


def test_unknown_model_record_raises():
    ledger = CostLedger()
    with pytest.raises(UnknownModelError):
        ledger.record(Usage(model="not-in-catalog", prompt_tokens=10))


def test_custom_catalog_record():
    ledger = CostLedger()
    catalog = {"my-model": Pricing(model="my-model", input_per_mtok=1.0,
                                   output_per_mtok=2.0)}
    entry = ledger.record(make_usage(model="my-model"), catalog=catalog)
    assert entry.model_pricing == "my-model"
    assert entry.cost == pytest.approx(100 / 1e6 + 50 * 2 / 1e6)


def test_entries_are_copies():
    ledger = CostLedger()
    ledger.record(make_usage())
    snap = ledger.entries
    assert snap is not ledger.entries
    ledger.record(make_usage())
    assert len(snap) == 1
