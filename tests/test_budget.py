"""CostGuard tests: warn/block, per-dimension limits, daily rollover."""

from datetime import UTC, datetime, timedelta

import pytest

from cost_guard.budget import Alert, Budget, CostGuard, OverBudgetError
from cost_guard.pricing import get_pricing
from cost_guard.usage import Usage

T0 = datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self, start: datetime):
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kw):
        self.t = self.t + timedelta(**kw)


def usage(model="gpt-4o-mini", prompt=100, completion=100, ts=None):
    return Usage(
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        timestamp=ts or T0,
    )


def cost_of(u: Usage) -> float:
    return u.cost(get_pricing(u.model))


def make_guard(budget, policy="warn", start=T0, on_alert=None):
    return CostGuard(
        budget, policy=policy, now=FakeClock(start), on_alert=on_alert
    )


def test_first_observation_ok():
    g = make_guard(Budget(limit_usd=1.0))
    obs = g.observe(usage(), cost=0.0001)
    assert obs.allowed
    assert obs.level == "ok"
    assert obs.spent_total == pytest.approx(0.0001)
    assert obs.alerts == []


def test_warn_fires_at_threshold_once():
    alerts: list[Alert] = []
    g = make_guard(Budget(limit_usd=1.0, warn_at=0.5), on_alert=alerts.append)
    obs = g.observe(usage(), cost=0.50)
    assert obs.level == "warn"
    assert len(obs.alerts) == 1
    assert obs.alerts[0].level == "warn"
    # second warn-level observation must NOT re-fire
    obs2 = g.observe(usage(), cost=0.10)
    assert obs2.alerts == []
    assert len(alerts) == 1


def test_block_level_alert_recorded_in_warn_policy():
    g = make_guard(Budget(limit_usd=1.0), policy="warn")
    obs = g.observe(usage(), cost=1.50)
    assert obs.level == "block"
    assert obs.allowed  # warn policy never blocks
    assert any(a.level == "block" for a in obs.alerts)


def test_block_policy_raises():
    g = make_guard(Budget(limit_usd=1.0), policy="block")
    with pytest.raises(OverBudgetError) as ei:
        g.observe(usage(), cost=1.50)
    assert ei.value.dimension == "total"
    assert ei.value.limit == pytest.approx(1.0)
    assert ei.value.spent == pytest.approx(1.50)
    # spend was NOT committed on block
    assert g.status()["spent_total"] == 0.0


def test_block_policy_allows_under_limit():
    g = make_guard(Budget(limit_usd=1.0), policy="block")
    obs = g.observe(usage(), cost=0.90)
    assert obs.allowed
    assert g.status()["spent_total"] == pytest.approx(0.90)


def test_per_model_limit():
    g = make_guard(Budget(per_model_usd=1.0), policy="block")
    g.observe(usage(model="gpt-4o-mini"), cost=0.60)
    g.observe(usage(model="gpt-4o"), cost=0.90)  # other model unaffected
    with pytest.raises(OverBudgetError) as ei:
        g.observe(usage(model="gpt-4o-mini"), cost=0.40)
    assert ei.value.dimension == "model:gpt-4o-mini"


def test_daily_limit_and_rollover():
    clock = FakeClock(T0)
    g = CostGuard(Budget(daily_usd=1.0), policy="block", now=clock)
    g.observe(usage(), cost=0.70)
    with pytest.raises(OverBudgetError) as ei:
        g.observe(usage(), cost=0.40)
    assert ei.value.dimension == "day"
    # next UTC day: daily budget resets, total still tracked
    clock.advance(days=1)
    obs = g.observe(usage(ts=clock()), cost=0.70)
    assert obs.allowed
    assert obs.spent_daily == pytest.approx(0.70)
    assert g.status()["spent_total"] == pytest.approx(1.40)


def test_daily_limit_not_applied_to_old_timestamps():
    g = make_guard(Budget(daily_usd=1.0), policy="block")
    # a timestamp from an earlier day does not touch today's daily budget
    obs = g.observe(usage(ts=T0 - timedelta(days=1)), cost=0.90)
    assert obs.spent_daily == 0.0
    assert obs.allowed


def test_status_snapshot():
    g = make_guard(Budget(limit_usd=10.0))
    g.observe(usage(model="gpt-4o"), cost=1.0)
    st = g.status()
    assert st["spent_by_model"]["gpt-4o"] == pytest.approx(1.0)
    assert st["policy"] == "warn"
    assert st["budget"]["limit_usd"] == 10.0


def test_invalid_policy_rejected():
    with pytest.raises(ValueError, match="policy"):
        CostGuard(Budget(limit_usd=1.0), policy="explode")


def test_report_policy_only_block_alerts():
    alerts: list[Alert] = []
    g = make_guard(Budget(limit_usd=1.0), policy="report", on_alert=alerts.append)
    g.observe(usage(), cost=0.50)  # warn-level crossing, policy=report
    assert alerts == []
    g.observe(usage(), cost=0.60)  # block-level crossing
    assert any(a.level == "block" for a in alerts)


def test_cumulative_spending_across_days_total():
    clock = FakeClock(T0)
    g = CostGuard(Budget(limit_usd=2.0), policy="block", now=clock)
    g.observe(usage(), cost=1.0)
    clock.advance(days=1)
    g.observe(usage(ts=clock()), cost=0.9)
    with pytest.raises(OverBudgetError) as ei:
        g.observe(usage(ts=clock()), cost=0.2)
    assert ei.value.dimension == "total"
