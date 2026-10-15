"""Budgets and the CostGuard: observe spend, warn, block."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

from pydantic import BaseModel

from .usage import Usage


class OverBudgetError(RuntimeError):
    """Raised when the guard's policy is ``block`` and a limit is exceeded."""

    def __init__(self, message: str, dimension: str, limit: float, spent: float):
        super().__init__(message)
        self.dimension = dimension
        self.limit = limit
        self.spent = spent


@dataclass
class Alert:
    """One threshold crossing, delivered to registered callbacks."""

    level: str  # "warn" | "block"
    dimension: str
    limit: float
    spent: float
    message: str


class Budget(BaseModel):
    """Spend limits. Any of them may be set; unset limits are unlimited.

    * ``limit_usd`` — hard cap for the whole lifetime of this budget
    * ``per_model_usd`` — cap per model (checked against that model's spend)
    * ``daily_usd`` — rolling cap for the current UTC day
    """

    limit_usd: float | None = None
    per_model_usd: float | None = None
    daily_usd: float | None = None
    warn_at: float = 0.8  # fraction of any limit at which "warn" fires


@dataclass
class Observation:
    """Result of feeding one usage record through the guard."""

    allowed: bool
    spent_total: float
    spent_daily: float
    spent_by_model: dict[str, float] = field(default_factory=dict)
    level: str = "ok"  # ok | warn | block
    alerts: list[Alert] = field(default_factory=list)


class CostGuard:
    """Tracks spend against a :class:`Budget` and enforces policy.

    Policy:
      * ``"warn"``  — log alerts, never block
      * ``"block"`` — raise :class:`OverBudgetError` once a limit is exceeded
      * ``"report"`` — record everything, alert only at the block level

    A monotonic clock (injected ``now``) keeps daily windows testable.
    """

    def __init__(
        self,
        budget: Budget,
        *,
        policy: str = "warn",
        now: Callable[[], datetime] | None = None,
        on_alert: Callable[[Alert], None] | None = None,
    ):
        if policy not in ("warn", "block", "report"):
            raise ValueError(f"unknown policy: {policy!r}")
        self.budget = budget
        self.policy = policy
        self._now = now
        self._on_alert = on_alert
        self._spent_total = 0.0
        self._daily: dict[date, float] = {}  # spend per UTC day of usage
        self._spent_by_model: dict[str, float] = {}
        self._fired: set[tuple[str, str]] = set()  # (level, dimension)

    @property
    def clock(self) -> Callable[[], datetime]:
        if self._now is not None:
            return self._now
        from .usage import utc_now

        return utc_now

    def _spent_daily_for(self, ts: datetime) -> float:
        return self._daily.get(ts.date(), 0.0)

    def _prune_days(self, ts: datetime) -> None:
        # drop days strictly before the usage day to bound memory
        cutoff = ts.date()
        for d in [d for d in self._daily if d < cutoff]:
            del self._daily[d]
            self._fired = {
                (lvl, dim) for (lvl, dim) in self._fired if dim != "day"
            }

    def observe(self, usage: Usage, cost: float) -> Observation:
        """Record one usage's cost; return the post-observation state.

        Raises :class:`OverBudgetError` under policy ``block`` when any
        limit would be exceeded by this record.
        """
        ts = usage.timestamp
        self._prune_days(ts)

        prospective_total = self._spent_total + cost
        prospective_daily = self._spent_daily_for(ts) + cost
        prospective_model = self._spent_by_model.get(usage.model, 0.0) + cost

        limits: list[tuple[str, float | None, float]] = [
            ("total", self.budget.limit_usd, prospective_total),
            (f"model:{usage.model}", self.budget.per_model_usd, prospective_model),
            ("day", self.budget.daily_usd, prospective_daily),
        ]

        alerts: list[Alert] = []
        blocked_by: str | None = None
        for dim, limit, spent in limits:
            if limit is None or limit <= 0:
                continue
            if spent >= limit:
                msg = f"budget exceeded on {dim}: ${spent:.4f} >= ${limit:.2f}"
                key = ("block", dim)
                if key not in self._fired:
                    self._fired.add(key)
                    alerts.append(
                        Alert("block", dim, limit, spent, msg)
                    )
                blocked_by = dim
                continue
            if spent >= limit * self.budget.warn_at:
                key = ("warn", dim)
                if key not in self._fired:
                    self._fired.add(key)
                    msg = (
                        f"approaching budget on {dim}: "
                        f"${spent:.4f} / ${limit:.2f} "
                        f"({spent / limit:.0%})"
                    )
                    alerts.append(Alert("warn", dim, limit, spent, msg))

        if blocked_by is not None and self.policy == "block":
            limit, spent = next(
                (lim, s) for d, lim, s in limits if d == blocked_by
            )
            for a in alerts:
                if self._on_alert:
                    self._on_alert(a)
            raise OverBudgetError(
                f"blocked by budget ({blocked_by}): ${spent:.4f} >= ${limit:.2f}",
                dimension=blocked_by,
                limit=limit,
                spent=spent,
            )

        # commit
        self._spent_total += cost
        self._daily[ts.date()] = self._spent_daily_for(ts) + cost
        self._spent_by_model[usage.model] = self._spent_by_model.get(usage.model, 0.0) + cost

        level = "ok"
        if any(a.level == "block" for a in alerts):
            level = "block"
        elif any(a.level == "warn" for a in alerts):
            level = "warn"
        for a in alerts:
            if self._on_alert:
                self._on_alert(a)

        return Observation(
            allowed=True,
            spent_total=self._spent_total,
            spent_daily=self._spent_daily_for(self.clock()),
            spent_by_model=dict(self._spent_by_model),
            level=level,
            alerts=alerts,
        )

    # -- reporting ---------------------------------------------------------

    def status(self) -> dict:
        """Current spend snapshot for dashboards/tests."""
        return {
            "spent_total": self._spent_total,
            "spent_daily": self._spent_daily_for(self.clock()),
            "spent_by_model": dict(self._spent_by_model),
            "budget": self.budget.model_dump(),
            "policy": self.policy,
        }
