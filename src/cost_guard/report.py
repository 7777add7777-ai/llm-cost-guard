"""Spending reports: group-by, top-N, CSV export."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

from .ledger import CostLedger


@dataclass
class ReportRow:
    key: str
    cost: float
    requests: int
    tokens: int
    share: float = 0.0  # filled by top_models()


class SpendingReport:
    """Groups ledger entries by model, project, and day."""

    def __init__(self, ledger: CostLedger, **filters) -> None:
        self.entries = [e for e in ledger.entries if _match_filters(e, filters)]

    @property
    def total_cost(self) -> float:
        return sum(e.cost for e in self.entries)

    @property
    def total_requests(self) -> int:
        return len(self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.usage.total_tokens for e in self.entries)

    def by_model(self) -> list[ReportRow]:
        return self._group_by(lambda e: e.usage.model)

    def by_project(self) -> list[ReportRow]:
        return self._group_by(lambda e: e.usage.project)

    def by_day(self) -> list[ReportRow]:
        return self._group_by(lambda e: e.usage.timestamp.date().isoformat())

    def top_models(self, n: int = 5) -> list[ReportRow]:
        rows = self.by_model()
        total = self.total_cost
        for r in rows:
            r.share = (r.cost / total) if total else 0.0
        return rows[:n]

    def _group_by(self, key_fn) -> list[ReportRow]:
        buckets: dict[str, list] = {}
        for e in self.entries:
            buckets.setdefault(key_fn(e), []).append(e)
        rows = [
            ReportRow(key=k, cost=sum(x.cost for x in v), requests=len(v),
                      tokens=sum(x.usage.total_tokens for x in v))
            for k, v in sorted(buckets.items())
        ]
        return rows

    def to_csv(self) -> str:
        """Full per-entry CSV for spreadsheets/BI tools."""
        buf = StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["timestamp", "model", "project", "api_key_id",
             "prompt_tokens", "completion_tokens", "cached_prompt_tokens", "cost_usd"]
        )
        for e in self.entries:
            u = e.usage
            writer.writerow(
                [
                    u.timestamp.isoformat(),
                    u.model,
                    u.project,
                    u.api_key_id,
                    u.prompt_tokens,
                    u.completion_tokens,
                    u.cached_prompt_tokens,
                    f"{e.cost:.10f}",
                ]
            )
        return buf.getvalue()

    def summary(self) -> str:
        models = self.by_model()
        lines = [
            f"requests={self.total_requests}  "
            f"tokens={self.total_tokens}  cost=${self.total_cost:.4f}"
        ]
        for r in models[:5]:
            lines.append(f"  {r.key}: ${r.cost:.4f} over {r.requests} reqs")
        return "\n".join(lines)


def _match_filters(entry, filters: dict) -> bool:
    from .ledger import _matches

    return _matches(entry, filters)
