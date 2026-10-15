"""Cost ledger: append-only JSONL storage of priced usage entries."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .pricing import PRICING, get_pricing
from .usage import Usage


@dataclass(frozen=True)
class Entry:
    """A usage record with its computed cost."""

    usage: Usage
    cost: float
    model_pricing: str  # catalog key that was used


class CostLedger:
    """Append-only ledger with JSONL persistence.

    The ledger is the single source of truth for "how much did this
    workload cost". Everything else (budgets, reports) is derived from it.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._entries: list[Entry] = []
        if self.path and self.path.exists():
            self.load(self.path)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(self._entries)

    @property
    def entries(self) -> list[Entry]:
        return list(self._entries)

    def record(self, usage: Usage, catalog: dict | None = None) -> Entry:
        """Price one usage record, append it, and persist if a path is set."""
        pricing = get_pricing(usage.model, catalog)
        entry = Entry(usage=usage, cost=usage.cost(pricing), model_pricing=pricing.model)
        self._entries.append(entry)
        if self.path:
            line = json.dumps(
                {"usage": usage.model_dump(mode="json"), "cost": entry.cost,
                 "model_pricing": entry.model_pricing}
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return entry

    def total_cost(self, **filters) -> float:
        return sum(e.cost for e in self._filtered(**filters))

    def total_tokens(self, **filters) -> int:
        return sum(e.usage.total_tokens for e in self._filtered(**filters))

    def _filtered(self, **filters):
        for e in self._entries:
            if _matches(e, filters):
                yield e

    def save(self) -> None:
        """Rewrite the whole ledger as JSONL (compaction)."""
        if not self.path:
            raise ValueError("ledger has no path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for e in self._entries:
                line = json.dumps(
                    {"usage": e.usage.model_dump(mode="json"), "cost": e.cost,
                     "model_pricing": e.model_pricing}
                )
                f.write(line + "\n")

    def load(self, path: str | Path | None = None) -> None:
        """(Re)load entries from JSONL, replacing in-memory state."""
        p = Path(path) if path else self.path
        if p is None or not p.exists():
            self._entries = []
            return
        entries: list[Entry] = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                usage = Usage.model_validate(obj["usage"])
                entries.append(
                    Entry(usage=usage, cost=obj["cost"], model_pricing=obj["model_pricing"])
                )
        self._entries = entries


def _matches(entry: Entry, filters: dict) -> bool:
    """Filter entries by any of: model, project, api_key_id, since, until."""
    u = entry.usage
    if "model" in filters and u.model != filters["model"]:
        return False
    if "project" in filters and u.project != filters["project"]:
        return False
    if "api_key_id" in filters and u.api_key_id != filters["api_key_id"]:
        return False
    if "since" in filters and u.timestamp < filters["since"]:
        return False
    return not ("until" in filters and u.timestamp > filters["until"])


def default_catalog() -> dict:
    return PRICING
