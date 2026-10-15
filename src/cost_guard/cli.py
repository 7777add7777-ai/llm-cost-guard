"""coguard: offline demo — a simulated workload burns through a $5 budget."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import UTC, datetime, timedelta

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .budget import Alert, Budget, CostGuard, OverBudgetError
from .ledger import CostLedger
from .parser import parse_openai_response
from .report import SpendingReport

BASE = datetime(2026, 10, 15, 9, 0, 0, tzinfo=UTC)


def fake_chat_response(model: str, prompt_tokens: int, completion_tokens: int,
                       cached: int) -> dict:
    """A deterministic stand-in for a real /chat/completions response body."""
    return {
        "id": "chatcmpl-demo",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def simulate_day(seed: int = 7) -> list[dict]:
    """Generate one deterministic day of traffic across projects and models."""
    rng = random.Random(seed)
    events: list[dict] = []
    models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
    projects = ["support-bot", "doc-summarizer", "eval-harness"]
    t = BASE
    for _ in range(140):
        t = t + timedelta(seconds=rng.randint(20, 90))
        model = rng.choice(models)
        project = rng.choice(projects)
        prompt = rng.randint(2000, 80000)
        cached = int(prompt * rng.random() * 0.6)
        completion = rng.randint(200, 10000)
        events.append(
            {
                "ts": t,
                "body": fake_chat_response(model, prompt, completion, cached),
                "project": project,
            }
        )
    return events


def run_demo() -> int:
    console = Console(width=110, no_color=True)
    budget = Budget(limit_usd=5.0, per_model_usd=2.5, daily_usd=4.0)
    alerts: list[Alert] = []
    guard = CostGuard(budget, policy="block", on_alert=alerts.append)
    ledger = CostLedger()

    console.rule("[bold cyan]coguard — LLM cost guard, simulated day")
    console.print(
        f"budget: total=${budget.limit_usd}  per-model=${budget.per_model_usd}  "
        f"day=${budget.daily_usd}  policy=block"
    )
    console.print()

    blocked_at: str | None = None
    for ev in simulate_day():
        usage = parse_openai_response(
            ev["body"], project=ev["project"], timestamp=ev["ts"]
        )
        entry = ledger.record(usage)
        try:
            guard.observe(usage, entry.cost)
        except OverBudgetError as exc:
            blocked_at = f"{usage.timestamp:%H:%M:%S}  ({exc.dimension})"
            break

    report = SpendingReport(ledger)
    console.print(Panel.fit(f"[bold]Spent ${report.total_cost:.4f} "
                            f"on {report.total_requests} requests "
                            f"({report.total_tokens:,} tokens)"
                            + (f"\nBlocked at {blocked_at}" if blocked_at else "")))
    console.print()

    console.print("[bold]Spend by model:[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("model")
    table.add_column("cost", justify="right")
    table.add_column("requests", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("share", justify="right")
    for row in report.top_models(10):
        table.add_row(
            row.key, f"${row.cost:.4f}", str(row.requests),
            f"{row.tokens:,}", f"{row.share:.0%}",
        )
    console.print(table)
    console.print()

    console.print("[bold]Alerts fired:[/bold]")
    for a in alerts:
        icon = "[yellow]WARN[/yellow]" if a.level == "warn" else "[red]BLOCK[/red]"
        console.print(f"  {icon} {a.message}")
    console.print()
    console.print(f"ledger: {len(ledger)} entries recorded (in-memory demo)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coguard", description="LLM cost guard")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="simulate a day of traffic against a $5 budget")
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        return run_demo()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
