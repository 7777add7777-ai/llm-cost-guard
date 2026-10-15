# llm-cost-guard

Cost accounting, budgets, and **over-spend protection** for LLM workloads
that call OpenAI-compatible APIs. Offline-first, no network, no secrets.

```
provider response ──> parse ──> Usage ──> CostLedger (JSONL, priced)
                                       └──> CostGuard ──> warn / block
```

## Why

Every team that ships LLM features eventually asks "how much is this
costing, and what happens when it goes over budget?" This package is the
boring, reliable answer:

- **Pricing** — per-1M-token rates with prompt-cache discounts; dated model
  ids (`gpt-4o-mini-2024-07-18`) resolve to their catalog entry; custom
  catalogs and a zero-price default for self-hosted models.
- **Ledger** — append-only JSONL, the single source of truth for spend.
  Loadable after restart, filterable by model / project / key / time.
- **Guard** — limits (total, per-model, daily), a `warn` level at 80%,
  and a policy that either **warns** or **blocks** with
  `OverBudgetError`. Alert callbacks fire exactly once per crossing.
- **Reports** — group-by model/project/day, top-N with share, CSV export.
- **Parser** — understands chat/completions and embeddings bodies,
  including `prompt_tokens_details.cached_tokens`.

## Quickstart

```bash
pip install "llm-cost-guard[dev]"
coguard demo    # simulated day of traffic against a $5 budget
```

```python
from cost_guard import (
    Budget, CostGuard, CostLedger, get_pricing,
    parse_openai_response, OverBudgetError,
)

ledger = CostLedger("spend.jsonl")
guard = CostGuard(Budget(limit_usd=5.0, per_model_usd=2.5, daily_usd=4.0),
                  policy="block")

def after_response(resp: dict, project: str):
    usage = parse_openai_response(resp, project=project)
    ledger.record(usage)
    try:
        guard.observe(usage, usage.cost(get_pricing(usage.model)))
    except OverBudgetError as exc:
        raise SystemExit(f"spending halted: {exc}")
```

Wire `after_response` into your HTTP client's response hook and you have
budget enforcement on any OpenAI-compatible endpoint (OpenAI, Azure, vLLM,
Ollama, LiteLLM) without changing call sites.

## Testing

```bash
python -m pytest   # 50+ offline tests, no network
ruff check .
```

## Architecture notes

- The ledger never enforces budgets; the guard does. That split means the
  ledger stays a pure accounting record (auditable, replayable) while
  enforcement policy stays swappable.
- Daily limits roll over on the UTC day of the usage timestamp; the clock
  is injectable, so time-dependent behavior is deterministic in tests.
- Built-in prices are public list prices for reference. In production,
  pass your own `Pricing` catalog — contract pricing differs.

## License

MIT — see [LICENSE](LICENSE).
