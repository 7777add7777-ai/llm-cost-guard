"""llm-cost-guard: cost accounting, budgets, and over-spend protection."""

from .budget import Alert, Budget, CostGuard, Observation, OverBudgetError
from .ledger import CostLedger, Entry
from .parser import UnparseableResponseError, parse_openai_response, parse_response_json
from .pricing import PRICING, Pricing, UnknownModelError, get_pricing
from .report import ReportRow, SpendingReport
from .usage import Usage, utc_now

__version__ = "0.1.0"

__all__ = [
    "PRICING",
    "Alert",
    "Budget",
    "CostGuard",
    "CostLedger",
    "Entry",
    "Observation",
    "OverBudgetError",
    "Pricing",
    "ReportRow",
    "SpendingReport",
    "UnknownModelError",
    "UnparseableResponseError",
    "Usage",
    "get_pricing",
    "parse_openai_response",
    "parse_response_json",
    "utc_now",
    "__version__",
]
