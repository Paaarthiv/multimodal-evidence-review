"""Cost estimation from token usage (for the operational analysis)."""
from __future__ import annotations

from . import config
from .providers import CallStats


def summarize_cost(stats: CallStats) -> dict:
    """Approximate USD cost using PRICING assumptions in config.

    Token totals are aggregated across providers; we attribute them to the model
    that made the most calls (a reasonable approximation for a mixed run, and
    exact for the common single-provider case).
    """
    if not stats.by_model:
        model = config.GEMINI_MODEL
    else:
        model = max(stats.by_model, key=stats.by_model.get)

    price = config.PRICING.get(model, {"input": 0.0, "output": 0.0})
    in_cost = stats.input_tokens / 1_000_000 * price["input"]
    out_cost = stats.output_tokens / 1_000_000 * price["output"]
    return {
        "assumed_model": model,
        "input_per_mtok": price["input"],
        "output_per_mtok": price["output"],
        "input_cost": round(in_cost, 4),
        "output_cost": round(out_cost, 4),
        "total_cost": round(in_cost + out_cost, 4),
    }
