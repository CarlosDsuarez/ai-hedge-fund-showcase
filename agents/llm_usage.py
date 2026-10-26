"""Token/cost tracking for Anthropic API calls (audit Fase A item 6).

Every LLM call site records usage here; totals accumulate per process so
main can print a per-run cost summary. Recording must never break the
pipeline — all failures degrade to a warning.
"""

import logging
import threading

logger = logging.getLogger(__name__)

# $/MTok (input, output) — keep in sync with the models in config.py
_PRICES = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}
_DEFAULT_PRICE = (3.0, 15.0)  # unknown model: assume Sonnet pricing (conservative)

_lock = threading.Lock()
_totals = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cost_usd": 0.0}


def record_usage(agent: str, model: str, response) -> None:
    """Log and persist token usage from an Anthropic Message response."""
    try:
        u = response.usage
        tokens_in = int(getattr(u, "input_tokens", 0) or 0)
        tokens_out = int(getattr(u, "output_tokens", 0) or 0)
        cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        p_in, p_out = _PRICES.get(model, _DEFAULT_PRICE)
        cost = tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out

        with _lock:
            _totals["calls"] += 1
            _totals["input"] += tokens_in
            _totals["output"] += tokens_out
            _totals["cache_read"] += cache_read
            _totals["cost_usd"] += cost

        logger.info(
            "llm_usage agent=%s model=%s in=%d out=%d cache_read=%d cost_usd=%.6f",
            agent, model, tokens_in, tokens_out, cache_read, cost,
        )

        from db.database import save_llm_usage
        save_llm_usage(agent, model, tokens_in, tokens_out, cache_read)
    except Exception:
        logger.warning("llm_usage recording failed for agent=%s", agent, exc_info=True)


def run_summary() -> dict:
    """Snapshot of accumulated usage for this process/run."""
    with _lock:
        return dict(_totals)


def reset() -> None:
    with _lock:
        for k in _totals:
            _totals[k] = 0.0 if k == "cost_usd" else 0
