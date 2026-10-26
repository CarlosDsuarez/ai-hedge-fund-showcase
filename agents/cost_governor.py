"""Pre-flight LLM spend governor (audit follow-up to agents/llm_usage.py).

llm_usage.py *accounts* for what we spent. This module *decides* whether we
may spend at all, and in which mode, before each Anthropic call.

The ladder is evaluated on the remaining daily budget as a fraction of
config.DAILY_LLM_BUDGET_USD:

    tier      remaining        behaviour
    FULL      > 40%            requested model + max_tokens, extended thinking on
    REDUCED   15–40%           max_tokens x 0.6, no thinking, deep dive top-K 12 → 6
    MINIMAL    5–15%           force CLAUDE_SENTIMENT_MODEL (Haiku), max_tokens x 0.4,
                               no deep dive
    BLOCKED   < 5%, or the pre-flight estimate exceeds what is left
                               allowed=False — the caller falls back to its
                               deterministic path

Two hard rules:
  * never raise — any internal failure degrades to an allow-everything Decision
    with reason="governor_error". A budget guard must not take down the pipeline.
  * never hammer SQLite — spent_today() is cached for _SPEND_TTL_SEC seconds.
"""

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import date

import config
from agents.llm_usage import _DEFAULT_PRICE, _PRICES

logger = logging.getLogger(__name__)

_SPEND_TTL_SEC = 60.0

# Extended thinking budget handed out at FULL tier. `thinking_budget=None`
# means "thinking disabled" — the caller simply omits the thinking parameter.
# The Anthropic API requires budget_tokens >= 1024 and max_tokens > budget,
# so calls with a small max_tokens get None even at FULL tier.
_FULL_THINKING_BUDGET = 1024

_lock = threading.Lock()
_spend_cache: dict[str, tuple[float, float]] = {}  # day -> (spent_usd, fetched_at)
_alerted: set[tuple[str, str]] = set()             # (agent, day) already alerted on


@dataclass
class Decision:
    """Verdict for one prospective call. `model`/`max_tokens` may be degraded."""

    allowed: bool
    model: str
    max_tokens: int
    thinking_budget: int | None
    tier: str
    reason: str


def _today() -> str:
    return date.today().isoformat()


def _sum_cost_usd(conn, day: str) -> float:
    """Sum llm_usage.cost_usd for `day`, deriving it from tokens if needed.

    cost_usd arrives with the Prompt A3 migration; until it lands we
    reconstruct the same number from tokens_in/tokens_out via _PRICES rather
    than silently reporting zero spend (which would disable the governor).
    """
    like = f"{day}%"
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM llm_usage WHERE ts LIKE ?",
            (like,),
        ).fetchone()
        return float(row[0] or 0.0)
    except sqlite3.OperationalError as e:
        if "no such column" not in str(e).lower():
            raise
        rows = conn.execute(
            "SELECT model, SUM(tokens_in), SUM(tokens_out) FROM llm_usage "
            "WHERE ts LIKE ? GROUP BY model",
            (like,),
        ).fetchall()
        total = 0.0
        for model, tokens_in, tokens_out in rows:
            p_in, p_out = _PRICES.get(model, _DEFAULT_PRICE)
            total += (tokens_in or 0) / 1e6 * p_in + (tokens_out or 0) / 1e6 * p_out
        return total


def spent_today(day: str | None = None) -> float:
    """Total USD spent on Anthropic calls during `day` (default: today, local time).

    Cached for 60s. Returns 0.0 if the DB is unreachable or the table is absent.
    """
    day = day or _today()
    now = time.monotonic()

    with _lock:
        cached = _spend_cache.get(day)
        if cached is not None and now - cached[1] < _SPEND_TTL_SEC:
            return cached[0]

    try:
        import db.database as db

        with db._conn() as conn:
            total = _sum_cost_usd(conn, day)
    except Exception:
        logger.warning("cost_governor spent_today failed for day=%s", day, exc_info=True)
        return 0.0

    with _lock:
        _spend_cache[day] = (total, now)
    return total


def remaining_today() -> float:
    """Budget left for today. May be negative if we overshot."""
    return config.DAILY_LLM_BUDGET_USD - spent_today()


def estimate_cost(model: str, est_tokens_in: int, est_max_tokens_out: int) -> float:
    """Worst-case pre-flight cost: assumes the output runs to max_tokens."""
    p_in, p_out = _PRICES.get(model, _DEFAULT_PRICE)
    return est_tokens_in / 1e6 * p_in + est_max_tokens_out / 1e6 * p_out


def est_tokens(*texts: str) -> int:
    """Input-token estimate at the usual 4-chars-per-token heuristic.

    Pass *every* prompt block that goes on the wire — system as well as the
    user message — or the pre-flight estimate under-counts and the governor
    green-lights calls it cannot afford.
    """
    return sum(len(t) for t in texts if t) // 4


def notify_degraded(agent: str, decision: Decision) -> None:
    """Telegram-alert that `agent` is not running at full strength.

    Fired by the main pipeline (research / sentiment / pm) on *every* non-FULL
    tier, not only on denials: MINIMAL quietly swaps Sonnet for Haiku and cuts
    the output ceiling to 40%, which moves trading decisions just as a block
    does. A run degraded where nobody looks is the failure this whole module
    exists to prevent.

    De-duplicated per (agent, day, tier) — research fans out one call per
    ticker across eight threads, and the ladder only ever steps downward
    within a day, so this is at most one message per step. Never raises.
    """
    try:
        key = (agent, _today(), decision.tier)
        with _lock:
            if key in _alerted:
                return
            _alerted.add(key)

        spent = spent_today()
        budget = float(config.DAILY_LLM_BUDGET_USD)
        if decision.allowed:
            headline = (
                f"<b>⚠️ LLM degradado</b>\n"
                f"Agente <b>{agent}</b> en tier {decision.tier} ({decision.reason}): "
                f"modelo {decision.model}, max_tokens {decision.max_tokens}."
            )
            tail = "Sigue operando, con menos capacidad de análisis de la habitual."
        else:
            headline = (
                f"<b>⚠️ Presupuesto LLM agotado</b>\n"
                f"Agente <b>{agent}</b> bloqueado — tier {decision.tier} ({decision.reason})."
            )
            tail = "El agente cae a su ruta determinista: este run va DEGRADADO."

        from alerts.telegram_alert import send_message

        delivered = send_message(
            f"{headline}\n"
            f"Gasto de hoy: ${spent:.4f} / ${budget:.2f} "
            f"(restante ${budget - spent:.4f}).\n"
            f"{tail}"
        )
        if not delivered:
            # send_message persists the text to logs/telegram_fallback.log; say
            # it here too so a console-only operator still sees the degradation.
            logger.error(
                "cost_governor alert UNDELIVERED agent=%s tier=%s spent=%.4f — run is degraded",
                agent, decision.tier, spent,
            )
    except Exception:
        logger.warning("cost_governor alert failed for agent=%s", agent, exc_info=True)


def reset_cache() -> None:
    """Drop the spend cache and alert de-dup (tests, and after a budget change)."""
    with _lock:
        _spend_cache.clear()
        _alerted.clear()


def _thinking_for(max_tokens: int) -> int | None:
    return _FULL_THINKING_BUDGET if max_tokens > _FULL_THINKING_BUDGET else None


def _classify(remaining: float, budget: float) -> str:
    frac = remaining / budget
    if frac < config.BUDGET_TIER_BLOCKED_PCT:
        return "BLOCKED"
    if frac < config.BUDGET_TIER_MINIMAL_PCT:
        return "MINIMAL"
    if frac < config.BUDGET_TIER_REDUCED_PCT:
        return "REDUCED"
    return "FULL"


def authorize(agent: str, model: str, est_tokens_in: int, est_max_tokens_out: int) -> Decision:
    """Decide whether `agent` may make this call, and under what constraints."""
    try:
        budget = float(config.DAILY_LLM_BUDGET_USD)
        if budget <= 0:
            # No cap configured — governance is off rather than "spend nothing".
            return Decision(True, model, est_max_tokens_out,
                            _thinking_for(est_max_tokens_out), "FULL", "budget_not_configured")

        remaining = remaining_today()
        tier = _classify(remaining, budget)
        estimate = estimate_cost(model, est_tokens_in, est_max_tokens_out)
        if tier != "BLOCKED" and estimate > remaining:
            tier = "BLOCKED"

        if not config.BUDGET_ENFORCEMENT:
            # Observation mode: log what we *would* have done, degrade nothing.
            if tier != "FULL":
                logger.warning(
                    "cost_governor agent=%s tier=%s remaining=%.4f model=%s→%s "
                    "max_tokens=%d→%d (enforcement disabled, not applied)",
                    agent, tier, remaining, model, model,
                    est_max_tokens_out, est_max_tokens_out,
                )
            return Decision(True, model, est_max_tokens_out,
                            _thinking_for(est_max_tokens_out), "FULL", "enforcement_disabled")

        if tier == "FULL":
            return Decision(True, model, est_max_tokens_out,
                            _thinking_for(est_max_tokens_out), "FULL", "within_budget")

        if tier == "BLOCKED":
            new_model, new_max_tokens, thinking = model, est_max_tokens_out, None
            reason = "budget_exhausted"
            allowed = False
        elif tier == "MINIMAL":
            new_model = config.CLAUDE_SENTIMENT_MODEL
            new_max_tokens = max(1, int(est_max_tokens_out * 0.4))
            thinking = None
            reason = "budget_minimal"
            allowed = True
        else:  # REDUCED
            new_model = model
            new_max_tokens = max(1, int(est_max_tokens_out * 0.6))
            thinking = None
            reason = "budget_reduced"
            allowed = True

        logger.warning(
            "cost_governor agent=%s tier=%s remaining=%.4f model=%s→%s max_tokens=%d→%d",
            agent, tier, remaining, model, new_model, est_max_tokens_out, new_max_tokens,
        )
        return Decision(allowed, new_model, new_max_tokens, thinking, tier, reason)

    except Exception:
        logger.warning("cost_governor authorize failed for agent=%s", agent, exc_info=True)
        return Decision(True, model, est_max_tokens_out, None, "FULL", "governor_error")
