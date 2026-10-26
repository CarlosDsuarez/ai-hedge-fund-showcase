"""Slot-based position sizing.

Pure arithmetic, no I/O, no broker: the same function sizes a live entry and a
simulated one in backtest/reference_backtest.py.

    size = slot × regime_mult × conviction_mult × vol_scalar × sector_mult
    slot = TARGET_GROSS_EXPOSURE / TARGET_POSITIONS

clamped to [MIN_POSITION_PCT, MAX_POSITION_PCT] unless the caller overrides
the caps.

Why slots: an equal-weight budget/N shrinks every position as the candidate
list grows. A fixed slot makes exposure a design choice instead of a
by-product of how many names happened to clear the gates that day.
"""
import math

import pandas as pd

import config


def vol_scalar(sigma) -> float:
    """SIZING_VOL_TARGET / sigma, clamped; neutral (1.0) when sigma is unusable.

    sigma is the annualised 63-day realised vol. Below 0.01 (or None/NaN) it
    is treated as unknown rather than scaled to the cap — a stale or flat
    series must not produce the largest position in the book.
    """
    if sigma is None or not math.isfinite(sigma) or sigma <= 0.01:
        return 1.0
    raw = config.SIZING_VOL_TARGET / sigma
    return max(config.SIZING_VOL_SCALAR_MIN, min(config.SIZING_VOL_SCALAR_MAX, raw))


def conviction_mult(conviction, *, degraded: bool = False) -> float:
    """PM conviction → slot multiplier; 1.0 when unknown, degraded or unmapped.

    Degraded runs (PM output missing/unparseable) fall back to neutral rather
    than to the lowest tier so a research outage does not silently shrink the
    book; the backtest passes conviction=None for the same neutral 1.0.
    """
    if degraded or conviction is None:
        return 1.0
    return float(config.CONVICTION_SIZE_MULT.get(conviction, 1.0))


def slot_size_pct(regime_mult, conviction, sigma, *, degraded: bool = False,
                  sector_mult: float = 1.0, max_pos=None, min_pos=None) -> float:
    """Target position size as a fraction of equity for one entry."""
    slot = config.TARGET_GROSS_EXPOSURE / config.TARGET_POSITIONS
    raw = (slot * float(regime_mult) * conviction_mult(conviction, degraded=degraded)
           * vol_scalar(sigma) * float(sector_mult))
    hi = config.MAX_POSITION_PCT if max_pos is None else max_pos
    lo = config.MIN_POSITION_PCT if min_pos is None else min_pos
    return max(lo, min(hi, raw))


def realized_vol_63d(closes: pd.DataFrame) -> dict[str, float]:
    """Annualised 63-day realised vol per column:

        returns[t].tail(63).std() * sqrt(252)

    A ticker with too little history or an all-NaN column produces a NaN
    std; that ticker is simply omitted so vol_scalar(None) falls back to a
    neutral 1.0 instead of a scaled-up bet on unusable data.
    """
    if closes is None or closes.empty:
        return {}
    vols = closes.pct_change().tail(63).std() * math.sqrt(252)
    return {t: float(v) for t, v in vols.items() if v is not None and math.isfinite(v)}


def plan_entries(candidates: list[str], *, scores: dict, convictions: dict, sigmas: dict,
                 regime_mult: float, equity: float, long_market_value: float,
                 deployable: float, degraded: bool = False,
                 sector_mults: "dict | None" = None,
                 ordered: bool = False) -> "tuple[list[dict], list[tuple[str, str]]]":
    """Fill the book toward TARGET_GROSS_EXPOSURE, one slot at a time.

    Candidates are taken in descending composite-score order (stable on
    ties) unless `ordered=True`, in which case the caller's own order is
    kept verbatim — main.py's pending queue is already interleaved
    pending-before-new by agents.rotation_agent.prioritize_deployments, and
    re-sorting that by score would put a low-score pending item behind a
    high-score new one, which is the one ordering the queue exists to
    prevent.

    `gross_room = TARGET_GROSS_EXPOSURE * equity - long_market_value` is how
    much more long market value the book can carry before it hits the slot
    target; `budget = min(gross_room, deployable)` is the smaller of that and
    what the account can actually fund this cycle. Whichever of the two binds
    is the reason recorded against every candidate that does not get a slot —
    decided once for the whole cycle, not re-evaluated per candidate, because
    it is a property of the budget, not of any one ticker.

    Returns (plan, rejected):
      plan     -- [{"ticker", "size_pct", "dollars"}, ...] in fill order.
      rejected -- [(ticker, "GROSS_TARGET_REACHED" | "NO_BUDGET"), ...] for
                  every candidate that did not clear the minimum ticket size.
    """
    sector_mults = sector_mults or {}
    order = list(candidates) if ordered else sorted(
        candidates, key=lambda t: scores.get(t, 0.0), reverse=True)

    gross_room = config.TARGET_GROSS_EXPOSURE * equity - long_market_value
    budget = min(gross_room, deployable)
    reason = "GROSS_TARGET_REACHED" if gross_room <= deployable else "NO_BUDGET"
    min_ticket = config.MIN_POSITION_PCT * equity

    plan: list = []
    rejected: list = []
    for t in order:
        size_pct = slot_size_pct(
            regime_mult, convictions.get(t), sigmas.get(t),
            degraded=degraded, sector_mult=sector_mults.get(t, 1.0),
        )
        dollars = min(size_pct * equity, budget)
        if dollars >= min_ticket:
            plan.append({"ticker": t, "size_pct": dollars / equity, "dollars": dollars})
            budget -= dollars
        else:
            rejected.append((t, reason))
    return plan, rejected
