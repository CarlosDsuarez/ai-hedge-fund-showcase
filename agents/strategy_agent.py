"""
strategy_agent.py — cross-sectional signal generator.

SHOWCASE: REFERENCE IMPLEMENTATION. The private version combines several
factors and a cross-asset macro filter with tuned weights; those are not
published. This file keeps the same public contract (function names,
arguments, and the per-ticker output keys main.py and chat/ read) and fills it
with a textbook signal:

  mom_Z   = cross-sectional z-score of 12-1 month price return
            (Jegadeesh & Titman, 1993: MOMENTUM_WINDOW-day return that skips
            the most recent MOMENTUM_SKIP days to avoid short-term reversal).
  val_Z   = 0.0  (value factor omitted in the showcase)
  sent_Z  = 0.0  (sentiment is not part of the composite; see config)

  composite = mom_Z
  BUY       = composite >= SIGNAL_THRESHOLD and the macro filter passes.

The macro filter is a stub that always passes (`_credit_stress_signal`).
"""
import logging
import os
import sys

import pandas as pd
from data.fetcher import get_closes
from utils.env import load_project_env

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

load_project_env()

logger = logging.getLogger(__name__)

_NEUTRAL = {"mom_z": 0.0, "val_z": 0.0, "sent_z": 0.0}


# ── Macro filter (stub) ─────────────────────────────────────────────────────

def _credit_stress_signal() -> tuple[bool, float]:
    """Return (is_stressed, stress_z).

    SHOWCASE: stub. The private implementation derives a credit-stress state
    from cross-asset data and suppresses new longs when it fires. Here it never
    fires, so the interface and the `macro_stressed` field stay wired through
    the pipeline without publishing the rule.
    """
    return False, 0.0


# ── Momentum signal: 12-1 month ────────────────────────────────────────────

def _compute_momentum(closes: pd.DataFrame) -> pd.Series:
    """12-1 momentum: return from t-MOMENTUM_WINDOW to t-MOMENTUM_SKIP per column."""
    window, skip = config.MOMENTUM_WINDOW, config.MOMENTUM_SKIP
    if len(closes) < window + 1:
        return pd.Series(dtype=float)
    px_start = closes.iloc[-window - 1].dropna()
    px_end = closes.iloc[-skip - 1].dropna()
    valid = px_start.index.intersection(px_end.index)
    valid = valid[px_start[valid] > 0]
    return (px_end[valid] / px_start[valid] - 1).dropna()


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / (std if std and std > 1e-9 else 1e-9)


# ── Main composite signal ───────────────────────────────────────────────────

def compute_fullalloc_signals(
    tickers: list[str],
    sentiment_scores: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Score `tickers` cross-sectionally.

    Returns a dict keyed by ticker with keys:
      signal, composite_score, mom_z, val_z, sent_z, macro_stressed, credit_spread_z
    """
    closes = get_closes(tickers, period="2y")
    if not closes.empty:
        closes = closes[[t for t in tickers if t in closes.columns]].ffill().dropna(how="all")

    macro_stressed, credit_z = _credit_stress_signal()
    def hold() -> dict:
        return {"signal": "HOLD", "composite_score": 0.0, **_NEUTRAL,
                "macro_stressed": macro_stressed, "credit_spread_z": credit_z}

    if closes.empty or len(closes) < config.MOMENTUM_WINDOW + 1:
        return {t: hold() for t in tickers}

    mom = _compute_momentum(closes)
    if len(mom) < 2:
        return {t: hold() for t in tickers}

    mom_z = _zscore(mom)
    results: dict[str, dict] = {}
    for t in tickers:
        if t not in mom_z.index:
            results[t] = hold()
            continue
        score = float(mom_z[t])
        buy = score >= config.SIGNAL_THRESHOLD and not macro_stressed
        results[t] = {
            "signal": "BUY" if buy else "HOLD",
            "composite_score": round(score, 4),
            "mom_z": round(score, 4),
            "val_z": 0.0,
            "sent_z": 0.0,
            "macro_stressed": macro_stressed,
            "credit_spread_z": credit_z,
        }
    return results


# ── Public API (backward-compatible with main.py) ──────────────────────────

def generate_signals(
    tickers: list[str],
    sentiment_scores: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Public entry point called by main.py and chat/.

    Returns per-ticker dicts with: signal, vwap_signal, mom_signal,
    composite_score, mom_z, val_z, sent_z, macro_stressed, credit_spread_z, strategy
    """
    signals = compute_fullalloc_signals(tickers, sentiment_scores=sentiment_scores)
    for data in signals.values():
        data["strategy"] = "Reference-Momentum-12-1"
        data["vwap_signal"] = data["signal"]   # backward-compat alias
        data["mom_signal"] = data["signal"]    # backward-compat alias
    return signals
