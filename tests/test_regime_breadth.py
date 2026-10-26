"""detect_regime end to end with the universe breadth fetch it now runs
directly (Sprint 1, task 4).

The pure pieces (breadth_series, breadth_multiplier, vix_multiplier,
size_multiplier_for) are pinned in tests/test_regime_multipliers.py; this file
drives detect_regime() itself through fixed SPY/^VIX and universe-closes data
so the wiring — direction x breadth x VIX, never blocking an entry — is
exercised without any real network or market dependency.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from agents import regime_agent


def _spy_vix_raw(spy_close: np.ndarray, vix_values) -> pd.DataFrame:
    n = len(spy_close)
    idx = pd.bdate_range("2024-01-01", periods=n)
    cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Open"], ["SPY", "^VIX"]])
    raw = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for f in ("Close", "High", "Low", "Open"):
        raw[(f, "SPY")] = spy_close + (1.0 if f == "High" else -1.0 if f == "Low" else 0.0)
        raw[(f, "^VIX")] = vix_values
    return raw


def _bull_close(n: int = 260) -> np.ndarray:
    return np.linspace(300, 500, n)


def _bear_close() -> np.ndarray:
    """300 days of confirmed uptrend, then a 200-day crash — the same shape
    validated in test_regime_multipliers.test_direction_series_flips_to_bear_in_a_crash,
    which confirms BEAR well before the series ends."""
    return np.concatenate([np.linspace(100, 130, 300), np.linspace(130, 70, 200)])


def _universe_closes(frac_above: float, n_names: int = 40, n_days: int = 260) -> pd.DataFrame:
    """Wide closes frame where exactly `frac_above` of names sit above their
    own SMA200 on the last day and the rest sit below."""
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    n_up = round(n_names * frac_above)
    cols = {}
    for i in range(n_names):
        if i < n_up:
            cols[f"UP{i}"] = np.linspace(100, 200, n_days)
        else:
            cols[f"DN{i}"] = np.linspace(200, 100, n_days)
    return pd.DataFrame(cols, index=idx)


def _detect(monkeypatch, raw: pd.DataFrame, universe_closes) -> dict:
    """`universe_closes` is either a DataFrame to return, or a zero-arg
    callable (e.g. one that raises) standing in for _fetch_universe_closes."""
    monkeypatch.setattr(regime_agent.yf, "download", lambda *a, **k: raw)
    fetch = universe_closes if callable(universe_closes) else lambda: universe_closes
    monkeypatch.setattr(regime_agent, "_fetch_universe_closes", fetch)
    monkeypatch.setattr(regime_agent, "_load_cache", lambda: {})
    monkeypatch.setattr(regime_agent, "_save_cache", lambda *a, **k: None)
    return regime_agent.detect_regime()


# ── BULL, calm VIX ────────────────────────────────────────────────────────

def test_bull_strong_breadth_calm_vix_sizes_at_full(monkeypatch):
    raw = _spy_vix_raw(_bull_close(), 16.0)
    closes = _universe_closes(0.70)
    r = _detect(monkeypatch, raw, closes)
    assert r["regime"] == "BULL"
    assert r["new_entries_allowed"] is True
    assert r["breadth_pct"] == pytest.approx(0.70)
    assert r["size_multiplier"] == pytest.approx(1.0)




# ── BEAR, extreme VIX (crash-brake) ──────────────────────────────────────





# ── robustness: never block on a data problem ────────────────────────────

def test_universe_fetch_failure_falls_back_to_neutral_breadth(monkeypatch):
    raw = _spy_vix_raw(_bull_close(), 16.0)

    def _boom():
        raise RuntimeError("alpaca batch failed")

    r = _detect(monkeypatch, raw, _boom)
    assert r["breadth_pct"] == pytest.approx(1.0)
    assert r["new_entries_allowed"] is True
    # BULL, neutral breadth, calm VIX -> full size; the failure must not
    # silently shrink or block sizing.
    assert r["size_multiplier"] == pytest.approx(1.0)


def test_universe_too_thin_falls_back_to_neutral_breadth(monkeypatch):
    raw = _spy_vix_raw(_bull_close(), 16.0)
    thin = _universe_closes(0.5, n_names=10)   # < 20 columns
    r = _detect(monkeypatch, raw, thin)
    assert r["breadth_pct"] == pytest.approx(1.0)
    assert r["new_entries_allowed"] is True


def _stale_pre_task4_cache() -> dict:
    """A pre-Task-4 regime_state.json: 4-state regime label, entries blocked,
    a size_multiplier baked from the retired ADX ladder (below
    REGIME_SIZE_FLOOR), and the removed trend_strength field. This is the
    exact shape the Task 4 review found this machine's actual gitignored
    logs/regime_state.json left in by a run before this fix."""
    return {
        "regime": "CHOPPY",
        "trend_strength": "NONE",
        "adx": 11.29,
        "new_entries_allowed": False,
        "spy_price": 760.52,
        "spy_ema200": 717.32,
        "spy_sma20": 765.0,
        "buffer_pct": 0.0665,
        "confirmed": True,
        "days_in_regime": 307,
        "switched_today": False,
        "vix": 16.69,
        "vol_regime": "MED",
        "size_multiplier": 0.25,
        "credit_signal": "NORMAL",
        "switch_count_ytd": 0,
        "switch_year": 2026,
    }


# ── stale cache sanitation (Task 4 fix round 1) ──────────────────────────
# _load_cache() can hand detect_regime() a pre-Task-4 (or otherwise stale)
# result: a CHOPPY/SIDEWAYS label, new_entries_allowed=False, a
# sub-REGIME_SIZE_FLOOR size_multiplier. Every fallback site that returns a
# cached result must sanitize it first (_sanitize_cached) — one test per
# site below.

def test_stale_cache_never_blocks_entries_when_download_is_empty(monkeypatch):
    """raw.empty fallback site."""
    monkeypatch.setattr(regime_agent.yf, "download", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(regime_agent, "_load_cache", lambda: _stale_pre_task4_cache())
    monkeypatch.setattr(regime_agent, "_save_cache", lambda *a, **k: None)

    r = regime_agent.detect_regime()

    assert r["regime"] == "BULL"
    assert r["new_entries_allowed"] is True
    assert r["size_multiplier"] >= config.REGIME_SIZE_FLOOR
    assert "trend_strength" not in r


def test_stale_cache_never_blocks_entries_when_history_is_short(monkeypatch):
    """len(closes) < 220 fallback site."""
    short_history = _spy_vix_raw(_bull_close(n=50), 16.0)
    monkeypatch.setattr(regime_agent.yf, "download", lambda *a, **k: short_history)
    monkeypatch.setattr(regime_agent, "_load_cache", lambda: _stale_pre_task4_cache())
    monkeypatch.setattr(regime_agent, "_save_cache", lambda *a, **k: None)

    r = regime_agent.detect_regime()

    assert r["regime"] == "BULL"
    assert r["new_entries_allowed"] is True
    assert r["size_multiplier"] >= config.REGIME_SIZE_FLOOR
    assert "trend_strength" not in r


def test_stale_cache_never_blocks_entries_on_download_error(monkeypatch):
    """Outer `except Exception` fallback site — the exact reproduction from
    the Task 4 review finding: yf.download raises and _load_cache() returns
    a pre-Task-4 dict. Before the fix this made detect_regime() return the
    stale CHOPPY/blocked/sub-floor dict verbatim, which made
    main.run_entries reject every candidate with REGIME_BLOCKED."""
    def _boom(*a, **k):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(regime_agent.yf, "download", _boom)
    monkeypatch.setattr(regime_agent, "_load_cache", lambda: _stale_pre_task4_cache())
    monkeypatch.setattr(regime_agent, "_save_cache", lambda *a, **k: None)

    r = regime_agent.detect_regime()

    assert r["regime"] == "BULL"
    assert r["new_entries_allowed"] is True
    assert r["size_multiplier"] >= config.REGIME_SIZE_FLOOR
    assert "trend_strength" not in r


