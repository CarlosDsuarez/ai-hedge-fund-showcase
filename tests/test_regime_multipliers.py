"""Pure regime helpers (Sprint 1, design section A).

`detect_regime` is BULL/BEAR only (task 4 retired the ADX 4-state ladder);
this file covers the pure pieces the daily backtest imports so live and
backtest cannot drift: the direction state machine (extracted, not
re-implemented), the breadth/VIX multipliers and their product. Fixtures that
exercise `detect_regime` end to end with the breadth fetch live now uses
directly, live in tests/test_regime_breadth.py.
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from agents import regime_agent
from agents.regime_agent import (
    _sanitize_cached,
    breadth_multiplier,
    breadth_series,
    direction_series,
    direction_state,
    size_multiplier_for,
    vix_multiplier,
)


# ── config additions ─────────────────────────────────────────────────────────



# ── breadth / VIX ladders ────────────────────────────────────────────────────



def test_breadth_multiplier_neutral_when_unknown():
    assert breadth_multiplier(None) == 1.0
    assert breadth_multiplier(float("nan")) == 1.0




def test_vix_multiplier_neutral_when_unknown():
    assert vix_multiplier(None) == 1.0
    assert vix_multiplier(float("nan")) == 1.0








# ── product with floor ───────────────────────────────────────────────────────





# ── direction state machine ──────────────────────────────────────────────────

def _reference_direction_loop(buffer_series):
    """Verbatim copy of the loop that lived inside detect_regime before the
    extraction. If direction_state ever diverges from this, the live regime
    changed silently."""
    confirmed_regime = 'BULL'
    candidate = None
    candidate_streak = 0
    days_in_regime = 0
    for buf in buffer_series:
        if buf > 0.02:
            signal = 'BULL'
        elif buf < -0.02:
            signal = 'BEAR'
        else:
            signal = None
        if signal is None:
            candidate = None
            candidate_streak = 0
        elif signal == confirmed_regime:
            days_in_regime += 1
            candidate = None
            candidate_streak = 0
        else:
            days_in_regime = 0
            if signal == candidate:
                candidate_streak += 1
                if candidate_streak >= 5:
                    confirmed_regime = candidate
                    days_in_regime = candidate_streak
                    candidate = None
                    candidate_streak = 0
            else:
                candidate = signal
                candidate_streak = 1
    return confirmed_regime, days_in_regime


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_direction_state_matches_reference_loop(seed):
    rng = np.random.default_rng(seed)
    # Slow-moving buffer so both ±2% bands and the 5-day confirmation get exercised.
    buf = pd.Series(np.cumsum(rng.normal(0, 0.01, 300)))
    assert direction_state(buf) == _reference_direction_loop(buf)


def test_direction_state_needs_five_clear_days_to_flip():
    buf = pd.Series([0.05] * 10 + [-0.05] * 4)
    assert direction_state(buf) == ("BULL", 0)
    buf = pd.Series([0.05] * 10 + [-0.05] * 5)
    assert direction_state(buf) == ("BEAR", 5)


def test_direction_state_ignores_days_inside_the_buffer():
    buf = pd.Series([0.05] * 10 + [-0.05] * 3 + [0.0] + [-0.05] * 4)
    assert direction_state(buf)[0] == "BULL"   # streak reset by the neutral day


def test_direction_state_empty_series_is_bull():
    assert direction_state(pd.Series(dtype=float)) == ("BULL", 0)


def test_detect_regime_uses_direction_state():
    """detect_regime must call the extracted machine, not keep a private copy."""
    n = 260
    idx = pd.bdate_range("2025-01-01", periods=n)
    spy = pd.Series(np.linspace(400, 500, n), index=idx)
    vix = pd.Series(18.0, index=idx)
    cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Open"], ["SPY", "^VIX"]])
    raw = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for f in ("Close", "High", "Low", "Open"):
        raw[(f, "SPY")] = spy.values + (1.0 if f == "High" else -1.0 if f == "Low" else 0.0)
        raw[(f, "^VIX")] = vix.values
    calls = []
    real = regime_agent.direction_state

    def spy_direction_state(buffer_series):
        calls.append(len(buffer_series))
        return real(buffer_series)

    with patch.object(regime_agent.yf, "download", return_value=raw), \
         patch.object(regime_agent, "_fetch_universe_closes", return_value=pd.DataFrame()), \
         patch.object(regime_agent, "_load_cache", return_value={}), \
         patch.object(regime_agent, "_save_cache"), \
         patch.object(regime_agent, "direction_state", side_effect=spy_direction_state):
        result = regime_agent.detect_regime()
    assert calls, "detect_regime did not call direction_state"
    assert result["regime"] in ("BULL", "BEAR")
    assert result["days_in_regime"] == real(
        ((spy.rolling(20).mean() - spy.ewm(span=200, adjust=False).mean())
         / spy.ewm(span=200, adjust=False).mean()).dropna()
    )[1]


# ── vectorised series ────────────────────────────────────────────────────────

def test_direction_series_last_value_matches_direction_state():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2022-01-03", periods=700)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(-0.0005, 0.012, 700))), index=idx)
    series = direction_series(spy)
    assert isinstance(series, pd.Series)
    assert series.index.equals(spy.index)
    assert set(series.unique()) <= {"BULL", "BEAR"}
    ema200 = spy.ewm(span=200, adjust=False).mean()
    sma20 = spy.rolling(20).mean()
    buf = ((sma20 - ema200) / ema200).dropna()
    assert series.iloc[-1] == direction_state(buf)[0]


def test_direction_series_flips_to_bear_in_a_crash():
    idx = pd.bdate_range("2022-01-03", periods=500)
    px = np.concatenate([np.linspace(100, 130, 300), np.linspace(130, 70, 200)])
    series = direction_series(pd.Series(px, index=idx))
    assert series.iloc[100] == "BULL"
    assert series.iloc[-1] == "BEAR"
    # Flips are confirmed, not instantaneous: the first BEAR day comes after
    # at least 5 consecutive clear BEAR signals.
    first_bear = int(np.argmax(series.values == "BEAR"))
    assert first_bear > 300


def test_breadth_series_is_fraction_above_own_sma200():
    idx = pd.bdate_range("2022-01-03", periods=260)
    closes = pd.DataFrame({
        "UP":   np.linspace(100, 200, 260),
        "DOWN": np.linspace(200, 100, 260),
        "FLAT": np.full(260, 100.0),
    }, index=idx)
    b = breadth_series(closes)
    assert isinstance(b, pd.Series)
    sma = closes.rolling(200).mean()
    expected = (closes > sma).where(sma.notna()).mean(axis=1)
    pd.testing.assert_series_equal(b, expected)
    assert b.iloc[-1] == pytest.approx(1 / 3)   # only UP is above its own SMA200


def test_breadth_series_excludes_columns_without_an_sma200():
    """A name whose fetch failed and left an all-NaN column must not count as
    'below its SMA' and drag breadth down — it should simply be excluded.

    Task 1 review ruling: breadth_series must use
    `(px > sma).where(sma.notna())`, not a bare comparison, precisely so a
    missing column cannot bias breadth toward the weak tier.
    """
    idx = pd.bdate_range("2022-01-03", periods=260)
    closes = pd.DataFrame({
        "UP":      np.linspace(100, 200, 260),
        "DOWN":    np.linspace(200, 100, 260),
        "MISSING": np.nan,   # failed download: no price history at all
    }, index=idx)
    b = breadth_series(closes)
    # Without the fix, MISSING (always False, since NaN > NaN is False) would
    # pull the last day down to 1/3 instead of the true 1/2 among the two
    # names that actually have a trend measured.
    assert b.iloc[-1] == pytest.approx(0.5)


# ── _sanitize_cached (Task 4 fix round 1) ────────────────────────────────
# A cached regime_state.json can be pre-Task-4 (or otherwise stale): a
# CHOPPY/SIDEWAYS label, new_entries_allowed=False, a size_multiplier baked
# from the retired ADX ladder. _sanitize_cached is what every _load_cache()
# consumer in detect_regime() must run its result through before returning
# it — pinned here directly, independent of the end-to-end fallback-site
# tests in tests/test_regime_breadth.py.

def test_sanitize_cached_coerces_unknown_regime_to_bull():
    for stale_regime in ("CHOPPY", "SIDEWAYS", "", None, "WHATEVER"):
        out = _sanitize_cached({"regime": stale_regime})
        assert out["regime"] == "BULL"


def test_sanitize_cached_preserves_a_valid_regime():
    out = _sanitize_cached({"regime": "BEAR"})
    assert out["regime"] == "BEAR"


def test_sanitize_cached_forces_entries_open_and_drops_trend_strength():
    out = _sanitize_cached({
        "regime": "CHOPPY", "trend_strength": "NONE", "new_entries_allowed": False,
    })
    assert out["new_entries_allowed"] is True
    assert "trend_strength" not in out


def test_sanitize_cached_recomputes_size_multiplier_through_the_helper():
    """The stored size_multiplier is never trusted — it is always rederived
    from size_multiplier_for so it can never sit below REGIME_SIZE_FLOOR."""
    out = _sanitize_cached({
        "regime": "BEAR", "breadth_pct": 0.10, "vix": 50.0, "vix_sma10": 30.0,
        "size_multiplier": 0.01,   # legacy value that must be discarded
    })
    expected = size_multiplier_for("BEAR", 0.10, 50.0, 30.0, mode="product")
    assert out["size_multiplier"] == pytest.approx(expected)
    assert out["size_multiplier"] >= config.REGIME_SIZE_FLOOR


