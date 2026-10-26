import numpy as np
import pandas as pd

from agents.regime_agent import _compute_adx, detect_regime


def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    """Synthetic trending OHLCV: price rises 0.5/day from 100."""
    close = np.linspace(100, 130, n)
    high  = close + 1.0
    low   = close - 1.0
    return pd.DataFrame({"High": high, "Low": low, "Close": close})


def test_compute_adx_returns_float():
    df = _make_ohlcv(60)
    result = _compute_adx(df, period=14)
    assert isinstance(result, float), f"Expected float, got {type(result)}"


def test_compute_adx_in_valid_range():
    df = _make_ohlcv(60)
    adx = _compute_adx(df, period=14)
    assert 0.0 < adx < 100.0, f"ADX {adx} out of (0, 100)"


def test_compute_adx_trending_market_above_25():
    """A strongly directional trend should produce ADX > 25."""
    n = 80
    close = np.linspace(100, 180, n)
    high  = close + 0.5
    low   = close - 0.5
    df = pd.DataFrame({"High": high, "Low": low, "Close": close})
    adx = _compute_adx(df, period=14)
    assert adx > 25.0, f"Strong trend should have ADX > 25, got {adx:.1f}"


def test_compute_adx_returns_neutral_on_none():
    result = _compute_adx(None, period=14)
    assert result == 25.0, f"Should return 25.0 (neutral) on None, got {result}"


def test_compute_adx_returns_neutral_on_short_df():
    df = _make_ohlcv(10)  # too few rows (< period*3 = 42)
    result = _compute_adx(df, period=14)
    assert result == 25.0, f"Short df should return 25.0, got {result}"


def test_detect_regime_adx_in_range(synthetic_regime_market):
    """ADX stays in the result dict as observability only (Sprint 1, design A);
    it no longer gates or classifies anything."""
    r = detect_regime()
    adx = r["adx"]
    assert 0 < adx < 100, f"ADX {adx} out of (0, 100)"


# ── BEAR branch on missing data ─────────────────────────────────────────

def test_bear_signals_hold_all_on_empty_data(monkeypatch):
    """A data outage must not produce eight BUYs.

    _fetch_closes returns an empty frame now instead of raising, and without a
    guard every iMOM score falls to 0.0 while the `rank_i <= 8` cut still marks
    the first eight names BUY. main._signal_gate_ok accepts a regime BUY as
    sufficient evidence on its own, so those would have become real orders.
    """
    import pandas as pd

    import agents.regime_strategy_agent as rs

    monkeypatch.setattr(rs, "_fetch_closes", lambda *a, **k: pd.DataFrame())
    signals = rs._compute_bear_signals()

    buys = [t for t, v in signals.items() if v.get("signal") == "BUY"]
    assert buys == [], f"empty data must yield zero BUYs, got {len(buys)}"
    assert set(signals) == set(rs.BEAR_TICKERS)


def test_bear_signals_hold_all_when_returns_are_unusable(monkeypatch):
    """One row of prices yields no returns after dropna — same trap, later."""
    import pandas as pd

    import agents.regime_strategy_agent as rs

    one_row = pd.DataFrame({t: [100.0] for t in rs.BEAR_TICKERS[:3] + ["SPY", "IWM", "IWD", "IWG"]},
                           index=pd.to_datetime(["2026-08-31"]))
    monkeypatch.setattr(rs, "_fetch_closes", lambda *a, **k: one_row)
    signals = rs._compute_bear_signals()
    assert [t for t, v in signals.items() if v.get("signal") == "BUY"] == []
