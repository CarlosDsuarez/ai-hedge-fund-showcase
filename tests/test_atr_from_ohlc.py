"""atr_from_ohlc: the one ATR formula live and backtest share.

The daily backtest sizes stops off get_atr's formula (ewm span=14), so it
is factored out here and get_atr calls it. The simple-mean get_atr_stop that
used to sit beside it was retired with the legacy exit branch.
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetcher import atr_from_ohlc, get_atr


def _ohlc(n=60, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    return high, low, close


def _reference_ewm_atr(high, low, close, period):
    """The exact expression get_atr used before the extraction."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def test_atr_from_ohlc_series_matches_get_atr_formula():
    high, low, close = _ohlc()
    got = atr_from_ohlc(high, low, close, period=14)
    assert isinstance(got, pd.Series)
    pd.testing.assert_series_equal(got, _reference_ewm_atr(high, low, close, 14), check_names=False)


def test_atr_from_ohlc_dataframe_is_column_wise():
    panels = {t: _ohlc(seed=i) for i, t in enumerate(["AAA", "BBB", "CCC"])}
    high = pd.DataFrame({t: p[0] for t, p in panels.items()})
    low = pd.DataFrame({t: p[1] for t, p in panels.items()})
    close = pd.DataFrame({t: p[2] for t, p in panels.items()})
    got = atr_from_ohlc(high, low, close, period=14)
    assert isinstance(got, pd.DataFrame)
    assert list(got.columns) == ["AAA", "BBB", "CCC"]
    for t, (h, lo, c) in panels.items():
        pd.testing.assert_series_equal(got[t], _reference_ewm_atr(h, lo, c, 14), check_names=False)


def test_atr_from_ohlc_respects_period():
    high, low, close = _ohlc()
    a14 = atr_from_ohlc(high, low, close, period=14)
    a20 = atr_from_ohlc(high, low, close, period=20)
    assert not np.allclose(a14.values[20:], a20.values[20:])
    pd.testing.assert_series_equal(a20, _reference_ewm_atr(high, low, close, 20), check_names=False)


def test_get_atr_delegates_to_atr_from_ohlc():
    high, low, close = _ohlc()
    hist = pd.DataFrame({"High": high, "Low": low, "Close": close, "Open": close, "Volume": 1000})
    with patch("data.fetcher.get_stock_price", return_value=hist):
        got = get_atr("AAPL", period=14)
    assert got == pytest.approx(float(_reference_ewm_atr(high, low, close, 14).iloc[-1]))
    assert got == pytest.approx(float(atr_from_ohlc(high, low, close, 14).iloc[-1]))


def test_get_atr_still_returns_zero_on_short_history():
    high, low, close = _ohlc(n=10)
    hist = pd.DataFrame({"High": high, "Low": low, "Close": close})
    with patch("data.fetcher.get_stock_price", return_value=hist):
        assert get_atr("AAPL", period=14) == 0.0
