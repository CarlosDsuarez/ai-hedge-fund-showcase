import pandas as pd
import pytest
from unittest.mock import patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_hist(n=30, high=105.0, low=95.0, close=100.0):
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"High": [high]*n, "Low": [low]*n, "Close": [close]*n,
         "Open": [close]*n, "Volume": [1000]*n},
        index=dates,
    )


def test_unknown_position_is_registered_with_a_2_5x_atr_stop():
    """A broker position with no state record gets the engine's own stop:
    entry - 2.5 × ATR14 (data.fetcher.get_atr, EMA span 14 over 3mo), not
    the retired legacy branch's 2.0 × simple-mean ATR or a flat -5%."""
    import config
    from data.fetcher import get_atr

    positions = [{
        "symbol": "AAPL",
        "current_price": "104.0",
        "avg_entry_price": "100.0",
        "qty": "10",
        "unrealized_plpc": "0.04",
        "market_value": "1040.0",
    }]
    hist = _make_hist(30, high=102.0, low=98.0, close=100.0)   # true range 4.0 every bar
    saved = {}
    with patch("data.fetcher.get_stock_price", return_value=hist), \
         patch("execution.order_manager.load_position_state", return_value={}), \
         patch("execution.order_manager.save_position_state", side_effect=saved.update), \
         patch("execution.trade_executor.sell_and_confirm") as sell:
        expected_atr = get_atr("AAPL", 14, "3mo")
        from execution.order_manager import run_exit_engine
        exited, closed = run_exit_engine(positions, score_map={}, pm_sells={}, metadata={})

    sell.assert_not_called()
    assert exited == set()
    rec = saved["AAPL"]
    assert expected_atr == pytest.approx(4.0)
    assert rec["atr_14"] == pytest.approx(expected_atr)
    assert rec["initial_stop"] == pytest.approx(100.0 - config.STOP_ATR_MULT * 4.0)   # 90.0
    assert rec["trail_stop"] == pytest.approx(max(90.0, 104.0 - config.TRAIL_ATR_MULT * 4.0))
