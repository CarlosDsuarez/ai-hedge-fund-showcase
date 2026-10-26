"""data/fetcher.compute_beta_60d — moved out of tests/test_risk_sl_tp_agent.py
when the OCO audit (its only production caller) was retired. The function
stays in the fetcher for the chat and any future sizing overlay."""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_compute_beta_60d_returns_float():
    """Beta should be a positive float near 1.0 for SPY-correlated assets."""
    idx = pd.date_range("2026-01-01", periods=65, freq="B")
    spy_rets = pd.Series(np.random.normal(0.001, 0.01, 65), index=idx)
    ticker_rets = 1.2 * spy_rets + pd.Series(np.random.normal(0, 0.005, 65), index=idx)
    prices = pd.DataFrame({
        "LLY": (1 + ticker_rets).cumprod() * 1000,
        "SPY": (1 + spy_rets).cumprod() * 500,
    }, index=idx)

    # compute_beta_60d fetches each leg through the facade now, not with one
    # yf.download. Mocking the old call left this test reaching the real market:
    # it passed locally only because the fetch failed and returned the 1.0
    # default, and failed in CI on LLY's actual 3-month beta.
    def _fake_price(symbol, period="1y"):
        return pd.DataFrame({"Close": prices[symbol]}, index=idx)

    with patch("data.fetcher.get_stock_price", side_effect=_fake_price):
        from data.fetcher import compute_beta_60d
        beta = compute_beta_60d("LLY")
    assert isinstance(beta, float)
    assert 0.5 < beta < 3.0


def test_compute_beta_60d_default_on_error():
    from data.fetcher import compute_beta_60d
    with patch("data.fetcher.get_stock_price", side_effect=Exception("network error")):
        beta = compute_beta_60d("FAKETICKER")
    assert beta == 1.0


def test_compute_beta_60d_default_when_a_leg_is_missing():
    """A missing SPY leg must yield the 1.0 default, not a beta from one series."""
    from data.fetcher import compute_beta_60d
    with patch("data.fetcher.get_stock_price", return_value=None):
        assert compute_beta_60d("LLY") == 1.0
