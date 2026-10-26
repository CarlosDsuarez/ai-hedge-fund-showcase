import pytest
import pandas as pd
from unittest.mock import patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_get_stock_price_returns_none_on_exception():
    from data.fetcher import get_stock_price
    with patch("yfinance.Ticker") as mock_yf:
        mock_yf.return_value.history.side_effect = Exception("network error")
        result = get_stock_price("INVALID_XYZ")
    assert result is None, f"Expected None, got {result}"


def test_get_stock_price_does_not_raise():
    from data.fetcher import get_stock_price
    with patch("yfinance.Ticker") as mock_yf:
        mock_yf.return_value.history.side_effect = Exception("network error")
        try:
            get_stock_price("AAPL")
        except Exception as e:
            pytest.fail(f"get_stock_price raised: {e}")


def test_get_stock_price_returns_none_on_empty_df():
    from data.fetcher import get_stock_price
    with patch("yfinance.Ticker") as mock_yf:
        mock_yf.return_value.history.return_value = pd.DataFrame()
        result = get_stock_price("AAPL")
    assert result is None


def test_get_stock_info_returns_defaults_on_exception():
    from data.fetcher import get_stock_info
    with patch("data.fetcher._cache_get", return_value=None), \
         patch("yfinance.Ticker") as mock_yf:
        mock_yf.side_effect = Exception("connection refused")
        result = get_stock_info("AAPL")
    assert result["ticker"] == "AAPL"
    assert result["sector"] == "Unknown"
    assert result["pe_ratio"] is None


def test_get_stock_info_does_not_raise():
    from data.fetcher import get_stock_info
    with patch("data.fetcher._cache_get", return_value=None), \
         patch("yfinance.Ticker") as mock_yf:
        mock_yf.side_effect = Exception("timeout")
        try:
            get_stock_info("AAPL")
        except Exception as e:
            pytest.fail(f"get_stock_info raised: {e}")
