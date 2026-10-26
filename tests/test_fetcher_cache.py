import json
import time
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _sample_hist():
    return pd.DataFrame(
        {"High": [105.0], "Low": [95.0], "Close": [100.0],
         "Open": [100.0], "Volume": [1000]},
        index=pd.date_range("2025-01-01", periods=1),
    )


def test_cache_key_function_exists():
    from data.fetcher import _cache_key
    from pathlib import Path
    p = _cache_key("get_stock_price", "AAPL", period="1y")
    assert isinstance(p, Path)


def test_second_call_within_ttl_skips_yfinance(tmp_path, monkeypatch):
    import data.fetcher as fetcher
    monkeypatch.setattr(fetcher, "_CACHE_DIR", tmp_path)
    hist = _sample_hist()
    with patch("yfinance.Ticker") as mock_yf:
        mock_yf.return_value.history.return_value = hist
        _ = fetcher.get_stock_price("AAPL", period="1y")
        assert mock_yf.call_count == 1
        r2 = fetcher.get_stock_price("AAPL", period="1y")
        assert mock_yf.call_count == 1, "Second call should NOT hit yfinance"
    assert r2 is not None
    assert "Close" in r2.columns


def test_expired_cache_calls_yfinance_again(tmp_path, monkeypatch):
    import data.fetcher as fetcher
    monkeypatch.setattr(fetcher, "_CACHE_DIR", tmp_path)
    hist = _sample_hist()
    key = fetcher._cache_key("get_stock_price", "AAPL", period="1y")
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps({"ts": time.time() - 7200, "data": hist.to_json(), "is_df": True}))
    with patch("yfinance.Ticker") as mock_yf:
        mock_yf.return_value.history.return_value = hist
        fetcher.get_stock_price("AAPL", period="1y")
        assert mock_yf.call_count == 1, "Expired cache should trigger yfinance fetch"


def test_cache_miss_calls_yfinance(tmp_path, monkeypatch):
    import data.fetcher as fetcher
    monkeypatch.setattr(fetcher, "_CACHE_DIR", tmp_path)
    hist = _sample_hist()
    with patch("yfinance.Ticker") as mock_yf:
        mock_yf.return_value.history.return_value = hist
        fetcher.get_stock_price("MSFT", period="1y")
        assert mock_yf.call_count == 1


def test_get_stock_info_cached(tmp_path, monkeypatch):
    import data.fetcher as fetcher
    monkeypatch.setattr(fetcher, "_CACHE_DIR", tmp_path)
    fake_info = {"longName": "Apple Inc", "sector": "Technology",
                 "trailingPE": 28.5, "returnOnEquity": 0.15,
                 "revenueGrowth": 0.08, "marketCap": 3e12, "freeCashflow": 1e11}
    with patch("yfinance.Ticker") as mock_yf:
        mock_yf.return_value.info = fake_info
        _ = fetcher.get_stock_info("AAPL")
        assert mock_yf.call_count == 1
        r2 = fetcher.get_stock_info("AAPL")
        assert mock_yf.call_count == 1, "Second call should NOT hit yfinance"
    assert r2["sector"] == "Technology"


def test_yfinance_leg_is_forced_even_when_alpaca_credentials_are_present(tmp_path, monkeypatch):
    """Reproduction of the main-checkout failure, with neither .env nor network.

    fetcher prefers Alpaca Market Data whenever ALPACA_API_KEY/SECRET are set,
    which in the main checkout is always. The three cache tests above mock
    yfinance and count its calls; there they counted 0, because Alpaca served
    250 real rows from inside the test. Worktrees without .env never noticed.
    Alpaca's transport is mocked to *succeed* here so that the only thing
    keeping it out of the picture is conftest's autouse _no_alpaca_market_data.
    """
    import data.fetcher as fetcher
    monkeypatch.setattr(fetcher, "_CACHE_DIR", tmp_path)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    alpaca_payload = {
        "bars": {"AAPL": [{"t": "2025-01-02T05:00:00Z",
                           "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1}]},
        "next_page_token": None,
    }
    with patch.object(fetcher.alpaca_data, "requests") as req, \
         patch("yfinance.Ticker") as mock_yf:
        req.get.return_value.json.return_value = alpaca_payload
        req.get.return_value.raise_for_status.return_value = None
        mock_yf.return_value.history.return_value = _sample_hist()
        fetcher.get_stock_price("AAPL", period="1y")
    req.get.assert_not_called()
    assert mock_yf.call_count == 1


def test_price_cache_never_points_at_the_checkout(tmp_path):
    """data/cache/ is gitignored, per-checkout, 1-hour TTL — and shared by every test.

    A test that forgot to pin _CACHE_DIR read whatever the previous run had
    fetched for real (131 files in one worktree), so results depended on what
    ran an hour earlier. conftest pins it per test, like the DB.
    """
    import data.fetcher as fetcher
    checkout_cache = Path(fetcher.__file__).resolve().parent / "cache"
    assert Path(fetcher._CACHE_DIR).resolve() != checkout_cache
    assert str(Path(fetcher._CACHE_DIR).resolve()).startswith(str(tmp_path.resolve()))
