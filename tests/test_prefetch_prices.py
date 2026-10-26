# tests/test_prefetch_prices.py — batched price prefetch (auditoría Fase B, prompt 15)
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.fetcher as fetcher


def _fake_batch(tickers, n=30):
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
    rng = np.random.default_rng(3)
    data = rng.uniform(90, 110, size=(n, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_prefetch_populates_cache_one_request(tmp_path):
    tickers = ["AAPL", "MSFT", "NVDA"]
    with (
        patch.object(fetcher, "_CACHE_DIR", Path(tmp_path)),
        patch.object(fetcher.yf, "download", return_value=_fake_batch(tickers)) as mock_dl,
    ):
        cached = fetcher.prefetch_prices(tickers)
        assert cached == 3
        assert mock_dl.call_count == 1  # ONE network request for all tickers

        # get_stock_price must now be served from cache — no yf.Ticker call
        with patch.object(fetcher.yf, "Ticker") as mock_ticker:
            df = fetcher.get_stock_price("AAPL")
        assert df is not None and not df.empty
        assert "Close" in df.columns
        mock_ticker.assert_not_called()


def test_prefetch_failure_returns_zero_and_does_not_raise(tmp_path):
    with (
        patch.object(fetcher, "_CACHE_DIR", Path(tmp_path)),
        patch.object(fetcher.yf, "download", side_effect=ConnectionError("down")),
        patch("utils.retry.time.sleep"),
    ):
        assert fetcher.prefetch_prices(["AAPL"]) == 0  # caller falls back per-ticker


def test_prefetch_empty_universe():
    assert fetcher.prefetch_prices([]) == 0


# ── Mixed Alpaca + yfinance coverage ────────────────────────────────────
# The path production actually takes, and the one the suite never exercised:
# Alpaca serves most of the universe and yfinance picks up the rest. With no
# Alpaca credentials every earlier test took the yfinance-only branch, which is
# why a defect on the mixed path reached a green CI.

def _alpaca_stamps(days):
    """Alpaca stamps a daily bar at midnight Eastern: 04:00Z in EDT, 05:00Z in EST."""
    return pd.to_datetime([
        pd.Timestamp(f"{d:%Y-%m-%d} 00:00:00", tz="America/New_York").tz_convert("UTC")
        for d in days
    ])


def _alpaca_frame(n=30):
    """Alpaca daily bars as they arrive: UTC index, not at midnight UTC."""
    idx = _alpaca_stamps(pd.bdate_range(end=pd.Timestamp.today(), periods=n))
    return pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000.0},
        index=idx,
    )


def test_partial_alpaca_coverage_counts_both_sources(tmp_path):
    """The return value feeds run_signals' data-failure guard.

    Counting only the yfinance leg reports a ~98% failure rate on a healthy run
    and aborts the whole cycle before research.
    """
    tickers = ["AAPL", "MSFT", "NVDA"]
    alpaca_side = {"AAPL": _alpaca_frame(), "MSFT": _alpaca_frame()}

    with (
        patch.object(fetcher, "_CACHE_DIR", Path(tmp_path)),
        patch.object(fetcher.alpaca_data, "get_bars", return_value=alpaca_side),
        patch.object(fetcher.yf, "download", return_value=_fake_batch(["NVDA"])),
    ):
        cached = fetcher.prefetch_prices(tickers)

    assert cached == 3, f"expected 2 from Alpaca + 1 from yfinance, got {cached}"


def test_full_alpaca_coverage_skips_yfinance(tmp_path):
    tickers = ["AAPL", "MSFT"]
    with (
        patch.object(fetcher, "_CACHE_DIR", Path(tmp_path)),
        patch.object(fetcher.alpaca_data, "get_bars",
                     return_value={t: _alpaca_frame() for t in tickers}),
        patch.object(fetcher.yf, "download") as mock_dl,
    ):
        assert fetcher.prefetch_prices(tickers) == 2
        mock_dl.assert_not_called()


def test_get_closes_aligns_frames_from_different_sources(tmp_path):
    """Alpaca stamps bars at 04:00Z tz-naive; yfinance at midnight tz-aware.

    Combining them raised `Cannot join tz-naive with tz-aware DatetimeIndex`.
    Every caller wraps get_closes in its own try/except, so the crash surfaced
    as a plausible-looking result — for _credit_stress_signal, which fails
    CLOSED, it meant suppressing every BUY in the universe.
    """
    n = 30
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    alpaca = pd.DataFrame({"Close": np.linspace(100, 110, n)}, index=_alpaca_stamps(days))
    yfin = pd.DataFrame(
        {"Close": np.linspace(50, 55, n)},
        index=days.tz_localize("America/New_York"),
    )

    def _by_source(ticker, period="1y"):
        return alpaca if ticker == "HYG" else yfin

    with (
        patch.object(fetcher, "_CACHE_DIR", Path(tmp_path)),
        patch.object(fetcher, "prefetch_prices", return_value=2),
        patch.object(fetcher, "get_stock_price", side_effect=_by_source),
    ):
        closes = fetcher.get_closes(["HYG", "LQD"], period="1y")

    assert list(closes.columns) == ["HYG", "LQD"]
    assert len(closes.dropna()) == n, (
        "mixed-source frames must share every trading day, not none of them"
    )
    assert closes.index.tz is None


def test_sources_agree_on_the_trading_date_across_dst(tmp_path):
    """A year of history crosses a DST change, and the two sources shift with it.

    Alpaca stamps midnight Eastern (04:00Z in summer, 05:00Z in winter);
    yfinance returns tz-aware midnight Eastern. Normalising on the Eastern
    clock instead of the UTC date pushed winter bars back a day, so summer rows
    aligned and winter rows did not — a partial misalignment is harder to spot
    than a total one.
    """
    days = pd.bdate_range("2026-01-02", "2026-07-01")
    alpaca = pd.DataFrame({"Close": np.linspace(100, 110, len(days))},
                          index=_alpaca_stamps(days))
    yfin = pd.DataFrame({"Close": np.linspace(50, 55, len(days))},
                        index=days.tz_localize("America/New_York"))

    with (
        patch.object(fetcher, "_CACHE_DIR", Path(tmp_path)),
        patch.object(fetcher, "prefetch_prices", return_value=2),
        patch.object(fetcher, "get_stock_price",
                     side_effect=lambda t, period="1y": alpaca if t == "A" else yfin),
    ):
        closes = fetcher.get_closes(["A", "B"], period="1y")

    assert len(closes) == len(days)
    assert len(closes.dropna()) == len(days), "every trading day must align, winter included"
