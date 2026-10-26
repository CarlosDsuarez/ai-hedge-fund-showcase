"""Alpaca Market Data client — the pure parts, without touching the network.

Everything here is shape and translation: the pieces that decide whether a
downstream agent sees the frame it expects or a silent empty one.
"""
import os
import sys
from datetime import datetime, UTC
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import alpaca_data as ad

BARS = [
    {"t": "2026-08-25T04:00:00Z", "o": 100.0, "h": 103.0, "l": 99.0, "c": 102.0, "v": 1_000},
    {"t": "2026-08-26T04:00:00Z", "o": 102.0, "h": 105.0, "l": 101.0, "c": 104.0, "v": 1_200},
]


def test_bars_become_the_ohlcv_frame_callers_expect():
    """get_atr, _compute_adx and the momentum window all index by these names."""
    df = ad._bars_to_frame(BARS)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert float(df["Close"].iloc[-1]) == 104.0
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is None, "tz-aware index breaks comparison with cached frames"
    assert df.index.is_monotonic_increasing


def test_empty_bars_give_an_empty_frame_not_an_error():
    assert ad._bars_to_frame([]).empty


def test_share_class_symbols_round_trip():
    """The repo writes BRK-B, Alpaca writes BRK.B. Cache keys stay repo-shaped."""
    assert ad.to_alpaca_symbol("BRK-B") == "BRK.B"
    assert ad.from_alpaca_symbol("BRK.B") == "BRK-B"
    assert ad.from_alpaca_symbol(ad.to_alpaca_symbol("BRK-B")) == "BRK-B"
    assert ad.to_alpaca_symbol("AAPL") == "AAPL"


def test_indices_are_not_claimed_by_the_equity_feed():
    """^VIX and ^GSPC must fall through to yfinance, not come back empty."""
    assert ad.supports("AAPL")
    assert not ad.supports("^VIX")
    assert not ad.supports("")


def test_period_strings_map_to_real_windows():
    now = datetime.now(UTC)
    assert (now - ad.period_to_start("1y")).days == 365
    assert (now - ad.period_to_start("2y")).days == 730
    # Arbitrary counts are parsed, not defaulted: strategy_agent asks for "262d".
    assert (now - ad.period_to_start("262d")).days == 261
    # An unparseable string still yields a usable window rather than raising.
    assert (now - ad.period_to_start("nonsense")).days == 365


def test_no_credentials_means_no_request(real_alpaca_availability):
    """A missing key must not produce a 401 storm across 120 tickers."""
    with patch.dict(os.environ, {"ALPACA_API_KEY": "", "ALPACA_SECRET_KEY": ""},
                    clear=False), patch.object(ad, "requests") as req:
        assert ad.get_bars(["AAPL", "MSFT"]) == {}
        assert ad.get_latest_price("AAPL") is None
        req.get.assert_not_called()


def test_bars_are_keyed_by_repo_ticker(real_alpaca_availability):
    payload = {"bars": {"BRK.B": BARS}, "next_page_token": None}
    with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"},
                    clear=False), patch.object(ad, "requests") as req:
        req.get.return_value.json.return_value = payload
        req.get.return_value.raise_for_status.return_value = None
        frames = ad.get_bars(["BRK-B"])
    assert set(frames) == {"BRK-B"}
    assert float(frames["BRK-B"]["Close"].iloc[-1]) == 104.0


def test_pagination_accumulates_every_page(real_alpaca_availability):
    pages = [
        {"bars": {"AAPL": BARS[:1]}, "next_page_token": "tok"},
        {"bars": {"AAPL": BARS[1:]}, "next_page_token": None},
    ]
    with patch.dict(os.environ, {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"},
                    clear=False), patch.object(ad, "requests") as req:
        req.get.return_value.raise_for_status.return_value = None
        req.get.return_value.json.side_effect = pages
        frames = ad.get_bars(["AAPL"])
    assert len(frames["AAPL"]) == 2, "a dropped page silently shortens the history"
