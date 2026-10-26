"""
data/synthetic.py — seeded synthetic market data for the offline demo.

Enabled with DATA_MODE=synthetic. Every series is a deterministic function of
the symbol (FNV-1a hash → NumPy seed), so repeated runs and different modules
see the same prices. Nothing here is real market data:

  * equities / ETFs : geometric Brownian motion, annual drift −2%…+12%,
                      annual volatility 15%…45%, business-day calendar ending today
  * ^VIX            : mean-reverting (Ornstein–Uhlenbeck) level around 18
  * fundamentals    : P/E, ROE, market cap, etc. drawn from plausible ranges
  * headlines       : placeholder strings, clearly labelled synthetic

The shapes mirror what data/fetcher.py, data/alpaca_data.py and yfinance return,
so callers do not know which source they are reading.
"""
from __future__ import annotations

import os
import re
from datetime import date

import numpy as np
import pandas as pd

_EPOCH = "2015-01-02"


def enabled() -> bool:
    return os.getenv("DATA_MODE", "").strip().lower() == "synthetic"


def _seed(text: str) -> int:
    h = 0x811C9DC5
    for ch in text.encode():
        h = ((h ^ ch) * 0x01000193) & 0xFFFFFFFF
    return h


def _calendar() -> pd.DatetimeIndex:
    return pd.bdate_range(_EPOCH, date.today())


_CACHE: dict[str, pd.DataFrame] = {}


def _full_history(symbol: str) -> pd.DataFrame:
    if symbol in _CACHE:
        return _CACHE[symbol]
    rng = np.random.default_rng(_seed(f"px:{symbol}"))
    idx = _calendar()
    n = len(idx)
    if symbol.upper() in ("^VIX", "VIX"):
        level = np.empty(n)
        level[0] = 18.0
        shocks = rng.normal(0, 1.2, n)
        for i in range(1, n):
            level[i] = max(9.0, level[i - 1] + 0.08 * (18.0 - level[i - 1]) + shocks[i])
        close = level
        open_ = close * (1 + rng.normal(0, 0.01, n))
    else:
        traits = np.random.default_rng(_seed(f"traits:{symbol}"))
        start = traits.uniform(20, 400)
        vol = traits.uniform(0.15, 0.45) / np.sqrt(252)
        drift = traits.uniform(-0.02, 0.12) / 252 - 0.5 * vol**2
        rets = drift + vol * rng.standard_normal(n)
        close = start * np.exp(np.cumsum(rets))
        open_ = np.r_[close[0], close[:-1]] * np.exp(0.25 * vol * rng.standard_normal(n))
    wiggle = np.abs(rng.normal(0, 0.006, (2, n)))
    high = np.maximum(open_, close) * (1 + wiggle[0])
    low = np.minimum(open_, close) * (1 - wiggle[1])
    volume = np.round(rng.lognormal(mean=15, sigma=0.35, size=n))
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    ).round({"Open": 4, "High": 4, "Low": 4, "Close": 4})
    df.index.name = "Date"
    _CACHE[symbol] = df
    return df


_PERIOD = re.compile(r"^(\d+)\s*(d|wk|mo|y)$")


def _period_days(period: str) -> int:
    m = _PERIOD.match(str(period).strip().lower())
    if not m:
        return 365
    n, unit = int(m.group(1)), m.group(2)
    return n * {"d": 1, "wk": 7, "mo": 31, "y": 366}[unit]


def ohlcv(symbol: str, period: str = "1y") -> pd.DataFrame:
    """yfinance-style OHLCV frame for the trailing `period` (e.g. "1y", "3mo", "262d")."""
    full = _full_history(symbol)
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=_period_days(period))
    return full[full.index >= cutoff].copy()


def ohlcv_range(symbol: str, start: str, end: str) -> pd.DataFrame:
    full = _full_history(symbol)
    return full[(full.index >= pd.Timestamp(start)) & (full.index < pd.Timestamp(end))].copy()


def download(symbols: list[str], period: str = "1y") -> pd.DataFrame:
    """Shape of `yf.download(symbols, period=…)`: MultiIndex columns (field, symbol)."""
    frames = {s: ohlcv(s, period) for s in symbols}
    return pd.concat(frames, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)


def ohlc_panel(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """{"Open","High","Low","Close"} → wide frames, for the reference backtester."""
    frames = {s: ohlcv_range(s, start, end) for s in symbols}
    return {fld: pd.DataFrame({s.replace("^VIX", "VIX"): f[fld] for s, f in frames.items()})
            for fld in ("Open", "High", "Low", "Close")}


_SECTORS = ["Technology", "Healthcare", "Financial Services", "Energy", "Industrials",
            "Consumer Defensive", "Consumer Cyclical", "Utilities", "Real Estate",
            "Basic Materials", "Communication Services"]


def info(ticker: str) -> dict:
    """Fundamentals dict with the keys data.fetcher.get_stock_info returns."""
    r = np.random.default_rng(_seed(f"info:{ticker}"))
    price = float(ohlcv(ticker, "5d")["Close"].iloc[-1])
    shares = float(r.uniform(0.5e9, 8e9))
    pe = float(r.uniform(8, 45))
    return {
        "ticker": ticker,
        "name": f"{ticker} (synthetic)",
        "sector": _SECTORS[int(r.integers(len(_SECTORS)))],
        "pe_ratio": round(pe, 2),
        "forward_pe": round(pe * float(r.uniform(0.8, 1.05)), 2),
        "roe": round(float(r.uniform(0.02, 0.40)), 4),
        "revenue_growth": round(float(r.uniform(-0.05, 0.25)), 4),
        "market_cap": round(price * shares),
        "free_cashflow": round(price * shares * float(r.uniform(0.0, 0.06))),
        "price": round(price, 2),
        "book_value": round(price / float(r.uniform(1.2, 8.0)), 2),
    }


def headlines(ticker: str, n: int = 3) -> list[str]:
    return [f"[Synthetic] {ticker}: placeholder headline {i + 1} for the offline demo"
            for i in range(n)]


def latest_price(ticker: str) -> float:
    return float(ohlcv(ticker, "5d")["Close"].iloc[-1])
