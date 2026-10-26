"""Alpaca Market Data v2 — prices for the live path.

Why this module exists: yfinance was the only price source, and it stopped
holding up at the universe size this system runs. The 2026-08-06 run log records
`20 Failed downloads: [...] possibly delisted` on a 50-ticker batch and
`curl: (28) Operation timed out after 304760 ms` on DUK, NEE, EQIX, PLD, GE, SO,
LIN and AMT — five minutes per ticker, on a pipeline that must finish inside the
opening window. Widening the universe on that foundation multiplies the problem.

Alpaca credentials already exist for the broker side, the market-data tier comes
with the account, and one request returns bars for every symbol at once. This
module speaks only HTTP; `data/fetcher.py` stays the facade every caller uses,
so nothing downstream changes shape.

Feed: the free tier serves `iex`. Set ALPACA_DATA_FEED=sip if the account has
the paid full-market subscription.
"""

import logging
import os
import re
from datetime import datetime, timedelta, UTC

import pandas as pd
import requests

from utils.env import load_project_env
from utils.retry import with_retry
from data import synthetic

load_project_env()

logger = logging.getLogger(__name__)

DATA_BASE_URL = "https://data.alpaca.markets"
_TIMEOUT = 30
_MAX_BARS_PER_PAGE = 10000

# Alpaca is a US equities feed: indices (^VIX, ^GSPC) are not on it, so those
# callers stay on yfinance. Everything with a real ticker symbol comes here.
_UNSUPPORTED_PREFIXES = ("^",)


def _headers() -> dict:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    }


def credentials_present() -> bool:
    return bool(os.getenv("ALPACA_API_KEY")) and bool(os.getenv("ALPACA_SECRET_KEY"))


def supports(symbol: str) -> bool:
    """False for symbols Alpaca's equity feed cannot serve (indices)."""
    return bool(symbol) and not symbol.startswith(_UNSUPPORTED_PREFIXES)


def _feed() -> str:
    return os.getenv("ALPACA_DATA_FEED", "iex")


def to_alpaca_symbol(ticker: str) -> str:
    """BRK-B → BRK.B. Alpaca writes share classes with a dot, the repo with a dash."""
    return ticker.replace("-", ".")


def from_alpaca_symbol(symbol: str) -> str:
    """BRK.B → BRK-B. Inverse of to_alpaca_symbol, so cache keys stay repo-shaped."""
    return symbol.replace(".", "-")


_PERIOD_DAYS = {
    "1d": 1, "5d": 5, "1mo": 31, "3mo": 93, "6mo": 186,
    "1y": 366, "2y": 731, "3y": 1096, "5y": 1826, "10y": 3653,
}


_PERIOD_RE = re.compile(r"^(\d+)(d|wk|mo|y)$")
_UNIT_DAYS = {"d": 1, "wk": 7, "mo": 31, "y": 366}


def period_to_start(period: str) -> datetime:
    """yfinance-style period string → UTC start datetime.

    Calendar days, not trading days: a 252-bar momentum window needs ~366
    calendar days, which is what "1y" hands back.

    Arbitrary counts are parsed rather than defaulted. Callers do use them —
    strategy_agent asks for f"{MACRO_LOOKBACK + 10}d" — and quietly turning
    "262d" into a year would hand back more history than the caller asked for,
    hiding the case where the feed genuinely has less.
    """
    period = (period or "").lower()
    days = _PERIOD_DAYS.get(period)
    if days is None:
        m = _PERIOD_RE.match(period)
        days = int(m.group(1)) * _UNIT_DAYS[m.group(2)] if m else 366
    return datetime.now(UTC) - timedelta(days=days)


def _bars_to_frame(bars: list) -> pd.DataFrame:
    """Alpaca bar dicts → the OHLCV frame shape yfinance callers already expect."""
    if not bars:
        return pd.DataFrame()
    frame = pd.DataFrame(bars)
    frame["t"] = pd.to_datetime(frame["t"], format="ISO8601", utc=True)
    frame = frame.set_index("t").sort_index()
    # A daily bar is stamped at midnight Eastern — 04:00Z in summer, 05:00Z in
    # winter — not at midnight UTC. Reducing it to the UTC calendar date gives
    # the trading date under either offset, and is what lets a frame from here
    # line up with a yfinance frame for the same symbol.
    frame.index = frame.index.tz_convert("UTC").tz_localize(None).normalize()
    frame = frame.rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume",
    })
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in frame.columns]
    return frame[keep]


def get_bars(symbols: list[str], period: str = "1y",
             timeframe: str = "1Day") -> dict[str, pd.DataFrame]:
    """Daily OHLCV for many symbols in as few requests as possible.

    Returns {repo_ticker: DataFrame}. Symbols the feed did not return are simply
    absent from the mapping — the caller decides whether that is a fallback or a
    failure, because "missing" and "empty" mean different things upstream.
    Raises on transport failure so a caller can fall back deliberately rather
    than mistake an outage for an empty market.
    """
    if synthetic.enabled():
        return {s_: synthetic.ohlcv(s_, period) for s_ in symbols}
    wanted = [s for s in symbols if supports(s)]
    if not wanted or not credentials_present():
        return {}

    start = period_to_start(period).strftime("%Y-%m-%dT%H:%M:%SZ")
    collected: dict[str, list] = {}
    page_token = None

    # Alpaca caps the URL, not just the response, so symbols go up in chunks.
    for chunk_start in range(0, len(wanted), 100):
        chunk = wanted[chunk_start:chunk_start + 100]
        page_token = None
        while True:
            params = {
                "symbols": ",".join(to_alpaca_symbol(s) for s in chunk),
                "timeframe": timeframe,
                "start": start,
                "limit": _MAX_BARS_PER_PAGE,
                "adjustment": "all",
                "feed": _feed(),
            }
            if page_token:
                params["page_token"] = page_token

            def _do(params=params):
                resp = requests.get(f"{DATA_BASE_URL}/v2/stocks/bars",
                                    headers=_headers(), params=params, timeout=_TIMEOUT)
                resp.raise_for_status()
                return resp.json()

            payload = with_retry(_do, f"alpaca bars({len(chunk)} symbols, {period})")
            for sym, bars in (payload.get("bars") or {}).items():
                collected.setdefault(from_alpaca_symbol(sym), []).extend(bars)
            page_token = payload.get("next_page_token")
            if not page_token:
                break

    frames = {}
    for ticker, bars in collected.items():
        frame = _bars_to_frame(bars)
        if not frame.empty:
            frames[ticker] = frame
    logger.info("[AlpacaData] bars: %d/%d symbols returned (%s, feed=%s)",
                len(frames), len(wanted), period, _feed())
    return frames


def get_latest_price(symbol: str) -> "float | None":
    """Last trade price. None when unavailable — callers must not guess a price."""
    if synthetic.enabled():
        return synthetic.latest_price(symbol)
    if not supports(symbol) or not credentials_present():
        return None
    try:
        def _do():
            resp = requests.get(
                f"{DATA_BASE_URL}/v2/stocks/trades/latest",
                headers=_headers(),
                params={"symbols": to_alpaca_symbol(symbol), "feed": _feed()},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()

        payload = with_retry(_do, f"alpaca latest trade({symbol})")
        trade = (payload.get("trades") or {}).get(to_alpaca_symbol(symbol)) or {}
        price = trade.get("p")
        return float(price) if price else None
    except Exception as exc:
        logger.warning("[AlpacaData] latest price for %s failed: %s", symbol, exc)
        return None
