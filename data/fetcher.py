import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TypeVar
from collections.abc import Callable

import numpy as np
import pandas as pd
import yfinance as yf

from data import synthetic

from data import alpaca_data

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_YF_MAX_RETRIES = 3
_YF_RETRY_DELAYS = (2, 5)  # seconds between attempt 1→2 and 2→3

from utils.retry import with_retry as _shared_with_retry

# Every ticker whose price fetch produced nothing this process, and the symbols
# that had to fall back to yfinance. main.py reads these to decide whether the
# run has enough data to be worth trusting: a universe that half-failed used to
# produce a book of conviction-1 HOLDs and look like a normal quiet day.
FETCH_FAILURES: set[str] = set()
FETCH_FALLBACKS: set[str] = set()


def reset_fetch_stats() -> None:
    FETCH_FAILURES.clear()
    FETCH_FALLBACKS.clear()


def _with_retry(fn: Callable[[], _T], label: str) -> _T:
    """Thin wrapper over utils.retry.with_retry (historical fetcher behavior: retry any Exception)."""
    return _shared_with_retry(fn, label, max_retries=_YF_MAX_RETRIES, delays=_YF_RETRY_DELAYS)

_CACHE_DIR = Path(os.path.join(os.path.dirname(__file__), "cache"))
_PRICE_TTL = 3600    # 1 hour
_FUNDM_TTL = 21600   # 6 hours


def _normalize_index(df: "pd.DataFrame | None") -> "pd.DataFrame | None":
    """Give every price frame the same index shape: tz-naive, at midnight.

    The two sources disagree in both respects. Alpaca stamps a daily bar at the
    session open (04:00Z) and this module strips the zone; yfinance returns
    midnight in US/Eastern, tz-aware. Pandas refuses to join the two —
    `Cannot join tz-naive with tz-aware DatetimeIndex` — and since every caller
    wraps its fetch in a try/except, that crash surfaced as a plausible result
    rather than an error: _credit_stress_signal fails CLOSED, so it suppressed
    every BUY in the universe.

    Normalising here, at the single door frames come through, means no consumer
    has to know which source served which symbol.
    """
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    idx = df.index
    if idx.tz is not None:
        # UTC, not US/Eastern. Both sources stamp a daily bar somewhere inside
        # the US session, so the UTC calendar date is the trading date for
        # either convention — Alpaca's midnight-Eastern (04:00Z / 05:00Z) and
        # yfinance's tz-aware midnight-Eastern alike. Converting to Eastern
        # first would push a bar stamped at plain 00:00Z back a day.
        idx = idx.tz_convert("UTC").tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    return df[~df.index.duplicated(keep="last")]


def _cache_key(fn_name: str, ticker: str, **kwargs) -> Path:
    tag = hashlib.md5(f"{fn_name}{ticker}{sorted(kwargs.items())}".encode()).hexdigest()[:8]
    return _CACHE_DIR / f"{fn_name}_{ticker}_{tag}.json"


def _cache_get(path: Path, ttl: int):
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text())
    except Exception:
        return None
    if time.time() - meta.get("ts", 0) > ttl:
        return None
    if meta.get("is_df"):
        import io
        return pd.read_json(io.StringIO(meta["data"]))
    return meta["data"]


def _cache_set(path: Path, data, is_df: bool = False) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = data.to_json() if is_df else data
    path.write_text(json.dumps({"ts": time.time(), "data": payload, "is_df": is_df}))


def prefetch_prices(tickers: list[str], period: str = "1y") -> int:
    """Warm the per-ticker price cache with one batched request.

    Sequential get_stock_price() misses cost one network round-trip each
    (~20 tickers ≈ 20 requests; 10x universe would be 200). This fetches all
    tickers at once and writes each slice into the same cache slot
    get_stock_price(ticker, period=...) reads, so downstream code is unchanged.

    Alpaca serves the batch; whatever it does not return falls back to
    yf.download. Note that neither batch frame carries the Dividends/Stock
    Splits columns Ticker().history() adds; consumers here only use OHLCV.
    Returns the number of tickers cached (0 on failure — callers fall back to
    the per-ticker path).
    """
    if not tickers:
        return 0
    if synthetic.enabled():
        return len(tickers)  # get_stock_price generates on demand

    # Alpaca first: one request for the whole universe, no per-ticker timeouts.
    # from_alpaca counts what it served; the yfinance leg below adds to it. The
    # total is what run_signals checks against MAX_DATA_FAILURE_PCT, so counting
    # only the fallback leg reports a ~98% failure on a healthy run and aborts
    # the cycle before research.
    from_alpaca = 0
    try:
        frames = alpaca_data.get_bars(tickers, period=period)
        for t, df in frames.items():
            _cache_set(_cache_key("get_stock_price", t, period=period),
                       _normalize_index(df), is_df=True)
        if frames:
            from_alpaca = len(frames)
            missing = [t for t in tickers if t not in frames]
            logger.info("[Fetcher] prefetch_prices: %d/%d tickers from Alpaca%s",
                        from_alpaca, len(tickers),
                        f" ({len(missing)} missing → yfinance)" if missing else "")
            if not missing:
                return from_alpaca
            tickers = missing
            FETCH_FALLBACKS.update(missing)
    except Exception as exc:
        logger.warning("[Fetcher] Alpaca prefetch failed (%s) — falling back to yfinance", exc)
        FETCH_FALLBACKS.update(tickers)

    try:
        def _fetch() -> pd.DataFrame:
            return yf.download(
                tickers=" ".join(tickers), period=period,
                group_by="ticker", auto_adjust=True, progress=False, threads=True,
            )

        raw = _with_retry(_fetch, f"prefetch_prices({len(tickers)} tickers, {period})")
        if raw is None or raw.empty:
            return from_alpaca
        cached = 0
        for t in tickers:
            try:
                df = raw[t].dropna(how="all") if isinstance(raw.columns, pd.MultiIndex) else raw
                if df is None or df.empty:
                    continue
                _cache_set(_cache_key("get_stock_price", t, period=period),
                           _normalize_index(df), is_df=True)
                cached += 1
            except Exception:
                continue  # this ticker falls back to the per-ticker path
        logger.info("[Fetcher] prefetch_prices: %d/%d tickers cached in one request",
                    cached, len(tickers))
        return from_alpaca + cached
    except Exception as exc:
        logger.warning("[Fetcher] prefetch_prices failed (%s) — falling back to per-ticker fetches", exc)
        return from_alpaca


def get_stock_price(ticker: str, period: str = "1y") -> "pd.DataFrame | None":
    if synthetic.enabled():
        return synthetic.ohlcv(ticker, period)
    cache_path = _cache_key("get_stock_price", ticker, period=period)
    cached = _cache_get(cache_path, _PRICE_TTL)
    if cached is not None:
        # Normalised on read as well as on write: entries cached before this
        # existed still carry whichever index their source produced.
        return _normalize_index(cached)

    if alpaca_data.supports(ticker):
        try:
            frames = alpaca_data.get_bars([ticker], period=period)
            hist = _normalize_index(frames.get(ticker))
            if hist is not None and not hist.empty:
                _cache_set(cache_path, hist, is_df=True)
                return hist
        except Exception as exc:
            logger.warning("[Fetcher] Alpaca price for %s failed (%s) — trying yfinance",
                           ticker, exc)
        FETCH_FALLBACKS.add(ticker)

    try:
        def _fetch() -> pd.DataFrame:
            return yf.Ticker(ticker).history(period=period)

        hist = _normalize_index(_with_retry(_fetch, f"get_stock_price({ticker}, {period})"))
        if hist is None or hist.empty:
            logger.warning("[Fetcher] Empty price data for %s (period=%s)", ticker, period)
            FETCH_FAILURES.add(ticker)
            return None
        _cache_set(cache_path, hist, is_df=True)
        return hist
    except Exception as exc:
        logger.error("[Fetcher] get_stock_price failed for %s: %s: %s",
                     ticker, type(exc).__name__, exc)
        FETCH_FAILURES.add(ticker)
        return None


def get_stock_info(ticker: str) -> dict:
    _defaults = {
        "ticker": ticker, "name": ticker, "sector": "Unknown",
        "pe_ratio": None, "forward_pe": None, "roe": None, "revenue_growth": None,
        "market_cap": None, "free_cashflow": None, "price": None, "book_value": None,
    }
    if synthetic.enabled():
        return {**_defaults, **synthetic.info(ticker)}
    cache_path = _cache_key("get_stock_info", ticker)
    cached = _cache_get(cache_path, _FUNDM_TTL)
    if cached is not None:
        # Merged over the defaults, not returned raw: entries written before a
        # new key existed would otherwise come back missing it, and
        # _compute_bull_signals drops a candidate outright on a falsy `price`.
        # With a 6h TTL that silently empties the book after every deploy.
        return {**_defaults, **cached}
    try:
        stock = yf.Ticker(ticker)
        info = _with_retry(lambda: stock.info, f"get_stock_info({ticker})")

        def safe_get(key, default=None):
            val = info.get(key, default)
            return val if val is not None else default

        result = {
            "ticker": ticker,
            "name": safe_get("longName", ticker),
            "sector": safe_get("sector", "Unknown"),
            "pe_ratio": safe_get("trailingPE"),
            "forward_pe": safe_get("forwardPE"),
            "roe": safe_get("returnOnEquity"),
            "revenue_growth": safe_get("revenueGrowth"),
            "market_cap": safe_get("marketCap"),
            "free_cashflow": safe_get("freeCashflow"),
            # regime_strategy_agent screens on these two; without them its
            # BULL branch drops every candidate on the first `if not price`.
            "price": (safe_get("currentPrice")
                      or safe_get("regularMarketPrice")
                      or safe_get("previousClose")),
            "book_value": safe_get("bookValue"),
        }
        _cache_set(cache_path, result, is_df=False)
        return result
    except Exception as exc:
        logger.error("[Fetcher] get_stock_info failed for %s: %s: %s",
                     ticker, type(exc).__name__, exc)
        return _defaults


def atr_from_ohlc(high, low, close, period: int = 14):
    """Wilder-style ATR as an EMA (span=period, adjust=False) of the true range.

    The one ATR formula shared by live and backtest: get_atr (live stops and
    levels) and backtest/reference_backtest.py (simulated stops) both call this, so
    the stop distance the backtest validates is the one the broker will get.
    Accepts Series or wide DataFrames (one column per ticker, computed
    column-wise). The true range treats a missing previous close the way the
    original get_atr did — it falls back to high - low on that bar.
    """
    prev_close = close.shift(1)
    tr = np.fmax(np.fmax(high - low, (high - prev_close).abs()), (low - prev_close).abs())
    return tr.ewm(span=period, adjust=False).mean()


def get_atr(ticker: str, period: int = 14, lookback: str = "3mo") -> float:
    """Wilder ATR via EMA. Returns latest ATR in price units, 0.0 on error."""
    try:
        hist = get_stock_price(ticker, period=lookback)
        if hist is None or hist.empty or len(hist) < period + 1:
            return 0.0
        atr = atr_from_ohlc(hist["High"], hist["Low"], hist["Close"], period=period)
        return float(atr.iloc[-1])
    except Exception:
        return 0.0


def compute_beta_60d(ticker: str) -> float:
    """OLS beta of ticker vs SPY over ~60 trading days (3 calendar months). Returns 1.0 on error."""
    try:
        closes = {}
        for sym in (ticker, "SPY"):
            hist = get_stock_price(sym, period="3mo")
            if hist is None or hist.empty:
                return 1.0
            closes[sym] = hist["Close"]
        data = pd.DataFrame(closes).dropna()
        returns = data.pct_change().dropna()
        if len(returns) < 30 or ticker not in returns.columns or "SPY" not in returns.columns:
            return 1.0
        cov_matrix = returns.cov()
        spy_var = cov_matrix.loc["SPY", "SPY"]
        if spy_var == 0:
            return 1.0
        return float(cov_matrix.loc[ticker, "SPY"] / spy_var)
    except Exception:
        return 1.0


def prefetch_infos(tickers: list[str], max_workers: int = 8) -> dict[str, dict]:
    """Fundamentals for many tickers, fetched in parallel and cached 6h.

    get_stock_info is one blocking yfinance `.info` call. strategy_agent used to
    make two serial passes over the universe (earnings yield, then FCF yield),
    so 50 tickers meant 100 sequential round-trips on the critical path and 150
    tickers would mean 300. Same call, same cache, eight at a time.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        infos = list(pool.map(get_stock_info, tickers))
    return {info["ticker"]: info for info in infos}


def get_closes(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Wide frame of adjusted closes: one column per ticker, indexed by date.

    The shape `yf.download(...)["Close"]` used to return, but sourced through
    this facade so every consumer inherits the Alpaca-first path, the shared
    cache and the yfinance fallback. Tickers with no data are simply missing
    columns — callers already handle that, and it keeps one bad symbol from
    dragging the whole frame down through dropna().
    """
    if not tickers:
        return pd.DataFrame()
    prefetch_prices(tickers, period=period)
    series = {}
    for t in tickers:
        hist = _normalize_index(get_stock_price(t, period=period))
        if hist is None or hist.empty or "Close" not in hist:
            continue
        col = hist["Close"].dropna()
        if not col.empty:
            series[t] = col
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()
