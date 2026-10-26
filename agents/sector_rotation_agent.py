"""
Sector Rotation Overlay
=======================
Computes momentum for 11 SPDR sector ETFs, ranks them, and returns a
per-ticker size multiplier based on sector momentum rank, plus a helper
to cap any single sector's portfolio weight.

Momentum method: same weighted composite as individual stocks
  config.MOM_LOOKBACK_3M_WEIGHT  x 3M
+ config.MOM_LOOKBACK_6M_WEIGHT  x 6M
+ config.MOM_LOOKBACK_12M_WEIGHT x 12M   (with config.MOMENTUM_SKIP-day skip)

Sector tilt multipliers (11 sectors -> ranks 1-3 / 4-7 / 8-11):
  Top tercile    -> config.SECTOR_TOP_MULT (1.25)
  Middle tercile -> config.SECTOR_MID_MULT (1.00)
  Bottom tercile -> config.SECTOR_BOT_MULT (0.60)

Usage:
  from agents.sector_rotation_agent import get_sector_multipliers
  multipliers = get_sector_multipliers(["AAPL", "JPM", "LLY"])
  # -> {"AAPL": 1.25, "JPM": 1.0, "LLY": 0.60}
"""
import logging

from data.fetcher import get_closes, get_stock_info

import config

logger = logging.getLogger(__name__)

# SPDR sector ETF -> GICS sector name
SECTOR_ETFS: dict[str, str] = {
    "XLK":  "Technology",
    "XLF":  "Financial Services",
    "XLV":  "Healthcare",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLY":  "Consumer Cyclical",
    "XLP":  "Consumer Defensive",
    "XLU":  "Utilities",
    "XLB":  "Basic Materials",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
}

# yfinance sector string -> ETF ticker
SECTOR_TO_ETF: dict[str, str] = {v: k for k, v in SECTOR_ETFS.items()}

# Fallback map covering the full config.TICKERS universe (avoids slow per-ticker
# yf.Ticker().info lookups on the live path).
TICKER_SECTOR_CACHE: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "AMD": "Technology", "ORCL": "Technology",
    "ADBE": "Technology", "QCOM": "Technology", "TXN": "Technology",
    "CSCO": "Technology", "IBM": "Technology", "INTC": "Technology",
    "NOW": "Technology", "AMAT": "Technology", "MU": "Technology",
    # Communication Services
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services", "CMCSA": "Communication Services",
    # Consumer Cyclical
    "AMZN": "Consumer Cyclical", "TSLA": "Consumer Cyclical", "HD": "Consumer Cyclical",
    "LOW": "Consumer Cyclical", "NKE": "Consumer Cyclical", "SBUX": "Consumer Cyclical",
    "MCD": "Consumer Cyclical", "TGT": "Consumer Cyclical",
    # Consumer Defensive
    "PG": "Consumer Defensive", "COST": "Consumer Defensive",
    "WMT": "Consumer Defensive", "KO": "Consumer Defensive",
    "PEP": "Consumer Defensive", "PM": "Consumer Defensive",
    # Financial Services
    "JPM": "Financial Services", "BAC": "Financial Services", "GS": "Financial Services",
    "BRK-B": "Financial Services", "V": "Financial Services", "MA": "Financial Services",
    "WFC": "Financial Services", "C": "Financial Services", "MS": "Financial Services",
    "AXP": "Financial Services", "SPGI": "Financial Services", "BLK": "Financial Services",
    # Healthcare
    "JNJ": "Healthcare", "LLY": "Healthcare", "MRK": "Healthcare", "UNH": "Healthcare",
    "ABBV": "Healthcare", "TMO": "Healthcare", "AMGN": "Healthcare",
    "ABT": "Healthcare", "BMY": "Healthcare", "GILD": "Healthcare",
    "MDT": "Healthcare", "CVS": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "SLB": "Energy",
    "EOG": "Energy", "OXY": "Energy", "COP": "Energy",
    # Industrials
    "CAT": "Industrials", "HON": "Industrials", "GE": "Industrials",
    "RTX": "Industrials", "DE": "Industrials", "MMM": "Industrials",
    "UPS": "Industrials", "BA": "Industrials", "LMT": "Industrials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    # Real Estate
    "AMT": "Real Estate", "EQIX": "Real Estate", "PLD": "Real Estate",
    # Basic Materials
    "LIN": "Basic Materials", "APD": "Basic Materials", "NEM": "Basic Materials",
    "FCX": "Basic Materials",
    # Added with the 120-name universe (2026-08-31).
    # Technology
    "CRM": "Technology", "ACN": "Technology", "INTU": "Technology", "PANW": "Technology", "ADI": "Technology",
    # Communication Services
    "TMUS": "Communication Services",
    # Consumer Cyclical
    "BKNG": "Consumer Cyclical", "TJX": "Consumer Cyclical", "ORLY": "Consumer Cyclical", "MAR": "Consumer Cyclical",
    # Consumer Defensive
    "MO": "Consumer Defensive", "MDLZ": "Consumer Defensive", "CL": "Consumer Defensive", "KDP": "Consumer Defensive",
    # Financial Services
    "SCHW": "Financial Services", "PGR": "Financial Services", "CB": "Financial Services",
    # Healthcare
    "DHR": "Healthcare", "PFE": "Healthcare", "ISRG": "Healthcare", "SYK": "Healthcare",
    # Energy
    "PSX": "Energy", "MPC": "Energy",
    # Industrials
    "UNP": "Industrials", "ETN": "Industrials", "ADP": "Industrials",
    # Utilities
    "AEP": "Utilities", "SRE": "Utilities",
    # Real Estate
    "SPG": "Real Estate", "PSA": "Real Estate", "O": "Real Estate",
    # Basic Materials
    "SHW": "Basic Materials", "ECL": "Basic Materials", "NUE": "Basic Materials",
}


def _get_ticker_sector(ticker: str) -> str:
    """Return GICS sector string for a ticker. Cache first, then yfinance."""
    if ticker in TICKER_SECTOR_CACHE:
        return TICKER_SECTOR_CACHE[ticker]
    try:
        sector = get_stock_info(ticker).get("sector", "Unknown")
        TICKER_SECTOR_CACHE[ticker] = sector
        return sector
    except Exception:
        return "Unknown"


def _compute_sector_momentum() -> dict[str, float]:
    """
    Download 14 months of daily prices for all 11 sector ETFs and compute the
    weighted momentum composite (config weights, config.MOMENTUM_SKIP-day skip).
    Returns dict ETF -> raw momentum score (not yet ranked).
    """
    etfs = list(SECTOR_ETFS.keys())
    try:
        prices = get_closes(etfs, period="2y")
        if prices.empty:
            raise ValueError("no sector ETF closes returned")
    except Exception as e:
        logger.error(f"[SectorRotation] Download failed: {e}")
        return {etf: 0.0 for etf in etfs}

    scores: dict[str, float] = {}
    skip = config.MOMENTUM_SKIP  # avoid 1-month reversal

    for etf in etfs:
        if etf not in prices.columns:
            scores[etf] = 0.0
            continue
        try:
            p = prices[etf].dropna()
            if len(p) < 252 + skip:
                scores[etf] = 0.0
                continue

            ref = p.iloc[-(skip + 1)]  # price `skip` days ago

            def _ret(days: int) -> float:
                idx = -(days + skip)
                if abs(idx) > len(p):
                    return 0.0
                base = p.iloc[idx]
                return float(ref / base - 1) if base > 0 else 0.0

            mom = (
                config.MOM_LOOKBACK_3M_WEIGHT  * _ret(63)
                + config.MOM_LOOKBACK_6M_WEIGHT  * _ret(126)
                + config.MOM_LOOKBACK_12M_WEIGHT * _ret(252)
            )
            scores[etf] = round(mom, 6)
        except Exception:
            scores[etf] = 0.0

    return scores


def _rank_sectors(scores: dict[str, float]) -> dict[str, int]:
    """Rank ETFs 1 (best) to N (worst) by momentum score."""
    sorted_etfs = sorted(scores, key=lambda e: scores[e], reverse=True)
    return {etf: rank + 1 for rank, etf in enumerate(sorted_etfs)}


def _rank_to_multiplier(rank: int, n_sectors: int) -> float:
    tercile = n_sectors / 3
    if rank <= tercile:
        return config.SECTOR_TOP_MULT
    elif rank <= 2 * tercile:
        return config.SECTOR_MID_MULT
    else:
        return config.SECTOR_BOT_MULT


def get_sector_multipliers(tickers: list[str]) -> dict[str, float]:
    """
    Main entry point. Returns per-ticker size multiplier based on sector
    momentum rank. Tickers whose sector can't be mapped get a neutral 1.0.
    """
    logger.info("[SectorRotation] Computing sector momentum...")
    scores = _compute_sector_momentum()
    ranks = _rank_sectors(scores)
    n = len(ranks)

    logger.info("[SectorRotation] Sector ranks:")
    for etf, rank in sorted(ranks.items(), key=lambda x: x[1]):
        logger.info(f"  {rank:2d}. {etf} ({SECTOR_ETFS[etf]}): score={scores[etf]:+.4f}")

    result: dict[str, float] = {}
    for ticker in tickers:
        sector = _get_ticker_sector(ticker)
        etf = SECTOR_TO_ETF.get(sector)
        if etf and etf in ranks:
            rank = ranks[etf]
            mult = _rank_to_multiplier(rank, n)
            logger.info(f"  {ticker} -> {sector} (rank {rank}/{n}) -> multiplier {mult:.2f}x")
        else:
            mult = 1.0
            logger.warning(f"  {ticker} -> sector '{sector}' not mapped, using 1.0x")
        result[ticker] = mult

    return result


def apply_sector_cap(
    weights: dict[str, float],
    tickers: list[str],
    max_sector_pct: float = config.MAX_SECTOR_WEIGHT,
) -> dict[str, float]:
    """
    Cap each sector's total weight at max_sector_pct. If a sector exceeds the
    cap, scale down its members proportionally. This is intended to be the
    FINAL sizing step (do not renormalize afterwards, or the cap is re-violated).
    """
    sector_map: dict[str, list[str]] = {}
    for ticker in tickers:
        sector = _get_ticker_sector(ticker)
        sector_map.setdefault(sector, []).append(ticker)

    adjusted = dict(weights)

    for sector, members in sector_map.items():
        sector_weight = sum(adjusted.get(t, 0.0) for t in members)
        if sector_weight > max_sector_pct:
            scale = max_sector_pct / sector_weight
            logger.info(
                f"[SectorCap] {sector} at {sector_weight:.1%} > cap {max_sector_pct:.0%} "
                f"-> scaling by {scale:.3f}"
            )
            for t in members:
                adjusted[t] = adjusted.get(t, 0.0) * scale

    return adjusted
