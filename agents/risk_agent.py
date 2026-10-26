import logging

import numpy as np
import pandas as pd

from data.fetcher import get_closes, get_stock_price

logger = logging.getLogger(__name__)
from config import MAX_POSITION_PCT, RISK_FREE_ANNUAL

MAX_DRAWDOWN_LIMIT = -0.12

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "LLY", "JNJ", "MA", "AVGO", "PG", "HD",
    "COST", "MRK",
]


def _avg_pairwise_correlation(tickers: list, period: str = "3mo") -> float:
    """Average pairwise correlation across a universe using recent price returns."""
    if len(tickers) < 2:
        return 0.0
    try:
        prices = get_closes(tickers, period=period)
        if prices.empty:
            return 0.0
        rets = prices.pct_change().dropna()
        if rets.shape[1] < 2:
            return 0.0
        corr = rets.corr()
        n = len(corr)
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        return float(corr.values[mask].mean())
    except Exception:
        return 0.0


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    if returns.empty:
        return 0.0
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def calculate_max_drawdown(prices: pd.Series) -> float:
    if prices.empty:
        return 0.0
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return float(drawdown.min())


def _calculate_sharpe(returns: pd.Series, risk_free_rate: float = RISK_FREE_ANNUAL) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(252))


def analyze_portfolio_risk(positions: dict, universe: list | None = None) -> dict:
    """
    positions: dict of ticker -> weight (e.g. {'AAPL': 0.04, 'MSFT': 0.04})
    """
    violations = []
    all_returns = {}

    for ticker, weight in positions.items():
        if weight > MAX_POSITION_PCT:
            violations.append(f"{ticker} exceeds max position ({weight:.1%} > {MAX_POSITION_PCT:.1%})")

        try:
            hist = get_stock_price(ticker, period="1y")
            prices = hist["Close"]
            returns = prices.pct_change().dropna()
            all_returns[ticker] = returns
        except Exception as e:
            logger.warning("[RiskAgent] Returns fetch failed for %s: %s", ticker, e)

    if not all_returns:
        return {
            "var_95": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "avg_correlation": 0.0,
            "total_exposure": sum(positions.values()),
            "violations": violations,
            "status": "WARNING" if violations else "OK",
        }

    returns_df = pd.DataFrame(all_returns).dropna()
    weights = np.array([positions.get(t, 0) for t in returns_df.columns])
    if weights.sum() > 0:
        weights = weights / weights.sum()

    portfolio_returns = returns_df.dot(weights)

    var_95 = calculate_var(portfolio_returns)
    sharpe = _calculate_sharpe(portfolio_returns)

    ticker_list = list(all_returns.keys())
    all_prices = {}
    for ticker in ticker_list:
        try:
            hist = get_stock_price(ticker, period="1y")
            all_prices[ticker] = hist["Close"]
        except Exception as e:
            logger.warning("[RiskAgent] Price fetch failed for %s: %s", ticker, e)

    combined_prices = pd.Series(dtype=float)
    if all_prices:
        prices_df = pd.DataFrame(all_prices).dropna()
        w = np.array([positions.get(t, 0) for t in prices_df.columns])
        if w.sum() > 0:
            w = w / w.sum()
        combined_prices = prices_df.dot(w)

    max_dd = calculate_max_drawdown(combined_prices) if not combined_prices.empty else 0.0

    if max_dd < MAX_DRAWDOWN_LIMIT:
        violations.append(f"Portfolio max drawdown {max_dd:.1%} exceeds limit {MAX_DRAWDOWN_LIMIT:.1%}")

    # Always compute correlation on the full universe — a single-position portfolio
    # trivially has corr=1.0 with itself, which is meaningless for risk.
    corr_universe = list(universe) if universe else UNIVERSE
    avg_corr = _avg_pairwise_correlation(corr_universe)

    status = "WARNING" if violations else "OK"

    return {
        "var_95": round(var_95, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 2),
        "avg_correlation": round(avg_corr, 4),
        "total_exposure": round(sum(positions.values()), 4),
        "violations": violations,
        "status": status,
    }


def validate_risk_reward(entry: float, sl: float, tp3: float,
                          min_rr: float = 1.5) -> tuple[bool, float]:
    """Valida R:R usando TP3 como target completo. Retorna (es_valido, rr_actual)."""
    if entry <= 0 or sl <= 0 or tp3 <= 0:
        return False, 0.0
    risk   = abs(entry - sl)
    reward = abs(tp3 - entry)
    rr     = reward / risk if risk > 0 else 0.0
    return rr >= min_rr, round(rr, 2)


def compute_position_size(capital: float, entry: float, sl: float,
                           risk_pct: float = 0.01,
                           max_pos_pct: float = MAX_POSITION_PCT) -> int:
    """
    Fixed-risk sizing: size = floor((capital * risk_pct) / sl_distance)
    Capped por max_pos_pct. Retorna 0 si inputs inválidos.
    """
    import math
    if entry <= 0 or sl <= 0 or entry <= sl:
        return 0
    sl_distance  = entry - sl
    size_by_risk = math.floor((capital * risk_pct) / sl_distance)
    size_by_pct  = math.floor((capital * max_pos_pct) / entry)
    return max(0, min(size_by_risk, size_by_pct))
