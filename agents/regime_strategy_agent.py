import json
import logging
import os
from datetime import datetime

import pandas as pd
import config
from agents.sector_rotation_agent import TICKER_SECTOR_CACHE
from data.fetcher import get_closes

logger = logging.getLogger(__name__)

LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')

# Derived from the configured universe, never hand-listed.
#
# These used to be two literal lists that had drifted away from config.TICKERS
# in both directions: they named ACN, CRM, ADBE and NVO, which the pipeline
# never analysed, while SLB, DUK, AMT, ORCL, AMD and a dozen others in the
# universe had no regime signal at all. That mattered because on a healthy
# cycle a regime BUY is on its own sufficient evidence to enter
# (_signal_gate_ok in main.py), so the drift silently decided which names could
# ever be bought on regime strength alone.
#
# BEAR keeps the defensive tilt the old list expressed, now stated as a rule:
# sectors that historically hold up in a drawdown, plus the mega-caps.
_DEFENSIVE_SECTORS = {
    'Consumer Defensive', 'Healthcare', 'Utilities', 'Communication Services',
}

BULL_TICKERS = list(config.TICKERS)
BEAR_TICKERS = [
    t for t in config.TICKERS
    if TICKER_SECTOR_CACHE.get(t) in _DEFENSIVE_SECTORS
] or list(config.TICKERS)


def _today_cache_path(regime: str) -> str:
    return os.path.join(LOGS_DIR, f'regime_signals_{datetime.today().strftime("%Y-%m-%d")}_{regime}.json')


def _find_month_cache(regime: str) -> dict | None:
    today = datetime.today()
    month_prefix = today.strftime('%Y-%m')
    for day in range(today.day - 1, 0, -1):
        path = os.path.join(LOGS_DIR, f'regime_signals_{month_prefix}-{day:02d}_{regime}.json')
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("[RegimeStrategy] Month cache load failed at %s: %s", path, e)
    return None


def _switched_today() -> bool:
    state_path = os.path.join(LOGS_DIR, 'regime_state.json')
    try:
        if os.path.exists(state_path):
            with open(state_path) as f:
                return json.load(f).get('switched_today', False)
    except Exception as e:
        logger.warning("[RegimeStrategy] Could not read regime_state.json: %s", e)
    return False


def _save_signals(signals: dict, regime: str):
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(_today_cache_path(regime), 'w') as f:
        json.dump(signals, f, indent=2)


def _fetch_closes(tickers: list, period: str) -> pd.DataFrame:
    return get_closes(tickers, period=period)


def _hold_all(tickers: list, regime: str, strategy: str) -> dict:
    return {t: {'signal': 'HOLD', 'regime': regime, 'strategy': strategy,
                'score': 0.0, 'rank': 99} for t in tickers}


# ── Regime selections (SHOWCASE: reference implementation) ──────────────────
#
# SHOWCASE: the private version runs a different, regime-specific selection
# model in BULL and BEAR. Those screens, factor definitions and cut-offs are not
# published. Both regimes here use the same textbook 12-1 momentum rank; BEAR
# only differs by drawing from the defensive-sector subset. Output shape
# (signal / regime / strategy / score / rank per ticker) is unchanged.

TOP_N_BUY = 3   # SHOWCASE: placeholder — number of top-ranked names marked BUY
BULL_STRATEGY = 'Reference-Momentum-BULL'
BEAR_STRATEGY = 'Reference-Momentum-BEAR'


def _rank_by_momentum(tickers: list, regime: str, strategy: str) -> dict:
    try:
        hist = _fetch_closes(tickers, '2y')
    except Exception as e:
        print(f'  [RegimeStrategy] {regime} download error: {e}')
        return _hold_all(tickers, regime, strategy)
    if hist is None or hist.empty:
        print(f'  [RegimeStrategy] {regime}: no price data — holding all')
        return _hold_all(tickers, regime, strategy)

    window, skip = config.MOMENTUM_WINDOW, config.MOMENTUM_SKIP
    scores: dict = {}
    for t in tickers:
        if t not in hist.columns:
            continue
        px = hist[t].dropna()
        if len(px) <= window:
            continue
        scores[t] = float(px.iloc[-skip - 1] / px.iloc[-window - 1] - 1)

    if not scores:
        return _hold_all(tickers, regime, strategy)

    ranked = pd.Series(scores).sort_values(ascending=False)
    results: dict = {}
    for rank_i, (t, score) in enumerate(ranked.items(), 1):
        results[t] = {
            'signal': 'BUY' if rank_i <= TOP_N_BUY else 'HOLD',
            'regime': regime, 'strategy': strategy,
            'score': round(score, 4), 'rank': rank_i,
        }
    for t in tickers:
        results.setdefault(t, {'signal': 'HOLD', 'regime': regime, 'strategy': strategy,
                               'score': 0.0, 'rank': len(ranked) + 1})
    return results


def _compute_bull_signals() -> dict:
    return _rank_by_momentum(BULL_TICKERS, 'BULL', BULL_STRATEGY)


def _compute_bear_signals() -> dict:
    return _rank_by_momentum(BEAR_TICKERS, 'BEAR', BEAR_STRATEGY)


# ── Public API ───────────────────────────────────────────────────────────────

def generate_regime_signals(regime: str) -> dict:
    today_cache = _today_cache_path(regime)

    if os.path.exists(today_cache):
        try:
            with open(today_cache) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("[RegimeStrategy] Today cache load failed: %s", e)

    # Reuse this month's cache unless regime switched today
    if not _switched_today():
        month_cache = _find_month_cache(regime)
        if month_cache is not None:
            return month_cache

    print(f'  [RegimeStrategy] Computing fresh signals for {regime} regime...')
    try:
        if regime == 'BULL':
            signals = _compute_bull_signals()
        elif regime == 'BEAR':
            signals = _compute_bear_signals()
        else:
            # Regime is only ever BULL or BEAR now (agents.regime_agent);
            # anything else has no strategy defined for it.
            signals = {}

    except Exception as e:
        print(f'  [RegimeStrategy] Computation failed: {e}')
        if regime == 'BEAR':
            signals = _hold_all(BEAR_TICKERS, regime, BEAR_STRATEGY)
        elif regime == 'BULL':
            signals = _hold_all(BULL_TICKERS, regime, BULL_STRATEGY)
        else:
            signals = {}

    _save_signals(signals, regime)
    return signals
