import json
import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

import config
from data.fetcher import get_closes
from data import synthetic

logger = logging.getLogger(__name__)

LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
REGIME_CACHE = os.path.join(LOGS_DIR, 'regime_state.json')


def _load_cache() -> dict:
    try:
        if os.path.exists(REGIME_CACHE):
            with open(REGIME_CACHE) as f:
                return json.load(f)
    except Exception as e:
        logger.warning("[RegimeAgent] Cache load failed: %s", e)
    return {}


def _save_cache(result: dict):
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(REGIME_CACHE, 'w') as f:
        json.dump(result, f, indent=2)


def _default_result() -> dict:
    return {
        'regime':              'BULL',
        'adx':                 25.0,
        'new_entries_allowed': True,
        'spy_price':           0.0,
        'spy_ema200':          0.0,
        'spy_sma20':           0.0,
        'buffer_pct':          0.0,
        'confirmed':           False,
        'days_in_regime':      0,
        'switched_today':      False,
        'vix':                 20.0,
        'vol_regime':          'MED',
        'size_multiplier':     1.0,
        'breadth_pct':         1.0,
        'breadth_mult':        1.0,
        'vix_mult':            1.0,
        'direction_mult':      1.0,
        'vix_sma10':           None,
        'credit_signal':       'NORMAL',
        'switch_count_ytd':    0,
        'switch_year':         0,
    }


MAX_SWITCHES_PER_YEAR = 6


def _apply_whipsaw_guard(candidate_regime: str, prev_cache: dict) -> str:
    """Block a regime switch if the annual limit has been reached."""
    from datetime import date
    current_year = date.today().year
    switch_year = prev_cache.get('switch_year', current_year)
    switch_count = prev_cache.get('switch_count_ytd', 0) if switch_year == current_year else 0
    if switch_count >= MAX_SWITCHES_PER_YEAR:
        prev_regime = prev_cache.get('regime', candidate_regime)
        # A legacy cache (pre-Task-4 CHOPPY/SIDEWAYS, or anything else that
        # isn't a direction) has nothing valid to preserve — fall through to
        # the freshly computed candidate instead of leaking the stale label
        # into final_regime.
        return prev_regime if prev_regime in ('BULL', 'BEAR') else candidate_regime
    return candidate_regime


def _compute_adx(spy_ohlcv: pd.DataFrame, period: int = 14) -> float:
    """
    Compute ADX (Average Directional Index) for SPY.

    ADX measures trend strength, not direction (0=no trend, 100=max trend).
    Uses Wilder's smoothing (EMA with alpha=1/period).

    Thresholds (Wilder 1978):
      ADX >= 25  → trending market (momentum strategies valid)
      ADX 20-25  → weakening trend (reduce momentum exposure)
      ADX < 20   → non-trending (momentum fails; mean-reversion dominates)
    """
    try:
        if spy_ohlcv is None or len(spy_ohlcv) < period * 3:
            return 25.0

        high  = spy_ohlcv["High"].values
        low   = spy_ohlcv["Low"].values
        close = spy_ohlcv["Close"].values

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:]  - close[:-1])
            )
        )
        # Directional Movement
        up_move   = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        def _wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
            out = np.empty_like(arr)
            out[0] = arr[:p].sum()
            for i in range(1, len(arr)):
                out[i] = out[i-1] - out[i-1]/p + arr[i]
            return out

        atr14      = _wilder_smooth(tr,       period)
        plus_di14  = 100 * _wilder_smooth(plus_dm,  period) / np.maximum(atr14, 1e-8)
        minus_di14 = 100 * _wilder_smooth(minus_dm, period) / np.maximum(atr14, 1e-8)

        di_sum  = plus_di14 + minus_di14
        dx      = 100 * np.abs(plus_di14 - minus_di14) / np.where(di_sum > 0, di_sum, 1)
        adx_raw = _wilder_smooth(dx, period)
        # _wilder_smooth seeds with sum (not average) of first p elements, so the
        # running value converges to p * true_avg.  Divide by period to recover
        # the standard 0-100 ADX scale.
        adx = adx_raw / period

        # Clamp to [0, 99.99] — ADX never reaches 100 in real markets;
        # perfect-trend synthetic data can produce exactly 100.0.
        return round(float(min(adx[-1], 99.99)), 2)

    except Exception:
        return 25.0


# ── Pure helpers shared with backtest/reference_backtest.py ─────────────────────
# Nothing below touches the network, the cache or config state that changes at
# runtime. The daily backtest imports these so the regime it simulates is the
# regime live will compute; a private re-implementation there would drift.

DIRECTION_BUFFER = 0.02       # (SMA20 - EMA200) / EMA200 must clear ±2% to count
DIRECTION_CONFIRM_DAYS = 5    # consecutive clear days needed to flip BULL <-> BEAR


def _direction_machine(buffer_values) -> tuple[list[str], int]:
    """Run the BULL/BEAR confirmation state machine over a buffer sequence.

    Returns the confirmed direction after each step and the final
    days_in_regime. Starts BULL; flips only after DIRECTION_CONFIRM_DAYS
    consecutive clear signals for the other side; a day inside the buffer
    resets the streak.
    """
    confirmed_regime = 'BULL'
    candidate = None
    candidate_streak = 0
    days_in_regime = 0
    path: list[str] = []

    for buf in buffer_values:
        if buf > DIRECTION_BUFFER:
            signal = 'BULL'
        elif buf < -DIRECTION_BUFFER:
            signal = 'BEAR'
        else:
            signal = None

        if signal is None:
            candidate = None
            candidate_streak = 0
        elif signal == confirmed_regime:
            days_in_regime += 1
            candidate = None
            candidate_streak = 0
        else:
            days_in_regime = 0
            if signal == candidate:
                candidate_streak += 1
                if candidate_streak >= DIRECTION_CONFIRM_DAYS:
                    confirmed_regime = candidate
                    days_in_regime = candidate_streak
                    candidate = None
                    candidate_streak = 0
            else:
                candidate = signal
                candidate_streak = 1
        path.append(confirmed_regime)

    return path, days_in_regime


def direction_state(buffer_series) -> tuple[str, int]:
    """Final (direction, days_in_regime) of the confirmation loop.

    This is the loop detect_regime has always run, lifted out unchanged so
    the backtest can replay it day by day.
    """
    path, days_in_regime = _direction_machine(buffer_series)
    return (path[-1] if path else 'BULL'), days_in_regime


def direction_series(spy_close: pd.Series) -> pd.Series:
    """Per-day confirmed direction ("BULL"/"BEAR") from a SPY close series.

    Same indicators as detect_regime — EMA200 vs SMA20 buffer, ±2% band, 5-day
    confirmation — evaluated cumulatively so the value on day t only uses
    closes up to t. Days before the SMA20 exists inherit the default BULL.
    """
    ema200 = spy_close.ewm(span=200, adjust=False).mean()
    sma20 = spy_close.rolling(20).mean()
    buffer_series = ((sma20 - ema200) / ema200).dropna()
    path, _ = _direction_machine(buffer_series.values)
    out = pd.Series('BULL', index=spy_close.index, dtype=object)
    out.loc[buffer_series.index] = path
    return out


def breadth_series(closes: pd.DataFrame) -> pd.Series:
    """Fraction of names closing above their own 200-day SMA, per day.

    A column without a 200-day SMA yet (e.g. a name whose fetch failed and
    left the column all-NaN) is excluded from the average via `.where`, not
    counted as "below" — otherwise a handful of missing names biases breadth
    down even though nothing about their trend is actually known.
    """
    sma = closes.rolling(200).mean()
    return (closes > sma).where(sma.notna()).mean(axis=1)


def breadth_multiplier(frac) -> float:
    """Breadth ladder from config: strong / middling / weak. Unknown → 1.0."""
    if frac is None or pd.isna(frac):
        return 1.0
    strong, mid, weak = config.BREADTH_MULTS
    if frac > config.BREADTH_STRONG_PCT:
        return strong
    if frac >= config.BREADTH_WEAK_PCT:
        return mid
    return weak


def vix_multiplier(vix, vix_sma10=None) -> float:
    """VIX ladder from config: calm / elevated / extreme. Unknown → 1.0.

    `vix_sma10` (the VIX's own 10-day mean) is optional. When given and the
    VIX is above VIX_EXTREME but at or below that mean — panic subsiding —
    the elevated tier applies instead of the extreme one. The extreme tier
    only sizes new entries, so a flat brake shrinks exactly the entries that
    catch the rebound; this keeps the brake while the spike is building and
    releases it once the VIX rolls over. `detect_regime` always passes
    `vix_sma10` (it may still be `None` early in a series, before 10 bars of
    VIX history exist); callers that omit it get the unchanged, VIX-only
    ladder.
    """
    if vix is None or pd.isna(vix):
        return 1.0
    calm, elevated, extreme = config.VIX_MULTS
    if vix < config.VIX_ELEVATED:
        return calm
    if vix <= config.VIX_EXTREME:
        return elevated
    if vix_sma10 is not None and not pd.isna(vix_sma10) and vix <= vix_sma10:
        return elevated
    return extreme


SIZE_MULT_MODES = ("product", "layered")


def size_multiplier_for(direction: str, breadth_frac, vix, vix_sma10=None, *,
                        mode: str = "product") -> float:
    """Regime size multiplier for one entry.

    mode="product": direction × breadth × VIX, floored at REGIME_SIZE_FLOOR.
    mode="layered": max(direction × breadth, REGIME_BASE_FLOOR) × VIX, then
                    clamped to [REGIME_SIZE_FLOOR, 1.0] — the base floor keeps a
                    confirmed BEAR with weak breadth at half slots, so only the
                    VIX tier takes the book below it.

    The floors are the design decision that the regime scales the book but
    never closes the entry path — the ADX 4-state gate that used to do that
    went 88 days without an order (docs/ADX_GATE_ANALYSIS.md; retired, see
    config.py). `vix_sma10` is passed through to vix_multiplier (see there).
    """
    if mode not in SIZE_MULT_MODES:
        raise ValueError(f"mode must be one of {SIZE_MULT_MODES}, got {mode!r}")
    direction_mult = config.BEAR_SIZE_MULT if direction == 'BEAR' else 1.0
    base = direction_mult * breadth_multiplier(breadth_frac)
    if mode == "layered":
        base = max(base, config.REGIME_BASE_FLOOR)
    product = base * vix_multiplier(vix, vix_sma10)
    if mode == "layered":
        return min(max(product, config.REGIME_SIZE_FLOOR), 1.0)
    return max(product, config.REGIME_SIZE_FLOOR)


def _sanitize_cached(result: dict) -> dict:
    """Coerce a cached regime_state.json onto the Task 4 contract.

    Every `_load_cache()` consumer in `detect_regime()` must pass its result
    through here before returning it. The cache can hold a pre-Task-4 (or
    otherwise stale) result — a CHOPPY/SIDEWAYS regime label,
    new_entries_allowed=False, a size_multiplier baked from the retired ADX
    ladder — and returning that verbatim reintroduces exactly the gate this
    task removed: main.run_entries rejects every candidate with
    REGIME_BLOCKED the moment it sees new_entries_allowed=False, and a
    stale sub-floor size_multiplier bypasses REGIME_SIZE_FLOOR entirely.

    `regime` is coerced to BULL/BEAR (unknown → BULL), `new_entries_allowed`
    is forced True, `trend_strength` (removed from the schema) is dropped,
    and `size_multiplier` is always recomputed through size_multiplier_for
    from the cached inputs — never trusted as a stored value — so it can
    never fall below REGIME_SIZE_FLOOR. Missing breadth/VIX inputs get the
    same neutral defaults detect_regime itself falls back to.
    """
    result = dict(result)
    regime = result.get('regime')
    if regime not in ('BULL', 'BEAR'):
        regime = 'BULL'

    def _numeric_or(value, default):
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
        return value

    breadth_pct = _numeric_or(result.get('breadth_pct'), 1.0)
    vix         = _numeric_or(result.get('vix'), 20.0)
    vix_sma10   = _numeric_or(result.get('vix_sma10'), None)

    result['regime']              = regime
    result['new_entries_allowed'] = True
    result['breadth_pct']         = breadth_pct
    result['vix']                 = vix
    result['vix_sma10']           = vix_sma10
    result['direction_mult']      = config.BEAR_SIZE_MULT if regime == 'BEAR' else 1.0
    result['breadth_mult']        = breadth_multiplier(breadth_pct)
    result['vix_mult']            = vix_multiplier(vix, vix_sma10)
    result['size_multiplier']     = size_multiplier_for(regime, breadth_pct, vix, vix_sma10, mode="product")
    result.pop('trend_strength', None)
    return result


def _fetch_universe_closes() -> pd.DataFrame:
    """Wide closes for config.TICKERS — breadth's input.

    Same period string as agents.strategy_agent.compute_fullalloc_signals
    ("2y") so the two share data.fetcher's 1-hour price cache instead of the
    run issuing a second Alpaca batch for the same tickers.
    """
    return get_closes(config.TICKERS, period="2y")


def detect_regime() -> dict:
    try:
        # Download FULL OHLCV (High/Low needed for ADX, kept for observability)
        raw = (synthetic.download(['SPY', '^VIX'], period='2y') if synthetic.enabled()
               else yf.download(['SPY', '^VIX'], period='2y', auto_adjust=True, progress=False))
        if raw.empty:
            cached = _load_cache()
            return _sanitize_cached(cached) if cached else _default_result()

        # Extract OHLCV for SPY
        if isinstance(raw.columns, pd.MultiIndex):
            spy_ohlcv = pd.DataFrame({
                "High":  raw["High"]["SPY"],
                "Low":   raw["Low"]["SPY"],
                "Close": raw["Close"]["SPY"],
            }).dropna()
            spy_closes = raw["Close"]["SPY"].dropna()
            vix_closes = (raw["Close"]["^VIX"].dropna()
                          if "^VIX" in raw["Close"].columns
                          else pd.Series(dtype=float))
        else:
            spy_closes = raw["Close"].dropna()
            spy_ohlcv  = raw[["High", "Low", "Close"]].dropna()
            vix_closes = pd.Series(dtype=float)

        closes = spy_closes
        if len(closes) < 220:
            cached = _load_cache()
            return _sanitize_cached(cached) if cached else _default_result()

        # Trend direction indicators
        ema200 = closes.ewm(span=200, adjust=False).mean()
        sma20  = closes.rolling(20).mean()
        buffer_series = ((sma20 - ema200) / ema200).dropna()

        # ADX(14) on SPY OHLCV — observability only, no longer a gate.
        adx = _compute_adx(spy_ohlcv, period=14)

        # Stateful simulation: BULL until 5 consecutive clear-signal days flip it
        confirmed_regime, days_in_regime = direction_state(buffer_series)

        spy_price  = float(closes.iloc[-1])
        spy_ema200 = float(ema200.iloc[-1])
        spy_sma20  = float(sma20.iloc[-1])
        buffer_pct = float(buffer_series.iloc[-1])

        # VIX / vol regime (label only — vix_multiplier does the sizing math)
        if not vix_closes.empty:
            vix = round(float(vix_closes.iloc[-1]), 2)
            vix_sma10_raw = vix_closes.rolling(10).mean().iloc[-1]
            vix_sma10 = None if pd.isna(vix_sma10_raw) else round(float(vix_sma10_raw), 2)
        else:
            vix = 20.0
            vix_sma10 = None

        if vix < 15:
            vol_regime = 'LOW'
        elif vix < config.VIX_ELEVATED:
            vol_regime = 'MED'
        elif vix < config.VIX_EXTREME:
            vol_regime = 'HIGH'
        else:
            vol_regime = 'EXTREME'
        credit_signal = 'STRESS' if vix >= config.VIX_ELEVATED else 'NORMAL'

        # Breadth: fraction of the traded universe above its own SMA200. A
        # failed fetch or a too-thin universe must not block sizing — fall
        # back to neutral (1.0) and say so.
        try:
            universe_closes = _fetch_universe_closes()
            if universe_closes is None or universe_closes.shape[1] < 20:
                print("  [RegimeAgent] Universe closes unavailable or too thin "
                      f"({0 if universe_closes is None else universe_closes.shape[1]} cols) "
                      "— breadth neutral (1.0)")
                breadth_pct = 1.0
            else:
                breadth_pct = float(breadth_series(universe_closes).iloc[-1])
                if pd.isna(breadth_pct):
                    breadth_pct = 1.0
        except Exception as e:
            print(f"  [RegimeAgent] Breadth fetch failed: {e} — breadth neutral (1.0)")
            breadth_pct = 1.0

        # Whipsaw guard on the confirmed direction — regime is BULL or BEAR,
        # it never blocks an entry; size_multiplier_for does the scaling.
        prev_cache   = _load_cache()
        prev_regime  = prev_cache.get('regime', 'BULL')
        final_regime = _apply_whipsaw_guard(confirmed_regime, prev_cache)

        direction_mult  = config.BEAR_SIZE_MULT if final_regime == 'BEAR' else 1.0
        breadth_mult    = breadth_multiplier(breadth_pct)
        vix_mult        = vix_multiplier(vix, vix_sma10)
        size_multiplier = size_multiplier_for(final_regime, breadth_pct, vix, vix_sma10, mode="product")

        switched_today = bool(prev_regime and prev_regime != final_regime)

        from datetime import date
        current_year = date.today().year
        prev_year    = prev_cache.get('switch_year', current_year)
        prev_count   = prev_cache.get('switch_count_ytd', 0) if prev_year == current_year else 0
        is_direction_switch = (
            prev_regime in ('BULL', 'BEAR') and final_regime in ('BULL', 'BEAR')
            and prev_regime != final_regime
        )
        switch_count_ytd = prev_count + (1 if is_direction_switch else 0)

        result = {
            'regime':              final_regime,
            'new_entries_allowed': True,
            'size_multiplier':     size_multiplier,
            'breadth_pct':         round(breadth_pct, 4),
            'breadth_mult':        breadth_mult,
            'vix_mult':            vix_mult,
            'direction_mult':      direction_mult,
            'vix_sma10':           vix_sma10,
            'vix':                 vix,
            'vol_regime':          vol_regime,
            'adx':                 adx,
            'spy_price':           round(spy_price, 2),
            'spy_ema200':          round(spy_ema200, 2),
            'spy_sma20':           round(spy_sma20, 2),
            'buffer_pct':          round(buffer_pct, 4),
            'confirmed':           days_in_regime >= 5,
            'days_in_regime':      days_in_regime,
            'switched_today':      switched_today,
            'credit_signal':       credit_signal,
            'switch_count_ytd':    switch_count_ytd,
            'switch_year':         current_year,
        }

        _save_cache(result)
        return result

    except Exception as e:
        print(f'  [RegimeAgent] Error: {e}')
        cached = _load_cache()
        return _sanitize_cached(cached) if cached else _default_result()
