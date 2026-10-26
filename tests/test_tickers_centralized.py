import re
from collections import Counter

import config
import agents.research_agent as ra
import agents.sentiment_agent as sa
from agents.sector_rotation_agent import TICKER_SECTOR_CACHE

TICKER_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")

# Share-class pairs that must never both be in the universe: correlation ~1.0
# poisons any covariance matrix built on the universe.
SHARE_CLASS_PAIRS = [("GOOG", "GOOGL"), ("BRK-A", "BRK-B")]

MIN_NAMES_PER_SECTOR = 3


def test_research_agent_tickers_match_config():
    assert ra.TICKERS == config.TICKERS, (
        "research_agent.TICKERS diverges from config.TICKERS"
    )


def test_sentiment_agent_tickers_match_config():
    assert sa.TICKERS == config.TICKERS, (
        "sentiment_agent.TICKERS diverges from config.TICKERS"
    )




def test_no_duplicate_tickers():
    dupes = [t for t, n in Counter(config.TICKERS).items() if n > 1]
    assert not dupes, f"duplicate tickers in config.TICKERS: {dupes}"


def test_ticker_symbols_are_well_formed():
    bad = [t for t in config.TICKERS if not TICKER_RE.match(t)]
    assert not bad, f"malformed ticker symbols: {bad}"


def test_no_share_class_duplicates():
    universe = set(config.TICKERS)
    both = [pair for pair in SHARE_CLASS_PAIRS if universe.issuperset(pair)]
    assert not both, (
        f"both share classes present (correlation ~1.0 breaks covariance): {both}"
    )


def test_every_ticker_has_a_cached_sector():
    # A ticker missing from the cache falls back to a slow live yfinance lookup
    # on the hot path, so the cache must cover the whole universe.
    missing = [t for t in config.TICKERS if t not in TICKER_SECTOR_CACHE]
    assert not missing, f"tickers absent from TICKER_SECTOR_CACHE: {missing}"




def test_regime_strategy_universe_tracks_config():
    """The regime layer must score the universe the pipeline actually analyses.

    BULL_TICKERS and BEAR_TICKERS were hand-written literals that drifted from
    config.TICKERS in both directions — naming symbols never analysed while
    leaving a third of the universe with no regime signal at all. On a healthy
    cycle a regime BUY is by itself sufficient evidence to enter
    (main._signal_gate_ok), so that drift quietly decided which names could be
    bought on regime strength alone.
    """
    from agents.regime_strategy_agent import BEAR_TICKERS, BULL_TICKERS

    universe = set(config.TICKERS)
    assert set(BULL_TICKERS) <= universe
    assert set(BEAR_TICKERS) <= universe
    assert set(BULL_TICKERS) == universe, "BULL should score the whole universe"
    assert BEAR_TICKERS, "the defensive subset must never be empty"
