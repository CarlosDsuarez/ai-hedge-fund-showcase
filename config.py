# config.py — Single source of truth for strategy and runtime parameters.
# All other modules import from here; change once, changes everywhere.
#
# SHOWCASE: every numeric parameter in this file is a neutral PLACEHOLDER chosen
# for the public demo (round numbers / textbook defaults). The tuned values,
# the curated universe and the research behind them are not published. The
# names, types and relationships between parameters are the real interface.

# ── Universe ────────────────────────────────────────────────────────────
# SHOWCASE: a small illustrative universe of liquid US large caps (two per
# GICS sector). Every name must appear in
# agents/sector_rotation_agent.TICKER_SECTOR_CACHE so the sector cap applies.
TICKERS = [
    "AAPL", "MSFT",     # Technology
    "GOOGL", "META",    # Communication Services
    "AMZN", "HD",       # Consumer Cyclical
    "PG", "KO",         # Consumer Defensive
    "JPM", "V",         # Financial Services
    "UNH", "JNJ",       # Healthcare
    "XOM", "CVX",       # Energy
    "CAT", "HON",       # Industrials
    "NEE", "DUK",       # Utilities
    "AMT", "PLD",       # Real Estate
    "LIN", "SHW",       # Basic Materials
]

# ── Signal parameters ───────────────────────────────────────────────────
# Composite = MOM_WEIGHT · z(momentum) + VALUE_WEIGHT · z(value) [+ sentiment].
MOM_WEIGHT        = 0.50   # SHOWCASE: placeholder
VALUE_WEIGHT      = 0.50   # SHOWCASE: placeholder
SENTIMENT_WEIGHT  = 0.0    # sentiment feeds the PM prompt / gates, not the composite
MOMENTUM_WINDOW   = 252    # 12-month lookback (Jegadeesh-Titman 12-1)
MOMENTUM_SKIP     = 21     # skip the most recent month
VALUE_LOOKBACK    = 252    # rolling window for the 52-week-high proxy
# Multi-lookback momentum weights (must sum to 1.0)
MOM_LOOKBACK_3M_WEIGHT  = 0.25   # SHOWCASE: placeholder
MOM_LOOKBACK_6M_WEIGHT  = 0.25   # SHOWCASE: placeholder
MOM_LOOKBACK_12M_WEIGHT = 0.50   # SHOWCASE: placeholder
SIGNAL_THRESHOLD  = 0.5    # SHOWCASE: placeholder — composite z cutoff for BUY

# ── Cross-asset macro filter ─────────────────────────────────────────────
# Credit-stress proxy (HYG vs LQD); above the threshold new longs are suppressed.
CREDIT_STRESS_THRESHOLD = 2.0   # SHOWCASE: placeholder (z-score)
MACRO_LOOKBACK          = 252

# ── Budget & sizing ─────────────────────────────────────────────────────
BUDGET_ALLOCATION = 0.50          # SHOWCASE: placeholder — fraction of BUDGET_BASE deployed
BUDGET_BASE       = "equity"      # "buying_power" | "equity"
MAX_GROSS_EXPOSURE = 1.00         # SHOWCASE: placeholder — long market value / equity ceiling
MAX_POSITION_PCT  = 0.10          # SHOWCASE: placeholder — hard cap per position (fraction of equity)
DRIFT_TRIM_HYSTERESIS = 1.20      # trim only once a position exceeds cap × this
MIN_POSITION_PCT  = 0.01          # don't submit orders smaller than 1% of equity

# ── Slot-based sizing ───────────────────────────────────────────────────
# One slot = TARGET_GROSS_EXPOSURE / TARGET_POSITIONS of equity, then scaled by
# regime × conviction × volatility × sector multipliers (agents/position_sizer).
TARGET_POSITIONS       = 8        # SHOWCASE: placeholder
TARGET_GROSS_EXPOSURE  = 0.80     # SHOWCASE: placeholder (8 × 0.10)
SIZING_VOL_TARGET      = 0.20     # SHOWCASE: placeholder — annualised σ target
SIZING_VOL_SCALAR_MIN  = 0.5
SIZING_VOL_SCALAR_MAX  = 1.5
CONVICTION_SIZE_MULT   = {5: 1.0, 4: 0.8, 3: 0.6}   # SHOWCASE: placeholder

# ── Regime size multiplier ──────────────────────────────────────────────
# Direction × breadth × VIX, floored at REGIME_SIZE_FLOOR. The regime scales
# the slot; it never blocks an entry. See agents/regime_agent.size_multiplier_for.
BEAR_SIZE_MULT      = 0.5         # SHOWCASE: placeholder
REGIME_SIZE_FLOOR   = 0.5         # SHOWCASE: placeholder
REGIME_BASE_FLOOR   = 0.5         # SHOWCASE: placeholder (layered mode)
BREADTH_STRONG_PCT  = 0.50        # SHOWCASE: placeholder
BREADTH_WEAK_PCT    = 0.40        # SHOWCASE: placeholder
BREADTH_MULTS       = (1.0, 0.75, 0.5)   # SHOWCASE: placeholder
VIX_ELEVATED        = 20.0        # SHOWCASE: placeholder
VIX_EXTREME         = 30.0        # SHOWCASE: placeholder
VIX_MULTS           = (1.0, 0.75, 0.5)   # SHOWCASE: placeholder

# ── Sector rotation overlay ──────────────────────────────────────────────
SECTOR_ROTATION_ENABLED = True
MAX_SECTOR_WEIGHT       = 0.35    # SHOWCASE: placeholder — max sector weight of equity
SECTOR_TOP_MULT         = 1.10    # SHOWCASE: placeholder
SECTOR_MID_MULT         = 1.00
SECTOR_BOT_MULT         = 0.90    # SHOWCASE: placeholder

# ── Risk-free rate ───────────────────────────────────────────────────────
RISK_FREE_ANNUAL = 0.04           # SHOWCASE: placeholder

# ── Conviction gate for PM veto ─────────────────────────────────────────
MIN_PM_CONVICTION = 3             # PM conviction must be >= this to allow entry

# ── Exit engine ──────────────────────────────────────────────────────────
# One initial ATR stop, a chandelier trailing stop that only rises, a
# rank-decay exit, a maximum holding period and a re-entry cooldown.
STOP_ATR_MULT          = 2.0      # SHOWCASE: placeholder — initial stop = entry − k·ATR14
TRAIL_ATR_MULT         = 3.0      # textbook chandelier multiple
RANK_DECAY_PCTL        = 0.50     # SHOWCASE: placeholder
RANK_DECAY_DAYS        = 5        # SHOWCASE: placeholder
MAX_HOLDING_DAYS       = 120      # SHOWCASE: placeholder (calendar days)
PM_SELL_MIN_CONVICTION = 4        # SHOWCASE: placeholder
REENTRY_COOLDOWN_DAYS  = 5        # SHOWCASE: placeholder (trading days)

# ── Entry freeze ─────────────────────────────────────────────────────────
# Operational switch: when True the engine places no new buys (exits keep running).
ENTRIES_FROZEN = False

# ── Pending-signal queue ─────────────────────────────────────────────────
PENDING_EXPIRY_DAYS = 14          # drop queued BUYs older than this (calendar days)

# ── Regime-dependent conviction gate ─────────────────────────────────────
BEAR_MIN_CONVICTION = 4           # SHOWCASE: placeholder


def effective_min_conviction(regime: str) -> int:
    """The conviction bar entries actually face in `regime`.

    Single source of truth for the gate: run_entries enforces it, and the PM
    prompt quotes it so the model calibrates against the real bar.
    """
    return BEAR_MIN_CONVICTION if regime == "BEAR" else MIN_PM_CONVICTION

# ── Data quality guard ──────────────────────────────────────────────────
# Share of the universe whose price history may fail before the run aborts.
MAX_DATA_FAILURE_PCT = 0.20

# ── LLM model names ─────────────────────────────────────────────────────
CLAUDE_RESEARCH_MODEL  = "claude-sonnet-4-6"
CLAUDE_SENTIMENT_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_PM_MODEL        = "claude-sonnet-4-6"

# ── LLM cost governance ─────────────────────────────────────────────────
# agents/cost_governor.py decides, before every API call, whether we may spend
# and in which mode. Tiers are evaluated on the remaining fraction of the
# daily budget.
DAILY_LLM_BUDGET_USD      = 1.00   # SHOWCASE: placeholder — daily Anthropic spend cap
BUDGET_TIER_REDUCED_PCT   = 0.40   # < 40% remaining → REDUCED
BUDGET_TIER_MINIMAL_PCT   = 0.15   # < 15% remaining → MINIMAL
BUDGET_TIER_BLOCKED_PCT   = 0.05   # < 5%  remaining → BLOCKED
BUDGET_ENFORCEMENT        = True   # False = log only, never degrade

# ── Research call chunking ───────────────────────────────────────────────
# Split the universe into chunks so one bad LLM response only affects its chunk.
RESEARCH_CHUNK_SIZE            = 20    # tickers per LLM call
RESEARCH_MAX_WORKERS           = 5     # concurrent chunk calls
RESEARCH_CHUNK_RETRIES         = 2     # extra attempts per chunk
RESEARCH_DEGRADED_FALLBACK_PCT = 0.50  # > 50% of tickers on HOLD fallback ⇒ degraded run

# ── Ledger ───────────────────────────────────────────────────────────────
# execution/ledger.reconcile_fills() rebuilds a FIFO trades table from broker
# fills at the top of every exit cycle.
LEDGER_ENABLED       = True
LEDGER_BACKFILL_FROM = "2026-01-01"   # SHOWCASE: placeholder
