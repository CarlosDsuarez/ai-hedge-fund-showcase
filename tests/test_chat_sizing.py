"""chat/sizing.py — sizing ONE name without handing it the whole budget.

main.py divides the deployable fraction by the number of candidates. Feed that
one ticker and the divisor is 1, so the raw allocation is the entire budget
and MAX_POSITION_PCT stops being a ceiling and becomes the answer: every
proposal would be exactly the cap, whatever the name.
"""
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from chat.sizing import peer_count_from_run, suggest_size

EQUITY = 100_000.0
# 0.75 * 186,666.67 = 140,000 -> deployable_frac = 1.4, the plan's worked example.
BUYING_POWER = 186_666.67


def _size(**over):
    kw = dict(ticker="AAPL", price=200.0, equity=EQUITY, buying_power=BUYING_POWER,
              long_market_value=0.0, vol_size_mult=1.0, peer_count=1,
              current_mv=0.0, sector_mv=0.0)
    kw.update(over)
    return suggest_size(**kw)








def test_the_clamp_is_still_a_ceiling():
    """A tiny budget must not be inflated, a huge one must not breach the cap."""
    rich = _size(buying_power=1_000_000.0, peer_count=1)
    assert rich.alloc_pct <= config.MAX_POSITION_PCT

    poor = _size(buying_power=100.0, peer_count=1)
    assert poor.alloc_pct >= config.MIN_POSITION_PCT






def test_an_exhausted_cap_yields_zero_shares_not_a_negative_one():
    # A current position well above the per-name cap leaves negative headroom.
    s = _size(current_mv=25_000.0)
    assert s.cap("position").headroom < 0
    assert s.spend == 0.0
    assert s.qty == 0




def test_no_price_or_no_equity_is_reported_not_guessed():
    assert _size(price=0.0).qty == 0
    assert _size(equity=0.0).qty == 0
    assert "no se puede dimensionar" in _size(price=0.0).budget_reason


# ── peer_count_from_run ─────────────────────────────────────────────────────

def test_peer_count_prefers_the_gate_stage_counter():
    run = {"funnel": {"stages": {"pm": 40, "gates": 7}},
           "orders_submitted": [{"ticker": "A"}]}
    assert peer_count_from_run(run) == 7


def test_peer_count_falls_back_to_survivors_then_orders():
    assert peer_count_from_run({"funnel": {"executed": ["A", "B"], "queued": ["C"]}}) == 3
    assert peer_count_from_run({"orders_submitted": [{"ticker": "A"}, {"ticker": "B"}]}) == 2
    assert peer_count_from_run({}) == 1
