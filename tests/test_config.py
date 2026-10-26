# tests/test_config.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


def test_weights_sum_to_one():
    assert abs(config.MOM_WEIGHT + config.VALUE_WEIGHT + config.SENTIMENT_WEIGHT - 1.0) < 1e-9


def test_signal_threshold_in_range():
    assert 0.0 <= config.SIGNAL_THRESHOLD <= 1.0



def test_target_positions_at_the_cap_reach_the_gross_target():
    # Sprint 1 design B: the per-position cap must not bind before the book
    # reaches its slot-based gross target, or TARGET_GROSS_EXPOSURE is
    # unreachable by construction.
    assert config.TARGET_POSITIONS * config.MAX_POSITION_PCT >= config.TARGET_GROSS_EXPOSURE



def test_budget_base_and_gross_cap_are_coherent():
    assert config.BUDGET_BASE in ("buying_power", "equity")
    assert config.MAX_GROSS_EXPOSURE >= 1.0
    # Reaching the gross target requires this many names at the per-position cap;
    # a universe that cannot supply them makes the target unreachable by design.
    needed = config.MAX_GROSS_EXPOSURE / config.MAX_POSITION_PCT
    assert len(config.TICKERS) >= needed


def test_bear_min_conviction_above_normal():
    assert config.BEAR_MIN_CONVICTION == 4
    assert config.BEAR_MIN_CONVICTION > config.MIN_PM_CONVICTION


def test_effective_min_conviction_by_regime():
    assert config.effective_min_conviction("BEAR") == config.BEAR_MIN_CONVICTION
    assert config.effective_min_conviction("BEAR") == 4
    assert config.effective_min_conviction("BULL") == config.MIN_PM_CONVICTION
    assert config.effective_min_conviction("BULL") == 3


def test_momentum_window_positive():
    assert config.MOMENTUM_WINDOW > 0

def test_value_lookback_is_252():
    assert config.VALUE_LOOKBACK == 252


# ── Exit engine (Sprint 1, design C) ────────────────────────────────────────



def test_retired_exit_constants_are_gone():
    # The TP ladder, the time-decay agent and the weekly rotation are retired;
    # a constant that survives invites a caller to be wired back to it.
    for name in ("DECAY_THRESHOLD", "DECAY_CONSECUTIVE_DAYS", "WEEKLY_ROTATION_ENABLED",
                 "ROTATION_WEEKDAY", "ROTATION_BOTTOM_PCT", "ROTATION_MIN_HOLD_DAYS"):
        assert not hasattr(config, name), f"config.{name} should have been deleted"


def test_retired_adx_regime_constants_are_gone():
    # Sprint 1 task 4: the ADX 4-state ladder (BULL/SIDEWAYS/CHOPPY/BEAR) is
    # replaced by direction x breadth x VIX (agents.regime_agent). A survivor
    # here invites a caller to be wired back to the old gate.
    for name in ("ADX_TRENDING_THRESHOLD", "ADX_WEAKENING_THRESHOLD",
                 "SIDEWAYS_MIN_CONVICTION", "CHOPPY_ENTRIES_ENABLED",
                 "CHOPPY_SIZE_MULT", "CHOPPY_MIN_CONVICTION"):
        assert not hasattr(config, name), f"config.{name} should have been deleted"
