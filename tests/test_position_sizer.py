"""Slot-based sizing helpers (Sprint 1, design section B).

These are pure functions shared by the live entry path and the daily
backtest; the backtest is only evidence about live sizing if both call the
same code, so the tests pin the arithmetic here once.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import numpy as np
import pandas as pd

import config
from agents.position_sizer import (
    conviction_mult, plan_entries, realized_vol_63d, slot_size_pct, vol_scalar,
)


# ── config additions ─────────────────────────────────────────────────────────



# ── vol_scalar ───────────────────────────────────────────────────────────────



def test_vol_scalar_clamps_both_ends():
    assert vol_scalar(0.05) == pytest.approx(config.SIZING_VOL_SCALAR_MAX)   # 5.0 → 1.5
    assert vol_scalar(1.00) == pytest.approx(config.SIZING_VOL_SCALAR_MIN)   # 0.25 → 0.5


def test_vol_scalar_neutral_when_sigma_unknown_or_degenerate():
    assert vol_scalar(None) == 1.0
    assert vol_scalar(0.0) == 1.0
    assert vol_scalar(0.01) == 1.0
    assert vol_scalar(float("nan")) == 1.0


# ── conviction_mult ──────────────────────────────────────────────────────────



def test_conviction_mult_neutral_when_none_degraded_or_unmapped():
    assert conviction_mult(None) == 1.0
    assert conviction_mult(5, degraded=True) == 1.0
    assert conviction_mult(2) == 1.0


# ── slot_size_pct ────────────────────────────────────────────────────────────





def test_slot_size_pct_defaults_to_config_caps():
    # Without an explicit max_pos the live cap applies (config.MAX_POSITION_PCT).
    assert slot_size_pct(1.0, 5, 0.05) == pytest.approx(config.MAX_POSITION_PCT)




# ── realized_vol_63d ─────────────────────────────────────────────────────────
# Formula:
#   returns[t].tail(63).std() * sqrt(252)

def test_realized_vol_63d_matches_the_documented_formula():
    idx = pd.bdate_range("2024-01-01", periods=90)
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    closes = pd.DataFrame({"AAPL": prices}, index=idx)
    got = realized_vol_63d(closes)
    expected = closes["AAPL"].pct_change().tail(63).std() * math.sqrt(252)
    assert got["AAPL"] == pytest.approx(expected)


def test_realized_vol_63d_omits_tickers_with_no_usable_data():
    idx = pd.bdate_range("2024-01-01", periods=90)
    closes = pd.DataFrame({
        "AAPL": np.linspace(100, 110, len(idx)),
        "MSFT": [float("nan")] * len(idx),
    }, index=idx)
    got = realized_vol_63d(closes)
    assert "MSFT" not in got
    assert "AAPL" in got and math.isfinite(got["AAPL"])


def test_realized_vol_63d_empty_frame_returns_empty_dict():
    assert realized_vol_63d(pd.DataFrame()) == {}


# ── plan_entries ─────────────────────────────────────────────────────────────
# equity=100_000, long_market_value=0 -> gross_room = 1.20 * 100_000 = 120_000
_EQUITY = 100_000.0


def _plan(candidates, **kw):
    kw.setdefault("scores", {t: 0.0 for t in candidates})
    kw.setdefault("convictions", {})
    kw.setdefault("sigmas", {})
    kw.setdefault("regime_mult", 1.0)
    kw.setdefault("equity", _EQUITY)
    kw.setdefault("long_market_value", 0.0)
    kw.setdefault("deployable", 1_000_000.0)   # far above the gross room by default
    return plan_entries(candidates, **kw)






















