# tests/test_sector_rotation_agent.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch
import numpy as np
import pandas as pd

import config
from agents.sector_rotation_agent import (
    _rank_sectors,
    _rank_to_multiplier,
    _compute_sector_momentum,
    get_sector_multipliers,
    apply_sector_cap,
    SECTOR_ETFS,
)


def test_rank_to_multiplier_terciles():
    n = 11  # 11 SPDR sectors -> boundaries 1-3 / 4-7 / 8-11
    assert _rank_to_multiplier(1, n) == config.SECTOR_TOP_MULT
    assert _rank_to_multiplier(3, n) == config.SECTOR_TOP_MULT
    assert _rank_to_multiplier(4, n) == config.SECTOR_MID_MULT
    assert _rank_to_multiplier(7, n) == config.SECTOR_MID_MULT
    assert _rank_to_multiplier(8, n) == config.SECTOR_BOT_MULT
    assert _rank_to_multiplier(11, n) == config.SECTOR_BOT_MULT


def test_rank_sectors_orders_by_score_desc():
    scores = {"XLK": 0.5, "XLF": 0.1, "XLE": -0.2}
    ranks = _rank_sectors(scores)
    assert ranks["XLK"] == 1
    assert ranks["XLF"] == 2
    assert ranks["XLE"] == 3


def test_get_sector_multipliers_maps_by_rank():
    fake = {etf: 0.0 for etf in SECTOR_ETFS}
    fake["XLK"] = 0.9    # Technology -> rank 1 (top tercile)
    fake["XLE"] = -0.9   # Energy     -> rank 11 (bottom tercile)
    with patch("agents.sector_rotation_agent._compute_sector_momentum", return_value=fake):
        m = get_sector_multipliers(["AAPL", "XOM"])
    assert m["AAPL"] == config.SECTOR_TOP_MULT   # AAPL -> Technology -> XLK
    assert m["XOM"] == config.SECTOR_BOT_MULT    # XOM  -> Energy     -> XLE


def test_get_sector_multipliers_unknown_sector_is_neutral():
    fake = {etf: 0.0 for etf in SECTOR_ETFS}
    with patch("agents.sector_rotation_agent._compute_sector_momentum", return_value=fake), \
         patch("agents.sector_rotation_agent._get_ticker_sector", return_value="Unknown"):
        m = get_sector_multipliers(["ZZZZ"])
    assert m["ZZZZ"] == 1.0


def test_apply_sector_cap_caps_tech_at_30pct():
    # AAPL, AVGO, MSFT all map to Technology in the cache -> 45% > 30% cap
    weights = {"AAPL": 0.15, "AVGO": 0.15, "MSFT": 0.15, "JPM": 0.10}
    capped = apply_sector_cap(weights, list(weights), max_sector_pct=0.30)
    tech = capped["AAPL"] + capped["AVGO"] + capped["MSFT"]
    assert abs(tech - 0.30) < 1e-9          # tech scaled down to exactly 30%
    assert abs(capped["JPM"] - 0.10) < 1e-9  # financials untouched


def test_compute_sector_momentum_with_mocked_prices():
    idx = pd.date_range("2025-01-01", periods=300, freq="B")
    data = {etf: pd.Series(np.linspace(100.0, 200.0, 300), index=idx) for etf in SECTOR_ETFS}
    prices = pd.DataFrame(data)
    with patch("agents.sector_rotation_agent.get_closes", return_value=prices):
        scores = _compute_sector_momentum()
    assert set(scores) == set(SECTOR_ETFS)
    assert all(isinstance(v, float) for v in scores.values())
    assert scores["XLK"] > 0  # steady uptrend -> positive momentum
