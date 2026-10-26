import math
from unittest.mock import patch

import pandas as pd
import numpy as np
import config
from agents import risk_agent


def test_max_position_pct_matches_config():
    assert risk_agent.MAX_POSITION_PCT == config.MAX_POSITION_PCT, (
        f"risk_agent uses {risk_agent.MAX_POSITION_PCT} but config says {config.MAX_POSITION_PCT}"
    )


def test_no_false_violation_at_config_size():
    # Mock both fetches: get_stock_price for the per-ticker checks and
    # get_closes for _avg_pairwise_correlation, which went to Yahoo for real.
    fake_prices = pd.DataFrame({"Close": np.random.randn(252).cumsum() + 100})
    fake_prices.index = pd.date_range("2024-01-01", periods=252, freq="B")
    fake_closes = pd.DataFrame(
        {"AAPL": fake_prices["Close"].values, "MSFT": fake_prices["Close"].values[::-1]},
        index=fake_prices.index,
    )

    with patch("agents.risk_agent.get_stock_price", return_value=fake_prices), \
         patch("agents.risk_agent.get_closes", return_value=fake_closes):
        positions = {"AAPL": config.MAX_POSITION_PCT}
        result = risk_agent.analyze_portfolio_risk(positions, universe=["AAPL", "MSFT"])
        violations = [v for v in result["violations"] if "exceeds max position" in v]
        assert violations == [], f"False violation fired: {violations}"


def test_compute_position_size_default_uses_config_cap():
    # With no explicit max_pos_pct, the default should be config.MAX_POSITION_PCT (0.15)
    capital = 100_000
    entry = 100.0
    sl = 95.0
    shares = risk_agent.compute_position_size(capital=capital, entry=entry, sl=sl)
    expected_cap = math.floor(capital * config.MAX_POSITION_PCT / entry)  # = 150
    assert shares <= expected_cap, (
        f"compute_position_size returned {shares} shares, exceeding cap {expected_cap} "
        f"(implies default max_pos_pct > {config.MAX_POSITION_PCT})"
    )
    assert shares > 0, "compute_position_size returned 0 unexpectedly"
