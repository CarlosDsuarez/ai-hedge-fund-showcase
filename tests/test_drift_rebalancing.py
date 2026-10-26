import math
import pytest
import config
from execution.drift_rebalancer import compute_trim_orders, extract_weight_violating_tickers


# ── compute_trim_orders ──────────────────────────────────────────────────────

def test_compute_trim_orders_flags_overweight_position():
    positions = [
        {"symbol": "LLY", "market_value": "30200.0", "current_price": "800.0", "qty": "37"}
    ]
    equity = 100_000.0
    orders = compute_trim_orders(positions, equity)
    assert len(orders) == 1
    assert orders[0]["ticker"] == "LLY"
    assert orders[0]["weight"] == pytest.approx(0.302, rel=1e-4)
    assert orders[0]["shares_to_sell"] == 37 - math.floor(100_000 * config.MAX_POSITION_PCT / 800.0)


def test_compute_trim_orders_shares_formula():
    # equity=100_000, MAX_POSITION_PCT=0.15, price=800, qty=37
    # target_qty = floor(100_000 * 0.15 / 800) = floor(18.75) = 18
    # shares_to_sell = 37 - 18 = 19
    positions = [
        {"symbol": "LLY", "market_value": "30200.0", "current_price": "800.0", "qty": "37"}
    ]
    equity = 100_000.0
    orders = compute_trim_orders(positions, equity)
    expected_target = math.floor(equity * config.MAX_POSITION_PCT / 800.0)
    assert orders[0]["ticker"] == "LLY"
    assert orders[0]["weight"] == pytest.approx(0.302, rel=1e-4)
    assert orders[0]["shares_to_sell"] == 37 - expected_target






# ── Hysteresis (I4): slots are sized to the cap by design ────────────────────









def test_compute_trim_orders_skips_short_positions():
    positions = [
        {"symbol": "TSLA", "market_value": "-5000.0", "current_price": "200.0", "qty": "25"}
    ]
    equity = 100_000.0
    orders = compute_trim_orders(positions, equity)
    assert orders == []


def test_compute_trim_orders_empty_positions():
    orders = compute_trim_orders([], 100_000.0)
    assert orders == []


def test_compute_trim_orders_zero_equity():
    positions = [
        {"symbol": "LLY", "market_value": "30200.0", "current_price": "800.0", "qty": "37"}
    ]
    orders = compute_trim_orders(positions, 0.0)
    assert orders == []




def test_compute_trim_orders_normalizes_alpaca_symbol():
    # 27,000 clears the 25% trim trigger (20% cap x 1.25 hysteresis).
    positions = [
        {"symbol": "BRK.B", "market_value": "27000.0", "current_price": "400.0", "qty": "67"}
    ]
    equity = 100_000.0
    orders = compute_trim_orders(positions, equity)
    assert orders[0]["ticker"] == "BRK-B"


# ── extract_weight_violating_tickers ────────────────────────────────────────

def test_extract_weight_violating_tickers_parses_violations():
    violations = [
        "LLY exceeds max position (30.2% > 15.0%)",
        "AVGO exceeds max position (16.4% > 15.0%)",
    ]
    result = extract_weight_violating_tickers(violations)
    assert result == {"LLY", "AVGO"}


def test_extract_weight_violating_tickers_ignores_other_violations():
    violations = [
        "Portfolio max drawdown 25.0% exceeds limit 20.0%",
        "LLY exceeds max position (30.2% > 15.0%)",
    ]
    result = extract_weight_violating_tickers(violations)
    assert result == {"LLY"}


def test_extract_weight_violating_tickers_empty_violations():
    result = extract_weight_violating_tickers([])
    assert result == set()
