import math

import config


def compute_trim_orders(positions: list, equity: float) -> list:
    """Return trim orders for positions that have drifted well past the cap.

    A position is trimmed when its weight exceeds
    MAX_POSITION_PCT x DRIFT_TRIM_HYSTERESIS,
    and it is trimmed BACK TO MAX_POSITION_PCT. Slots are sized to the cap by
    design (agents/position_sizer), so a trigger at the cap itself sold a
    slice of every winner a few percent after entry.

    Each result item: {"ticker": str, "shares_to_sell": int, "weight": float}
    shares_to_sell is >= 1 for well-formed Alpaca position data; callers should
    skip items where shares_to_sell < 1 as a precaution against floating-point edge cases.
    """
    orders = []
    max_pct = config.MAX_POSITION_PCT
    trigger_pct = max_pct * config.DRIFT_TRIM_HYSTERESIS
    for pos in positions:
        market_value = float(pos.get("market_value", 0))
        if market_value <= 0:
            continue
        ticker = pos.get("symbol", "").replace(".", "-")
        if not ticker:
            continue
        price = float(pos.get("current_price", 0))
        qty = int(float(pos.get("qty", 0)))
        if equity <= 0 or price <= 0:
            continue
        weight = market_value / equity
        if weight > trigger_pct:
            target_qty = math.floor(equity * max_pct / price)
            shares_to_sell = qty - target_qty
            orders.append({"ticker": ticker, "shares_to_sell": shares_to_sell, "weight": weight})
    return orders


def extract_weight_violating_tickers(violations: list) -> set:
    """Parse risk_report violations and return tickers with 'exceeds max position' strings."""
    result = set()
    for v in violations:
        if "exceeds max position" in v:
            result.add(v.split()[0])
    return result
