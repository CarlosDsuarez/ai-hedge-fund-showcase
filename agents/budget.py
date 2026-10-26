"""How many dollars this cycle may deploy.

The number used to be computed inline in run_entries as
`BUDGET_ALLOCATION * equity - committed`, which capped the book at a fixed
fraction of equity no matter how much liquidity the broker was offering.

The base is now configurable. With BUDGET_BASE = "buying_power" the run
deploys BUDGET_ALLOCATION of the broker's available buying power — real
leverage on a Reg-T margin account — and MAX_GROSS_EXPOSURE is the brake that
keeps long market value under a fixed multiple of equity.

Kept in its own module so the arithmetic is testable directly. The previous
budget test re-implemented the formula rather than importing it, so the numbers
it protected were its own, not production's.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def compute_deployment_budget(
    equity: float,
    buying_power: "float | None" = None,
    long_market_value: float = 0.0,
    *,
    base: "str | None" = None,
    allocation: "float | None" = None,
    max_gross: "float | None" = None,
) -> dict:
    """Dollars deployable this cycle, plus the reasoning behind the number.

    Returns {deployable, base, raw, gross_cap, degraded, reason}.

    `degraded` is True when the configured base was unusable and the equity
    fallback took over. That case is reported, never silent: an account read
    that fails and quietly halves the budget looks exactly like a cautious day.

    Note that with base="buying_power" the broker's figure is ALREADY net of
    open positions, so long_market_value is not subtracted from it — doing that
    would charge the existing book twice. It constrains the gross cap instead.
    """
    allocation = config.BUDGET_ALLOCATION if allocation is None else allocation
    base = (config.BUDGET_BASE if base is None else base).lower()
    max_gross = config.MAX_GROSS_EXPOSURE if max_gross is None else max_gross

    equity = max(float(equity or 0.0), 0.0)
    long_market_value = max(float(long_market_value or 0.0), 0.0)
    degraded, reason = False, ""

    if base == "buying_power":
        bp = float(buying_power or 0.0)
        if bp > 0:
            raw = allocation * bp
        else:
            raw = allocation * equity - long_market_value
            degraded = True
            reason = ("buying power unavailable or zero — fell back to "
                      f"{allocation:.0%} of equity")
    else:
        # Legacy semantics: the cap is a share of equity and the existing book
        # is charged against it.
        raw = allocation * equity - long_market_value

    gross_cap = max(max_gross * equity - long_market_value, 0.0)
    deployable = max(min(raw, gross_cap), 0.0)
    if raw > gross_cap and not degraded:
        reason = (f"gross exposure cap binding: {max_gross:.2f}x equity "
                  f"minus ${long_market_value:,.0f} already invested")

    return {
        "deployable": deployable,
        "base": base,
        "raw": max(raw, 0.0),
        "gross_cap": gross_cap,
        "degraded": degraded,
        "reason": reason,
    }


def deployable_fraction(deployable: float, equity: float) -> float:
    """Budget expressed as a multiple of equity — the unit position sizing uses.

    Above 1.0 when leverage is in play, which is the point: sizing that divides
    a fixed 0.75 across N candidates can never spend a leveraged budget.
    """
    if equity <= 0:
        return 0.0
    return deployable / equity
