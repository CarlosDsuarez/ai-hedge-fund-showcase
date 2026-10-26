# tests/test_main_sizing.py — the deployment budget.
#
# This file used to re-implement main.py's equal-weight sizing formula
# (deployable_frac / N) and assert against the copy, so it protected its own
# arithmetic rather than production's. Sprint 1 design B replaced that
# equal-weight sizing with slot-based sizing — see agents/position_sizer.py
# and tests/test_position_sizer.py, which pin plan_entries against production
# directly. Only the deployment-budget arithmetic (unchanged by design B)
# still lives here.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.budget import compute_deployment_budget, deployable_fraction


# ── deployment budget ───────────────────────────────────────────────────

def test_budget_deploys_a_share_of_buying_power():
    r = compute_deployment_budget(equity=100_000, buying_power=200_000,
                                  long_market_value=0, base="buying_power",
                                  allocation=0.75, max_gross=1.5)
    assert r["deployable"] == 150_000        # 0.75 * 200k, under the 1.5x cap
    assert r["degraded"] is False


def test_gross_exposure_cap_binds_before_buying_power():
    """The brake on the leverage. 1.5x equity minus what is already invested."""
    r = compute_deployment_budget(equity=100_000, buying_power=200_000,
                                  long_market_value=60_000, base="buying_power",
                                  allocation=0.75, max_gross=1.5)
    assert r["deployable"] == 90_000         # 1.5*100k - 60k, not 0.75*200k
    assert "gross exposure cap" in r["reason"]


def test_existing_book_is_not_charged_twice():
    """Alpaca's buying power is already net of open positions.

    Subtracting long market value from it as well — the shape of the old
    equity-based formula — would under-deploy by exactly the size of the book.
    """
    r = compute_deployment_budget(equity=1_000_000, buying_power=200_000,
                                  long_market_value=60_000, base="buying_power",
                                  allocation=0.75, max_gross=1.5)
    assert r["deployable"] == 150_000        # gross cap is far away here


def test_missing_buying_power_falls_back_and_says_so():
    """A failed account read must not quietly halve the budget."""
    r = compute_deployment_budget(equity=100_000, buying_power=0,
                                  long_market_value=20_000, base="buying_power",
                                  allocation=0.75, max_gross=1.5)
    assert r["deployable"] == 55_000         # 0.75*100k - 20k, legacy shape
    assert r["degraded"] is True
    assert "buying power" in r["reason"]


def test_equity_base_preserves_the_old_behaviour():
    r = compute_deployment_budget(equity=100_000, buying_power=200_000,
                                  long_market_value=20_000, base="equity",
                                  allocation=0.75, max_gross=1.5)
    assert r["deployable"] == 55_000
    assert r["degraded"] is False


def test_budget_never_goes_negative():
    r = compute_deployment_budget(equity=100_000, buying_power=0,
                                  long_market_value=400_000, base="buying_power",
                                  allocation=0.75, max_gross=1.5)
    assert r["deployable"] == 0.0


def test_deployable_fraction_expresses_leverage():
    assert deployable_fraction(150_000, 100_000) == 1.5
    assert deployable_fraction(0, 0) == 0.0
