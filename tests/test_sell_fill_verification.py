"""Sell-side fill verification (P0.2).

Buys already reconcile against the real fill (main.py `_execute_buy` calls
wait_for_fill). Sells did not: the drift trim marked a ticker trimmed the
instant submit_order returned. An unfilled trim therefore left a position
above MAX_POSITION_PCT while being recorded as rebalanced. (The weekly
rotation sell this file also covered is retired with design C; the exit
engine's own fill handling is pinned in tests/test_exit_engine.py.)
"""
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main as main_mod


def _order(order_id="o1", status="filled", filled_qty=10, price=100.0):
    return {"id": order_id, "status": status,
            "filled_qty": str(filled_qty), "filled_avg_price": str(price)}


# ── _sell_and_confirm ────────────────────────────────────────────────────────

def test_sell_and_confirm_returns_filled_quantity():
    with patch("execution.alpaca.release_reserved_shares"), \
         patch("execution.alpaca.submit_order", return_value=_order()), \
         patch("execution.alpaca.wait_for_fill", return_value=_order()):
        filled, price = main_mod._sell_and_confirm("AAPL", 10, tag="drift")
    assert filled == 10
    assert price == pytest.approx(100.0)


def test_sell_and_confirm_reports_zero_when_rejected():
    rejected = _order(status="rejected", filled_qty=0, price=0)
    with patch("execution.alpaca.release_reserved_shares"), \
         patch("execution.alpaca.submit_order", return_value=rejected), \
         patch("execution.alpaca.wait_for_fill", return_value=rejected):
        filled, _ = main_mod._sell_and_confirm("AAPL", 10, tag="drift")
    assert filled == 0, "a rejected sell must not be reported as filled"


def test_sell_and_confirm_reports_partial_fill():
    partial = _order(status="filled", filled_qty=4, price=99.0)
    with patch("execution.alpaca.release_reserved_shares"), \
         patch("execution.alpaca.submit_order", return_value=_order(filled_qty=0)), \
         patch("execution.alpaca.wait_for_fill", return_value=partial):
        filled, price = main_mod._sell_and_confirm("AAPL", 10, tag="drift")
    assert filled == 4
    assert price == pytest.approx(99.0)


def test_sell_and_confirm_returns_zero_when_submit_raises():
    with patch("execution.alpaca.release_reserved_shares"), \
         patch("execution.alpaca.submit_order", side_effect=RuntimeError("403")):
        filled, _ = main_mod._sell_and_confirm("AAPL", 10, tag="drift")
    assert filled == 0


# ── drift trim: only a real fill counts as trimmed ───────────────────────────

def _run_exits_with(trim_orders, sell_result, tmp_path, monkeypatch):
    """Drive run_exits past the trim block with everything else stubbed out."""
    positions = [{"symbol": "AAPL", "qty": "100", "market_value": "30000",
                  "current_price": "300", "unrealized_pl": "0"}]
    audit = {"protected": [], "placed": [], "replaced": [], "unprotected": [], "errors": []}
    # The position-state book matches the (mocked) broker book so
    # reconcile_fills' "missing from get_positions()" signal finds nothing.
    import execution.order_manager as order_manager
    order_manager.save_position_state({"AAPL": {"qty_remaining": 100, "entry_price": 300.0}})
    with patch("execution.alpaca.get_account", return_value={"equity": "100000"}), \
         patch("execution.alpaca.get_positions", return_value=positions), \
         patch("execution.alpaca.get_activities", return_value=[]), \
         patch("execution.order_manager.run_exit_engine", return_value=(set(), [])), \
         patch("execution.drift_rebalancer.compute_trim_orders", return_value=trim_orders), \
         patch("agents.risk_sl_tp_agent.run_stop_audit", return_value=audit), \
         patch.object(main_mod, "_sell_and_confirm", return_value=sell_result):
        return main_mod.run_exits()


def test_unfilled_trim_is_not_marked_trimmed(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "DRY_RUN", False)
    trim = [{"ticker": "AAPL", "shares_to_sell": 50, "weight": 0.30}]
    result = _run_exits_with(trim, sell_result=(0, 0.0), tmp_path=tmp_path, monkeypatch=monkeypatch)
    trimmed, failed = result[5], result[6]
    assert "AAPL" not in trimmed, "an unfilled trim must not clear the weight block"
    assert any(f["ticker"] == "AAPL" for f in failed)


def test_filled_trim_is_marked_trimmed(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "DRY_RUN", False)
    trim = [{"ticker": "AAPL", "shares_to_sell": 50, "weight": 0.30}]
    result = _run_exits_with(trim, sell_result=(50, 300.0), tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert "AAPL" in result[5]


def test_filled_trim_shrinks_the_engine_record(monkeypatch, tmp_path):
    """The ledger reports an untagged exit at qty_remaining; a trim that left
    the record at 100 shares would overstate the next such exit by 50."""
    import execution.order_manager as om
    monkeypatch.setattr(main_mod, "DRY_RUN", False)
    trim = [{"ticker": "AAPL", "shares_to_sell": 50, "weight": 0.30}]
    _run_exits_with(trim, sell_result=(50, 300.0), tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert om.load_position_state()["AAPL"]["qty_remaining"] == 50


def test_partial_trim_still_over_cap_is_not_marked_trimmed(monkeypatch, tmp_path):
    """Selling 5 of 50 shares leaves the position far above MAX_POSITION_PCT."""
    monkeypatch.setattr(main_mod, "DRY_RUN", False)
    trim = [{"ticker": "AAPL", "shares_to_sell": 50, "weight": 0.30}]
    result = _run_exits_with(trim, sell_result=(5, 300.0), tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert "AAPL" not in result[5], "a partial trim that stays over cap is not a trim"
