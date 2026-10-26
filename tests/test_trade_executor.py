"""execution/trade_executor.py — the broker primitives shared by main.py and the chat.

The split exists because chat/ cannot import main.py: main.py calls
get_account()+get_positions() at import time (main.py:78). These tests pin the
one deliberate behavioural difference between the two callers — what happens
when the fill poll times out.
"""
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.trade_executor import buy_and_register, sell_and_confirm


def _order(order_id="o1", status="filled", qty=5, filled_qty=5, price=100.0):
    return {"id": order_id, "status": status, "qty": str(qty),
            "filled_qty": str(filled_qty), "filled_avg_price": str(price)}


def _buy(order, *, filled=None, **kw):
    filled = order if filled is None else filled
    with patch("execution.alpaca.submit_order", return_value=order), \
         patch("execution.alpaca.wait_for_fill", return_value=filled):
        return buy_and_register("AAPL", 500.0, price=100.0, register=False, **kw)


# ── the fill-timeout split ──────────────────────────────────────────────────

def test_timeout_without_assumption_registers_nothing():
    """The chat's contract: no fill seen, no position claimed."""
    pending = _order(status="new", filled_qty=0, price=0)
    with patch("execution.alpaca.submit_order", return_value=pending), \
         patch("execution.alpaca.wait_for_fill", return_value=pending), \
         patch("execution.trade_executor.register_levels") as levels:
        result = buy_and_register("AAPL", 500.0, price=100.0, atr_14=2.0)

    assert result.status == "PENDING"
    assert result.filled_qty == 0
    assert result.spent == 0.0
    levels.assert_not_called()


def test_timeout_with_assumption_accounts_the_requested_qty():
    """The pipeline's contract: assume the fill or the remaining budget drifts."""
    pending = _order(status="new", filled_qty=0, price=0)
    result = _buy(pending, assume_fill_on_timeout=True)

    assert result.status == "FILLED"
    assert result.assumed_fill is True
    assert result.filled_qty == 5
    assert result.fill_price == pytest.approx(100.0)


# ── reconciliation ──────────────────────────────────────────────────────────

def test_full_fill():
    result = _buy(_order())
    assert (result.status, result.filled_qty) == ("FILLED", 5)
    assert result.spent == pytest.approx(500.0)


def test_partial_fill():
    result = _buy(_order(), filled=_order(filled_qty=2, price=99.0))
    assert result.status == "PARTIAL"
    assert result.filled_qty == 2
    assert result.spent == pytest.approx(198.0)


def test_rejected_order():
    rejected = _order(status="rejected", filled_qty=0, price=0)
    result = _buy(rejected)
    assert result.status == "REJECTED"
    assert result.reason_code == "ORDER_REJECTED"


def test_no_price_is_skipped_not_ordered():
    with patch("execution.alpaca.submit_order") as submit:
        result = buy_and_register("AAPL", 500.0, price=0.0)
    submit.assert_not_called()
    assert result.reason_code == "NO_PRICE_DATA"


def test_spend_below_one_share_is_skipped():
    with patch("execution.alpaca.submit_order") as submit:
        result = buy_and_register("AAPL", 50.0, price=100.0)
    submit.assert_not_called()
    assert result.reason_code == "QTY_LT_1"


def test_broker_error_becomes_a_result_not_a_raise():
    with patch("execution.alpaca.submit_order", side_effect=RuntimeError("403")):
        result = buy_and_register("AAPL", 500.0, price=100.0)
    assert result.status == "ERROR"
    assert result.reason_code == "ORDER_ERROR"
    assert "RuntimeError" in result.detail


# ── levels: local state before broker protection ────────────────────────────

def test_levels_are_registered_from_the_real_fill_not_the_request():
    with patch("execution.alpaca.submit_order", return_value=_order()), \
         patch("execution.alpaca.wait_for_fill",
               return_value=_order(filled_qty=2, price=99.0)), \
         patch("execution.order_manager.register_new_position") as reg:
        buy_and_register("AAPL", 500.0, price=100.0, atr_14=2.0)

    # No stop multiplier travels with the call: it is config.STOP_ATR_MULT,
    # never an argument and never the retired ATR_MULTIPLIER_SL env var.
    reg.assert_called_once_with("AAPL", 99.0, 2, 2.0)


def test_missing_atr_falls_back_to_the_live_atr():
    """A research row without atr_14 (a failed chunk) must not leave the fill
    without a record and a broker stop until tomorrow: get_atr(14, "3mo") —
    the same feed the engine registers on the fly with — fills the gap."""
    with patch("execution.alpaca.submit_order", return_value=_order()), \
         patch("execution.alpaca.wait_for_fill", return_value=_order()), \
         patch("data.fetcher.get_atr", return_value=3.25) as atr, \
         patch("execution.order_manager.register_new_position",
               return_value={"initial_stop": 91.88}) as reg:
        result = buy_and_register("AAPL", 500.0, price=100.0, atr_14=None)

    atr.assert_called_once_with("AAPL", 14, "3mo")
    reg.assert_called_once_with("AAPL", 100.0, 5, 3.25)
    assert result.levels == {"initial_stop": 91.88}


def test_no_levels_when_even_the_live_atr_is_unavailable():
    with patch("execution.alpaca.submit_order", return_value=_order()), \
         patch("execution.alpaca.wait_for_fill", return_value=_order()), \
         patch("data.fetcher.get_atr", return_value=0.0), \
         patch("execution.order_manager.register_new_position") as reg:
        result = buy_and_register("AAPL", 500.0, price=100.0, atr_14=None)

    reg.assert_not_called()
    assert result.levels is None
    assert result.status == "FILLED", "a missing ATR must not undo the fill"


def test_a_provided_atr_is_used_without_refetching():
    with patch("execution.alpaca.submit_order", return_value=_order()), \
         patch("execution.alpaca.wait_for_fill", return_value=_order()), \
         patch("data.fetcher.get_atr", side_effect=AssertionError("must not refetch")), \
         patch("execution.order_manager.register_new_position", return_value={}) as reg:
        buy_and_register("AAPL", 500.0, price=100.0, atr_14=2.0)
    reg.assert_called_once_with("AAPL", 100.0, 5, 2.0)


def test_a_failed_registration_keeps_the_fill():
    """The shares are owned either way; the next engine pass registers on the fly."""
    with patch("execution.alpaca.submit_order", return_value=_order()), \
         patch("execution.alpaca.wait_for_fill", return_value=_order()), \
         patch("execution.order_manager.register_new_position",
               side_effect=RuntimeError("db locked")):
        result = buy_and_register("AAPL", 500.0, price=100.0, atr_14=2.0)

    assert result.status == "FILLED"
    assert result.levels is None


# ── sell_and_confirm moved verbatim; main.py re-exports it ──────────────────

def test_sell_and_confirm_releases_reservations_first():
    calls = []
    with patch("execution.alpaca.release_reserved_shares",
               side_effect=lambda t: calls.append("release")), \
         patch("execution.alpaca.submit_order",
               side_effect=lambda *a, **k: calls.append("submit") or _order()), \
         patch("execution.alpaca.wait_for_fill", return_value=_order(filled_qty=5)):
        filled, price = sell_and_confirm("AAPL", 5, tag="chat")

    assert calls == ["release", "submit"]
    assert filled == 5


def _forbidden(text):
    import requests
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.status_code = 403
    resp.text = text
    return requests.HTTPError(f"403 Forbidden — {text}", response=resp)


def test_sell_waits_for_the_cancelled_stop_to_die_before_selling():
    """Every position now carries a GTC stop. release_reserved_shares only
    FIRES the cancel; selling before it lands is a 403 "insufficient qty"
    race that used to read as an unfilled sell — position kept, and the audit
    then refusing to re-place (stop at market): a stopped-out name with
    neither a software exit nor a broker stop until tomorrow."""
    calls = []
    with patch("execution.alpaca.release_reserved_shares",
               side_effect=lambda t: calls.append("release") or ["stop-1"]), \
         patch("execution.alpaca.wait_until_dead",
               side_effect=lambda ids, **kw: calls.append(("wait", tuple(ids))) or
               {"dead": True, "filled": [], "pending": [], "errors": []}), \
         patch("execution.alpaca.submit_order",
               side_effect=lambda *a, **k: calls.append("submit") or _order()), \
         patch("execution.alpaca.wait_for_fill", return_value=_order(filled_qty=5)):
        filled, price = sell_and_confirm("AAPL", 5, tag="stop_trail")

    assert calls == ["release", ("wait", ("stop-1",)), "submit"]
    assert filled == 5


def test_sell_does_not_wait_when_nothing_was_resting():
    with patch("execution.alpaca.release_reserved_shares", return_value=[]), \
         patch("execution.alpaca.wait_until_dead") as wait, \
         patch("execution.alpaca.submit_order", return_value=_order()), \
         patch("execution.alpaca.wait_for_fill", return_value=_order(filled_qty=5)):
        filled, _ = sell_and_confirm("AAPL", 5, tag="drift")
    wait.assert_not_called()
    assert filled == 5


def test_sell_retries_once_after_a_403_insufficient_qty():
    """The cancel reported dead but the broker still had the shares locked
    for a beat: one retry after another wait, then give up honestly."""
    submits = []

    def _submit(*a, **k):
        submits.append(1)
        if len(submits) == 1:
            raise _forbidden("insufficient qty available for order (requested: 5, available: 0)")
        return _order()

    with patch("execution.alpaca.release_reserved_shares", return_value=["stop-1"]), \
         patch("execution.alpaca.wait_until_dead",
               return_value={"dead": True, "filled": [], "pending": [], "errors": []}) as wait, \
         patch("execution.alpaca.submit_order", side_effect=_submit), \
         patch("execution.alpaca.wait_for_fill", return_value=_order(filled_qty=5)):
        filled, price = sell_and_confirm("AAPL", 5, tag="stop_initial")

    assert len(submits) == 2
    assert wait.call_count == 2, "a second wait precedes the single retry"
    assert filled == 5


def test_sell_gives_up_after_the_single_retry():
    def _submit(*a, **k):
        raise _forbidden("insufficient qty available for order (requested: 5, available: 0)")

    with patch("execution.alpaca.release_reserved_shares", return_value=["stop-1"]), \
         patch("execution.alpaca.wait_until_dead",
               return_value={"dead": True, "filled": [], "pending": [], "errors": []}), \
         patch("execution.alpaca.submit_order", side_effect=_submit) as submit:
        filled, _ = sell_and_confirm("AAPL", 5, tag="stop_initial")
    assert submit.call_count == 2
    assert filled == 0


def test_sell_is_abandoned_when_the_stop_filled_during_the_cancel():
    """The broker sold the position through its own stop while we were
    cancelling it: a market sell now would open a short. The ledger picks the
    broker fill up as BROKER_STOP on the next reconciliation."""
    with patch("execution.alpaca.release_reserved_shares", return_value=["stop-1"]), \
         patch("execution.alpaca.wait_until_dead",
               return_value={"dead": True, "filled": ["stop-1"], "pending": [], "errors": []}), \
         patch("execution.alpaca.submit_order") as submit:
        filled, _ = sell_and_confirm("AAPL", 5, tag="stop_trail")
    submit.assert_not_called()
    assert filled == 0


def test_main_still_exposes_sell_and_confirm():
    import main as main_mod

    with patch("execution.alpaca.release_reserved_shares"), \
         patch("execution.alpaca.submit_order", return_value=_order()), \
         patch("execution.alpaca.wait_for_fill", return_value=_order()):
        assert main_mod._sell_and_confirm("AAPL", 5, tag="drift") == (5, 100.0)
