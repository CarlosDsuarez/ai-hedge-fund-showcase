"""Partial sells from the chat, and the fourth step nobody remembers.

close_partial_position releases every resting sell for the symbol — the OCO
included — and returns the order, not the fill. So a trim leaves the remainder
unprotected, and a position_state.json that still claims the pre-trim quantity
makes the next TP1 ask for shares that no longer exist (403 "insufficient qty
available"). "Trim ejecutado" must never imply "sigue protegido".
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chat.actions as actions
import db.database as db


def _tmp_db(tmp_path):
    return patch.object(db, "DB_PATH", str(tmp_path / "test.db"))


def _ctx(qty=100):
    return {
        "account": {"equity": 100_000.0, "buying_power": 150_000.0, "cash": 50_000.0},
        "live_positions": [{"symbol": "AAPL", "qty": qty, "current_price": 190.0,
                            "market_value": qty * 190.0, "unrealized_pl": 500.0,
                            "unrealized_plpc": 0.03}],
        "open_orders": [],
    }


def _trim(tmp_path, action, *, filled=(40, 190.0), protect=("oco-2", True, "ok"),
          reduce_ret="record"):
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.alpaca.get_open_orders", return_value=[]), \
         patch("execution.trade_executor.sell_and_confirm",
               return_value=filled) as sell, \
         patch("execution.order_manager.reduce_position_qty",
               return_value=({"qty_remaining": 60} if reduce_ret == "record" else None)) as reduce_, \
         patch("chat.actions._protect", return_value=protect) as protect_mock, \
         patch("builtins.input", return_value="s"):
        db.init_db()
        actions.handle_action(dict(action), _ctx())
        return sell, reduce_, protect_mock, db.load_chat_decisions(5)


PCT_TRIM = {"type": "sell_partial", "ticker": "AAPL", "qty": None, "pct": 0.4,
            "sl_price": None, "tp_price": None}


def test_a_percentage_becomes_a_share_count(tmp_path):
    sell, _, _, rows = _trim(tmp_path, PCT_TRIM)
    sell.assert_called_once_with("AAPL", 40, tag="chat")
    assert rows[0]["proposed_qty"] == 40


def test_a_percentage_over_one_is_read_as_a_percent(tmp_path):
    action = dict(PCT_TRIM, pct=30)
    sell, _, _, _ = _trim(tmp_path, action)
    sell.assert_called_once_with("AAPL", 30, tag="chat")


def test_the_trim_goes_through_sell_and_confirm_not_close_partial(tmp_path):
    """close_partial_position returns an order, never a fill."""
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.alpaca.get_open_orders", return_value=[]), \
         patch("execution.alpaca.close_partial_position") as close_partial, \
         patch("execution.trade_executor.sell_and_confirm", return_value=(40, 190.0)), \
         patch("execution.order_manager.reduce_position_qty", return_value={}), \
         patch("chat.actions._protect", return_value=(None, True, "ok")), \
         patch("builtins.input", return_value="s"):
        db.init_db()
        actions.handle_action(dict(PCT_TRIM), _ctx())
    close_partial.assert_not_called()


def test_the_local_ladder_is_shrunk(tmp_path):
    _, reduce_, _, _ = _trim(tmp_path, PCT_TRIM)
    reduce_.assert_called_once_with("AAPL", 40)


def test_the_remainder_is_re_protected(tmp_path):
    _, _, protect, rows = _trim(tmp_path, PCT_TRIM)
    protect.assert_called_once_with("AAPL")
    assert rows[0]["protected"] == 1
    assert rows[0]["oco_order_id"] == "oco-2"


def test_failed_re_protection_says_it_out_loud(tmp_path, capsys):
    _, _, _, rows = _trim(
        tmp_path, PCT_TRIM,
        protect=(None, False, f"⚠ no se pudo proteger — {actions._UNPROTECTED_WARNING}"))
    assert actions._UNPROTECTED_WARNING in capsys.readouterr().out
    assert rows[0]["protected"] == 0


def test_an_unfilled_trim_changes_no_local_state(tmp_path, capsys):
    _, reduce_, protect, rows = _trim(tmp_path, PCT_TRIM, filled=(0, 0.0))
    reduce_.assert_not_called()
    protect.assert_not_called()
    assert "NO ejecutada" in capsys.readouterr().out
    assert rows[0]["outcome"] == "NOT_EXECUTED"


def test_a_trim_that_closes_the_position_needs_no_protection(tmp_path):
    _, _, protect, rows = _trim(tmp_path, PCT_TRIM, filled=(100, 190.0),
                                reduce_ret=None)
    protect.assert_not_called()
    assert rows[0]["outcome"] == "EXECUTED"


def test_a_trim_is_refused_when_the_order_state_is_unknown(tmp_path, capsys):
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.alpaca.get_open_orders", return_value=None), \
         patch("execution.trade_executor.sell_and_confirm") as sell, \
         patch("builtins.input", side_effect=AssertionError("must not prompt")):
        db.init_db()
        actions.handle_action(dict(PCT_TRIM), _ctx())
        rows = db.load_chat_decisions(5)

    sell.assert_not_called()
    assert rows[0]["outcome"] == "BLOCKED"
    assert "órdenes abiertas" in capsys.readouterr().out
