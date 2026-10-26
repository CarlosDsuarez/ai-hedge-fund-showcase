# tests/test_chat_paper_guard.py — the chat must not reach a live account.
#
# chat/actions.py used to submit whatever the credentials pointed at, while
# main.py:1376 gated its entire execution phase on PAPER_TRADING. These tests
# pin the guard and the fill reconciliation that replaced the unconditional
# "✓ Ejecutado".
#
# Imports chat.actions, never chat.router: the router builds an
# anthropic.Anthropic client at import and would need an API key.
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chat.actions as actions
import db.database as db


def _tmp_db(tmp_path):
    return patch.object(db, "DB_PATH", str(tmp_path / "test.db"))


def _ctx():
    return {
        "account": {"equity": 100000.0, "buying_power": 150000.0, "cash": 50000.0},
        "live_positions": [{"symbol": "AAPL", "qty": 10, "current_price": 190.0,
                            "market_value": 1900.0, "unrealized_pl": 50.0,
                            "unrealized_plpc": 0.027}],
        "open_orders": [{"id": "ord-9", "symbol": "AAPL", "qty": 10,
                         "side": "sell", "status": "new", "type": "limit",
                         "order_class": "oco"}],
    }


BUY = {"type": "submit_buy", "ticker": "AAPL", "qty": 5,
       "sl_price": None, "tp_price": None}


# ── the guard ───────────────────────────────────────────────────────────────

def test_no_broker_call_when_paper_trading_is_false(tmp_path, capsys):
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "false"}), \
         patch("execution.alpaca.submit_order") as submit, \
         patch("execution.alpaca.close_position") as close, \
         patch("execution.alpaca.submit_oco_order") as oco, \
         patch("builtins.input", side_effect=AssertionError("must not prompt")):
        db.init_db()
        actions.handle_action(dict(BUY), _ctx())
        rows = db.load_chat_decisions(5)

    submit.assert_not_called()
    close.assert_not_called()
    oco.assert_not_called()
    assert "PAPER_TRADING" in capsys.readouterr().out
    assert rows[0]["outcome"] == "BLOCKED"
    assert rows[0]["paper_trading"] == 0


def test_no_broker_call_when_paper_trading_is_unset(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PAPER_TRADING"}
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, env, clear=True), \
         patch("execution.alpaca.submit_order") as submit:
        db.init_db()
        actions.handle_action(dict(BUY), _ctx())
        rows = db.load_chat_decisions(5)

    submit.assert_not_called()
    assert rows[0]["outcome"] == "BLOCKED"


def test_guard_reads_env_at_call_time(tmp_path):
    """Importing the module before .env was loaded must not defeat the guard."""
    with patch.dict(os.environ, {"PAPER_TRADING": "false"}):
        assert actions._paper_trading() is False
    with patch.dict(os.environ, {"PAPER_TRADING": "TRUE"}):
        assert actions._paper_trading() is True
    with patch.dict(os.environ, {"PAPER_TRADING": " true "}):
        assert actions._paper_trading() is True


# ── confirmation ────────────────────────────────────────────────────────────

def test_declining_executes_nothing_and_is_journalled(tmp_path):
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.alpaca.submit_order") as submit, \
         patch("builtins.input", return_value="n"):
        db.init_db()
        actions.handle_action(dict(BUY), _ctx())
        rows = db.load_chat_decisions(5)

    submit.assert_not_called()
    assert rows[0]["user_choice"] == "rejected"
    assert rows[0]["outcome"] == "NOT_EXECUTED"


def test_eof_on_the_prompt_cancels(tmp_path):
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.alpaca.submit_order") as submit, \
         patch("builtins.input", side_effect=EOFError):
        db.init_db()
        actions.handle_action(dict(BUY), _ctx())
        rows = db.load_chat_decisions(5)

    submit.assert_not_called()
    assert rows[0]["user_choice"] == "rejected"


# ── fill reconciliation ─────────────────────────────────────────────────────
#
# Every broker call is patched: nothing in this file may reach the network.
# resolve_price and _protect are patched too — the buy path now fetches a quote
# and places the OCO in the same turn. get_atr as well: register_levels pulls
# 3mo of bars to set the initial stop, which is a price fetch, not a broker call.

def _run_confirmed_buy(tmp_path, order, filled, *, protect=("oco-1", True, "ok")):
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.trade_executor.resolve_price", return_value=190.0), \
         patch("data.fetcher.get_atr", return_value=3.5), \
         patch("execution.alpaca.submit_order", return_value=order), \
         patch("execution.alpaca.wait_for_fill", return_value=filled), \
         patch("chat.actions._protect", return_value=protect), \
         patch("builtins.input", return_value="s"):
        db.init_db()
        actions.handle_action(dict(BUY), _ctx())
        return db.load_chat_decisions(5)


def test_filled_order_reports_the_real_fill(tmp_path, capsys):
    rows = _run_confirmed_buy(
        tmp_path,
        {"id": "ord-1", "status": "accepted", "qty": "5"},
        {"id": "ord-1", "status": "filled", "qty": "5",
         "filled_qty": "5", "filled_avg_price": "190.25"},
    )
    assert "Ejecutado" in capsys.readouterr().out
    assert rows[0]["outcome"] == "EXECUTED"
    assert rows[0]["filled_qty"] == 5
    assert rows[0]["fill_price"] == 190.25
    assert rows[0]["oco_order_id"] == "oco-1"
    assert rows[0]["protected"] == 1


def test_rejected_order_is_not_reported_as_executed(tmp_path, capsys):
    rows = _run_confirmed_buy(
        tmp_path,
        {"id": "ord-1", "status": "accepted", "qty": "5"},
        {"id": "ord-1", "status": "rejected", "qty": "5", "filled_qty": "0"},
    )
    out = capsys.readouterr().out
    assert "NO ejecutada" in out
    assert "✓ Ejecutado" not in out
    assert rows[0]["outcome"] == "NOT_EXECUTED"


def test_unfilled_order_places_no_oco_and_claims_nothing(tmp_path, capsys):
    """An assumed fill would register a position and protect shares that may not exist."""
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.trade_executor.resolve_price", return_value=190.0), \
         patch("execution.alpaca.submit_order",
               return_value={"id": "ord-1", "status": "accepted", "qty": "5"}), \
         patch("execution.alpaca.wait_for_fill",
               return_value={"id": "ord-1", "status": "new", "qty": "5",
                             "filled_qty": "0"}), \
         patch("chat.actions._protect") as protect, \
         patch("builtins.input", return_value="s"):
        db.init_db()
        actions.handle_action(dict(BUY), _ctx())
        rows = db.load_chat_decisions(5)

    out = capsys.readouterr().out
    assert "SIN fill confirmado" in out
    assert "✓ Ejecutado" not in out
    protect.assert_not_called()
    assert rows[0]["outcome"] == "PENDING"
    assert rows[0]["filled_qty"] == 0


def test_partial_fill_is_reported_as_partial(tmp_path, capsys):
    rows = _run_confirmed_buy(
        tmp_path,
        {"id": "ord-1", "status": "accepted", "qty": "5"},
        {"id": "ord-1", "status": "filled", "qty": "5",
         "filled_qty": "2", "filled_avg_price": "190.00"},
    )
    assert "PARCIAL" in capsys.readouterr().out
    assert rows[0]["outcome"] == "PARTIAL"
    assert rows[0]["filled_qty"] == 2


def test_failed_protection_says_so_out_loud(tmp_path, capsys):
    """"Comprado" must never imply "protegido"."""
    rows = _run_confirmed_buy(
        tmp_path,
        {"id": "ord-1", "status": "accepted", "qty": "5"},
        {"id": "ord-1", "status": "filled", "qty": "5",
         "filled_qty": "5", "filled_avg_price": "190.25"},
        protect=(None, False, "⚠ no se pudo proteger — sin protección"),
    )
    assert "no se pudo proteger" in capsys.readouterr().out
    assert rows[0]["protected"] == 0


def test_broker_error_is_journalled(tmp_path, capsys):
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.trade_executor.resolve_price", return_value=190.0), \
         patch("execution.alpaca.submit_order", side_effect=RuntimeError("403")), \
         patch("builtins.input", return_value="s"):
        db.init_db()
        actions.handle_action(dict(BUY), _ctx())
        rows = db.load_chat_decisions(5)

    assert "Error al ejecutar" in capsys.readouterr().out
    assert rows[0]["outcome"] == "ERROR"
    assert "RuntimeError" in rows[0]["error"]


def test_oco_is_submitted_not_awaited(tmp_path):
    """An OCO rests at the broker — there is no fill of its own to wait for."""
    action = {"type": "submit_oco", "ticker": "AAPL", "qty": 10,
              "sl_price": 180.0, "tp_price": 210.0}
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.alpaca.get_open_orders", return_value=[]), \
         patch("execution.alpaca.submit_oco_order",
               return_value={"id": "oco-1", "status": "new"}) as oco, \
         patch("execution.alpaca.wait_for_fill") as wait, \
         patch("builtins.input", return_value="s"):
        db.init_db()
        actions.handle_action(action, _ctx())
        rows = db.load_chat_decisions(5)

    oco.assert_called_once_with("AAPL", 10, 180.0, 210.0, tag="chatoco")
    wait.assert_not_called()
    assert rows[0]["outcome"] == "EXECUTED"
    assert rows[0]["oco_order_id"] == "oco-1"


def test_selling_is_refused_when_order_state_is_unknown(tmp_path, capsys):
    """get_open_orders() returning None means unknown, and selling blind risks a 403."""
    action = {"type": "close_position", "ticker": "AAPL", "qty": None,
              "sl_price": None, "tp_price": None}
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.alpaca.get_open_orders", return_value=None), \
         patch("execution.alpaca.close_position") as close, \
         patch("builtins.input", side_effect=AssertionError("must not prompt")):
        db.init_db()
        actions.handle_action(action, _ctx())
        rows = db.load_chat_decisions(5)

    close.assert_not_called()
    assert "órdenes abiertas" in capsys.readouterr().out
    assert rows[0]["outcome"] == "BLOCKED"
