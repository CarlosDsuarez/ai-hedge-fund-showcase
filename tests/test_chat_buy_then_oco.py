"""Buy from the chat: local state first, broker protection second — always.

The two failure modes are asymmetric. Local record written but the broker stop
failing self-heals: the next run's audit sees no stop and places one. The stop
placed but the local write failing leaves the position to be registered on the
fly by the next engine pass off avg_entry_price — a stop level the user was
never shown.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chat.actions as actions
import db.database as db


def _tmp_db(tmp_path):
    return patch.object(db, "DB_PATH", str(tmp_path / "test.db"))


def _ctx():
    return {"account": {"equity": 100_000.0, "buying_power": 150_000.0},
            "live_positions": [], "open_orders": []}


BUY = {"type": "submit_buy", "ticker": "AAPL", "qty": 5,
       "sl_price": None, "tp_price": None}

_ACCEPTED = {"id": "ord-1", "status": "accepted", "qty": "5"}
_FILLED = {"id": "ord-1", "status": "filled", "qty": "5",
           "filled_qty": "5", "filled_avg_price": "190.0"}


def test_local_levels_are_written_before_the_broker_stop(tmp_path):
    calls = []
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.trade_executor.resolve_price", return_value=190.0), \
         patch("execution.alpaca.submit_order", return_value=_ACCEPTED), \
         patch("execution.alpaca.wait_for_fill", return_value=_FILLED), \
         patch("execution.order_manager.register_new_position",
               side_effect=lambda *a, **k: calls.append("local") or {}), \
         patch("chat.actions._protect",
               side_effect=lambda t: calls.append("broker") or (None, True, "ok")), \
         patch("builtins.input", return_value="s"):
        db.init_db()
        action = dict(BUY)
        action["_proposal"] = _FakeProposal(atr_14=4.0, price=190.0)
        actions.handle_action(action, _ctx())

    assert calls == ["local", "broker"]


class _FakeProposal:
    """Only the attributes chat/actions.py reads off a proposal."""
    def __init__(self, atr_14=0.0, price=0.0):
        self.atr_14 = atr_14
        self.conviction = 4
        self.composite = 0.8
        self.regime = "BULL"
        self.ref_price = price
        self.chat_gates = []
        self.entry_gates = []
        self.size = type("S", (), {
            "price": price, "alloc_pct": 0.14, "n": 10, "n_floor": 10,
            "peer_count": 9, "deployable_frac": 1.4, "binding": "target",
            "caps": [],
        })()


def _run(tmp_path, filled, *, atr=4.0):
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.trade_executor.resolve_price", return_value=190.0), \
         patch("execution.alpaca.submit_order", return_value=_ACCEPTED), \
         patch("execution.alpaca.wait_for_fill", return_value=filled), \
         patch("execution.order_manager.register_new_position",
               return_value={}) as local, \
         patch("chat.actions._protect",
               return_value=("oco-1", True, "ok")) as broker, \
         patch("builtins.input", return_value="s"):
        db.init_db()
        action = dict(BUY)
        action["_proposal"] = _FakeProposal(atr_14=atr, price=190.0)
        actions.handle_action(action, _ctx())
        return local, broker, db.load_chat_decisions(5)


def test_an_unfilled_order_registers_nothing_and_protects_nothing(tmp_path):
    pending = {"id": "ord-1", "status": "new", "qty": "5", "filled_qty": "0"}
    local, broker, rows = _run(tmp_path, pending)
    local.assert_not_called()
    broker.assert_not_called()
    assert rows[0]["outcome"] == "PENDING"


def test_a_rejected_order_registers_nothing_and_protects_nothing(tmp_path):
    rejected = {"id": "ord-1", "status": "rejected", "qty": "5", "filled_qty": "0"}
    local, broker, rows = _run(tmp_path, rejected)
    local.assert_not_called()
    broker.assert_not_called()
    assert rows[0]["outcome"] == "NOT_EXECUTED"


def test_levels_use_the_filled_quantity_not_the_requested_one(tmp_path):
    partial = {"id": "ord-1", "status": "filled", "qty": "5",
               "filled_qty": "2", "filled_avg_price": "189.0"}
    local, _, rows = _run(tmp_path, partial)
    local.assert_called_once_with("AAPL", 189.0, 2, 4.0)
    assert rows[0]["outcome"] == "PARTIAL"


def test_a_missing_atr_still_protects_via_the_audit(tmp_path):
    """No ATR from the proposal and none from the live feed either: no local
    record from the fill, but _protect still runs (and registers on the fly
    once an ATR is available)."""
    with patch("data.fetcher.get_atr", return_value=0.0):
        local, broker, rows = _run(tmp_path, _FILLED, atr=0.0)
    local.assert_not_called()
    broker.assert_called_once_with("AAPL")
    assert rows[0]["outcome"] == "EXECUTED"


def test_a_missing_proposal_atr_uses_the_live_atr_for_the_record(tmp_path):
    with patch("data.fetcher.get_atr", return_value=3.0):
        local, broker, rows = _run(tmp_path, _FILLED, atr=0.0)
    local.assert_called_once_with("AAPL", 190.0, 5, 3.0)
    assert rows[0]["outcome"] == "EXECUTED"


def test_position_metadata_is_written_so_holding_days_can_accrue(tmp_path):
    """Without it main.py sees the ticker held but untracked, and the exit
    engine has no entry date to fall back on."""
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.trade_executor.resolve_price", return_value=190.0), \
         patch("execution.alpaca.submit_order", return_value=_ACCEPTED), \
         patch("execution.alpaca.wait_for_fill", return_value=_FILLED), \
         patch("execution.order_manager.register_new_position", return_value={}), \
         patch("chat.actions._protect", return_value=(None, True, "ok")), \
         patch("builtins.input", return_value="s"):
        db.init_db()
        action = dict(BUY)
        action["_proposal"] = _FakeProposal(atr_14=4.0, price=190.0)
        actions.handle_action(action, _ctx())
        meta = db.load_position_metadata()

    assert meta["AAPL"]["entry_price"] == 190.0
    assert meta["AAPL"]["holding_days"] == 0
    assert meta["AAPL"]["conviction"] == 4


# ── _protect itself ─────────────────────────────────────────────────────────

def _positions(qty=5, qty_available=None):
    """qty_available defaults to qty (nothing resting); a test that puts a
    resting sell in open_orders must pass the Alpaca-consistent 0."""
    return [{"symbol": "AAPL", "qty": str(qty),
             "qty_available": str(qty if qty_available is None else qty_available),
             "current_price": "190.0", "market_value": str(qty * 190.0),
             "unrealized_pl": "0.0"}]


def _record(trail=180.0):
    return {"AAPL": {"entry_price": 190.0, "atr_14": 4.0, "initial_stop": 180.0,
                     "trail_stop": trail, "highest_close": 190.0,
                     "qty_total": 5, "qty_remaining": 5, "entry_time": "2026-09-01T10:00:00"}}


def test_protect_reports_a_position_not_yet_visible_at_the_broker(tmp_path):
    with patch("execution.alpaca.get_positions", return_value=[]):
        stop_id, protected, note = actions._protect("AAPL")
    assert (stop_id, protected) == (None, False)
    assert actions._UNPROTECTED_WARNING in note


def test_protect_fails_closed_when_the_order_state_is_unknown(tmp_path):
    import execution.order_manager as om
    om.save_position_state(_record())
    with patch("execution.alpaca.get_positions", return_value=_positions()), \
         patch("agents.risk_sl_tp_agent.get_open_orders", return_value=None), \
         patch("agents.risk_sl_tp_agent.submit_stop_order") as submit:
        stop_id, protected, note = actions._protect("AAPL")
    submit.assert_not_called()
    assert protected is False
    assert actions._UNPROTECTED_WARNING in note


def test_protect_places_the_engines_trail_as_a_gtc_stop(tmp_path):
    import execution.order_manager as om
    om.save_position_state(_record(trail=182.5))
    with patch("execution.alpaca.get_positions", return_value=_positions()), \
         patch("agents.risk_sl_tp_agent.get_open_orders", return_value=[]), \
         patch("agents.risk_sl_tp_agent.submit_stop_order",
               return_value={"id": "stop-1", "status": "accepted"}) as submit:
        stop_id, protected, note = actions._protect("AAPL")
    submit.assert_called_once_with("AAPL", 5, 182.5)
    assert (stop_id, protected) == ("stop-1", True)
    assert "182.50" in note
    assert om.load_position_state()["AAPL"]["broker_stop_order_id"] == "stop-1"


def test_protect_leaves_a_matching_stop_alone(tmp_path):
    import execution.order_manager as om
    om.save_position_state(_record(trail=182.5))
    resting = [{"id": "stop-1", "symbol": "AAPL", "side": "sell", "type": "stop",
                "stop_price": "182.50", "qty": "5", "order_class": "simple", "status": "new"}]
    with patch("execution.alpaca.get_positions", return_value=_positions(qty_available=0)), \
         patch("agents.risk_sl_tp_agent.get_open_orders", return_value=resting), \
         patch("agents.risk_sl_tp_agent.submit_stop_order") as submit, \
         patch("agents.risk_sl_tp_agent.replace_order") as replace:
        stop_id, protected, note = actions._protect("AAPL")
    submit.assert_not_called()
    replace.assert_not_called()
    assert protected is True
    assert "ya tiene stop" in note


