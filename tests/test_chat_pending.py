"""The pipeline's pending queue, surfaced in the chat.

save_pending_signals holds BUYs the pipeline approved but could not fund
(main.py's classify_buy_outcome -> "QUEUE"). Until now nothing but the next
scheduled run could see them.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chat.executor as executor
import db.database as db


def _tmp_db(tmp_path):
    return patch.object(db, "DB_PATH", str(tmp_path / "test.db"))


def _ctx():
    return {"account": {"equity": 100_000.0, "buying_power": 150_000.0, "cash": 0.0},
            "live_positions": [], "open_orders": [], "portfolio_state": {}}


def _routing(agents, **over):
    base = {"agents": agents, "tickers": [], "response_language": "es",
            "action": {"type": None, "ticker": None, "qty": None, "pct": None,
                       "sl_price": None, "tp_price": None}}
    base.update(over)
    return base


def test_the_queue_is_listed_with_score_and_conviction(tmp_path):
    with _tmp_db(tmp_path):
        db.init_db()
        db.save_pending_signals("2026-09-04T09:00:00", [
            {"ticker": "AAPL", "composite_score": 0.55, "conviction": 4,
             "pm_action": "BUY", "size_pct": 12.0,
             "reason_deferred": "insufficient_budget"},
            {"ticker": "MSFT", "composite_score": 0.41, "conviction": 3,
             "pm_action": "BUY", "size_pct": 10.0,
             "reason_deferred": "insufficient_budget"},
        ])
        text, action = executor.execute_routing(_routing(["list_pending"]), _ctx())

    assert action is None, "listing the queue must never propose an order by itself"
    assert "AAPL" in text and "MSFT" in text
    assert "conv=4/5" in text
    assert "+0.5500" in text


def test_an_empty_queue_says_so(tmp_path):
    with _tmp_db(tmp_path):
        db.init_db()
        text, action = executor.execute_routing(_routing(["list_pending"]), _ctx())
    assert "No hay señales pendientes" in text
    assert action is None


def test_executed_entries_leave_the_queue(tmp_path):
    with _tmp_db(tmp_path):
        db.init_db()
        db.save_pending_signals("2026-09-04T09:00:00", [
            {"ticker": "AAPL", "composite_score": 0.55, "conviction": 4,
             "pm_action": "BUY", "size_pct": 12.0, "reason_deferred": "x"},
        ])
        db.mark_pending_executed("AAPL")
        text, _ = executor.execute_routing(_routing(["list_pending"]), _ctx())
    assert "No hay señales pendientes" in text


# ── propose_trade routes through the sizing engine, never the LLM's guess ───

def test_propose_trade_without_a_ticker_asks_for_one(tmp_path):
    with _tmp_db(tmp_path):
        db.init_db()
        text, action = executor.execute_routing(_routing(["propose_trade"]), _ctx())
    assert "ticker" in text.lower()
    assert action is None


def test_propose_trade_attaches_the_proposal_to_the_action(tmp_path):
    """chat/actions.py needs it to journal the sizing and the gate verdicts."""
    fake = MagicMock()
    fake.to_action.return_value = {"type": "submit_buy", "ticker": "NVDA", "qty": 70,
                                   "sl_price": 194.0, "tp_price": 212.0}
    with _tmp_db(tmp_path), \
         patch("chat.proposal.build_proposal", return_value=fake), \
         patch("chat.proposal.render", return_value="PANEL"):
        db.init_db()
        text, action = executor.execute_routing(
            _routing(["propose_trade"], tickers=["NVDA"]), _ctx())

    assert "PANEL" in text
    assert action["ticker"] == "NVDA"
    assert action["qty"] == 70
    assert action["_proposal"] is fake
