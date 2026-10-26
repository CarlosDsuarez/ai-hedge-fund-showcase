# tests/test_db_and_router.py — db round-trip/migración + router malformed (auditoría Fase B, prompt 14)
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db.database as db


def _tmp_db(tmp_path):
    return patch.object(db, "DB_PATH", str(tmp_path / "test.db"))


# ── db: save_run round-trip ─────────────────────────────────────────────────

def test_save_run_roundtrip(tmp_path):
    run = {
        "timestamp": "2026-07-05T09:00:00",
        "paper_trading": True,
        "regime": {"regime": "BULL", "vol_regime": "LOW", "vix": 14.2},
        "orders_submitted": [{"ticker": "AAPL", "qty": 3}],
        "closed_positions": [],
    }
    with _tmp_db(tmp_path):
        db.init_db()
        db.save_run(run)
        loaded = db.load_recent_runs(5)

    assert len(loaded) == 1
    assert loaded[0]["timestamp"] == run["timestamp"]
    assert loaded[0]["orders_submitted"][0]["ticker"] == "AAPL"


def test_save_run_never_raises_on_bad_data(tmp_path):
    with _tmp_db(tmp_path):
        db.init_db()
        db.save_run({"orders_submitted": object()})  # len() fails → caught, no raise
        assert db.load_recent_runs(5) == []          # nothing partially written


# ── db: signals column migration idempotent ────────────────────────────────

def test_save_signals_migration_idempotent(tmp_path):
    rows = [{"ticker": "AAPL", "recommendation": "BUY", "conviction": 4,
             "thesis": "t", "sl_price": 1.0, "tp1_price": 2.0,
             "tp2_price": 3.0, "tp3_price": 4.0}]
    with _tmp_db(tmp_path):
        db.init_db()
        db.save_signals("2026-07-05", rows)   # first call runs ALTER TABLE migration
        db.save_signals("2026-07-06", rows)   # second call: duplicate-column path
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        n = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        cols = {r[1] for r in conn.execute("PRAGMA table_info(signals)")}
        conn.close()

    assert n == 2
    assert {"sl_price", "tp1_price", "tp2_price", "tp3_price", "thesis"} <= cols


# ── db: transaction rollback ────────────────────────────────────────────────

def test_conn_rolls_back_on_exception(tmp_path):
    with _tmp_db(tmp_path):
        db.init_db()
        try:
            with db._conn() as conn:
                conn.execute(
                    "INSERT INTO runs (run_timestamp, run_json, created_at) "
                    "VALUES ('x', '{}', datetime('now'))"
                )
                raise RuntimeError("boom mid-transaction")
        except RuntimeError:
            pass
        assert db.load_recent_runs(5) == []  # insert rolled back


# ── chat/router: malformed LLM responses fall back, never raise ────────────

def _router_with_response(tool_input):
    # route_question now uses forced strict tool use. tool_input=None simulates
    # a response with no tool_use block (the analog of malformed free text).
    import chat.router as router
    mock_msg = MagicMock()
    if tool_input is None:
        text_block = MagicMock()
        text_block.type = "text"
        mock_msg.content = [text_block]
    else:
        block = MagicMock()
        block.type = "tool_use"
        block.input = tool_input
        mock_msg.content = [block]
    # router.route_question calls .stream(...) as a context manager and reads
    # get_final_message(), not .create() directly — see chat/router.py.
    stream_cm = MagicMock()
    stream_cm.__enter__.return_value.get_final_message.return_value = mock_msg
    return router, patch.object(router._client.messages, "stream", return_value=stream_cm)


def test_router_malformed_json_falls_back():
    router, p = _router_with_response(None)
    with p:
        out = router.route_question("¿cómo está mi portafolio?", {"account": {}, "live_positions": []})
    assert out["agents"] == ["portfolio_summary"]  # _FALLBACK shape
    assert "action" in out and "tickers" in out


def test_router_missing_fields_get_defaults():
    router, p = _router_with_response({"agents": ["research"]})
    with p:
        out = router.route_question("análisis de AAPL", {"account": {}, "live_positions": []})
    assert out["agents"] == ["research"]
    assert out["tickers"] == []            # setdefault filled
    assert out["action"]["type"] is None   # setdefault filled


def test_router_api_error_falls_back():
    import chat.router as router
    with patch.object(router._client.messages, "stream", side_effect=ConnectionError("api down")):
        out = router.route_question("hola", {"account": {}, "live_positions": []})
    assert out["agents"] == ["portfolio_summary"]


# ── chat/router: the decision-desk intents ─────────────────────────────────

def test_router_fallback_carries_the_pct_field():
    """sell_partial needs a fraction; the strict schema must expose it."""
    router, p = _router_with_response(None)
    with p:
        out = router.route_question("hola", {"account": {}, "live_positions": []})
    assert "pct" in out["action"]
    assert out["action"]["pct"] is None


def test_router_schema_declares_pct_as_required():
    import chat.router as router

    action_schema = router._ROUTER_TOOL["input_schema"]["properties"]["action"]
    assert "pct" in action_schema["properties"]
    assert "pct" in action_schema["required"], "strict tool use rejects missing keys"


def test_router_prompt_forbids_inventing_a_quantity():
    """The sizing engine computes the share count; the LLM must not guess it."""
    import chat.router as router

    prompt = router._SYSTEM_PROMPT
    assert "propose_trade" in prompt
    assert "action.qty null" in prompt or "never invent one" in prompt
    for name in ("action_sell_partial", "list_pending"):
        assert name in prompt


def test_router_passes_through_a_proposal_routing():
    router, p = _router_with_response({
        "agents": ["propose_trade"], "tickers": ["NVDA"],
        "action": {"type": None, "ticker": None, "qty": None, "pct": None,
                   "sl_price": None, "tp_price": None},
        "response_language": "es", "reasoning": "",
    })
    with p:
        out = router.route_question("¿compro NVDA?", {"account": {}, "live_positions": []})
    assert out["agents"] == ["propose_trade"]
    assert out["tickers"] == ["NVDA"]
    assert out["action"]["qty"] is None


def test_router_passes_through_a_partial_sell():
    router, p = _router_with_response({
        "agents": ["action_sell_partial"], "tickers": ["AAPL"],
        "action": {"type": "sell_partial", "ticker": "AAPL", "qty": None,
                   "pct": 0.5, "sl_price": None, "tp_price": None},
        "response_language": "es", "reasoning": "",
    })
    with p:
        out = router.route_question("vende la mitad de AAPL",
                                    {"account": {}, "live_positions": []})
    assert out["action"]["type"] == "sell_partial"
    assert out["action"]["pct"] == 0.5


# ── db: chat_decisions journal ──────────────────────────────────────────────

def _chat_journal(**over):
    base = {
        "session_id": "sess123",
        "question": "compra 5 de AAPL",
        "kind": "buy",
        "ticker": "AAPL",
        "paper_trading": 1,
        "source": "manual",
        "proposed_qty": 5,
        "user_choice": "confirmed",
        "overridden": 0,
    }
    base.update(over)
    return base


def test_chat_decision_roundtrip(tmp_path):
    with _tmp_db(tmp_path):
        db.init_db()
        assert db.record_chat_decision("dec1", _chat_journal()) is True
        assert db.update_chat_decision_outcome(
            "dec1", outcome="EXECUTED", filled_qty=5, fill_price=190.25,
            order_id="ord-1",
        ) is True
        rows = db.load_chat_decisions(10)

    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["user_choice"] == "confirmed"
    assert rows[0]["outcome"] == "EXECUTED"
    assert rows[0]["filled_qty"] == 5


def test_chat_decision_records_rejections_too(tmp_path):
    """A rejected proposal is evidence, not noise — funnel_log would drop it."""
    with _tmp_db(tmp_path):
        db.init_db()
        db.record_chat_decision("dec1", _chat_journal(user_choice="rejected"))
        db.update_chat_decision_outcome("dec1", outcome="NOT_EXECUTED")
        # Same ticker proposed again: both decisions must survive.
        db.record_chat_decision("dec2", _chat_journal(user_choice="confirmed"))
        rows = db.load_chat_decisions(10, ticker="AAPL")

    assert [r["user_choice"] for r in rows] == ["confirmed", "rejected"]  # newest first


def test_chat_decision_ignores_unknown_outcome_fields(tmp_path):
    with _tmp_db(tmp_path):
        db.init_db()
        db.record_chat_decision("dec1", _chat_journal())
        # An unknown key must not raise, and must not block the known ones.
        assert db.update_chat_decision_outcome(
            "dec1", outcome="PARTIAL", not_a_column="boom"
        ) is True
        rows = db.load_chat_decisions(10)

    assert rows[0]["outcome"] == "PARTIAL"


def test_chat_decision_never_raises_on_missing_table(tmp_path):
    """No init_db(): the writer warns and returns False instead of exploding."""
    with _tmp_db(tmp_path):
        assert db.record_chat_decision("dec1", _chat_journal()) is False
        assert db.load_chat_decisions(10) == []


# ── db: position_state migration + DB-primary round trip ───────────────────

def test_position_state_migration_idempotent(tmp_path):
    """init_db() adds the exit-engine columns on an old table and tolerates re-runs."""
    import sqlite3
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE position_state (
        ticker TEXT PRIMARY KEY, entry_price REAL, atr_14 REAL, sl_price REAL,
        sl_is_breakeven INTEGER DEFAULT 0, tp1_price REAL, tp1_filled INTEGER DEFAULT 0,
        tp2_price REAL, tp2_filled INTEGER DEFAULT 0, tp3_price REAL, tp3_filled INTEGER DEFAULT 0,
        qty_total INTEGER DEFAULT 0, qty_remaining INTEGER DEFAULT 0, entry_time TEXT,
        updated_at TEXT DEFAULT (datetime('now')))""")
    conn.commit()
    conn.close()

    with patch.object(db, "DB_PATH", path):
        db.init_db()   # first call: ALTER TABLE ... ADD COLUMN for each new column
        db.init_db()   # second call: duplicate-column path must be silent
        conn = sqlite3.connect(path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(position_state)")}
        conn.close()

    assert {"initial_stop", "trail_stop", "highest_close", "decay_count",
            "broker_stop_order_id", "broker_stop_price", "broker_stop_qty"} <= cols
    assert {"sl_price", "tp1_price"} <= cols, "legacy columns stay; nothing is dropped"


def test_position_state_round_trips_the_exit_engine_record(tmp_path):
    record = {
        "entry_price": 100.0, "entry_time": "2026-09-01T10:00:00", "atr_14": 2.0,
        "initial_stop": 95.0, "trail_stop": 97.5, "highest_close": 103.5,
        "qty_total": 10, "qty_remaining": 10, "decay_count": 2,
        "broker_stop_order_id": "ord-1", "broker_stop_price": 97.5, "broker_stop_qty": 10,
    }
    with _tmp_db(tmp_path):
        db.init_db()
        db.save_position_state({"AAPL": record})
        loaded = db.load_position_state()
    assert loaded == {"AAPL": record}, "every exit-engine column must flow back out"
