"""The confirmation ladder: s/n, and 'override' when a soft gate failed.

Soft gates are opinions and budget headroom — a person may overrule them, but
only in writing. 's' alone is not enough, and neither is 'yes': the override
word is the word.
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
    return {
        "account": {"equity": 100_000.0, "buying_power": 150_000.0, "cash": 50_000.0},
        "live_positions": [],
        "open_orders": [],
    }


BUY = {"type": "submit_buy", "ticker": "AAPL", "qty": 5,
       "sl_price": None, "tp_price": None}

_FILLED = {"id": "ord-1", "status": "filled", "qty": "5",
           "filled_qty": "5", "filled_avg_price": "190.0"}


def _run_with(tmp_path, inputs, *, duplicate=False):
    """duplicate=True seeds a run where the pipeline already bought AAPL today,
    which is the soft gate 'not_bought_today'."""
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("execution.trade_executor.resolve_price", return_value=190.0), \
         patch("data.fetcher.get_atr", return_value=3.5), \
         patch("execution.alpaca.submit_order", return_value=_FILLED) as submit, \
         patch("execution.alpaca.wait_for_fill", return_value=_FILLED), \
         patch("chat.actions._protect", return_value=("oco-1", True, "ok")), \
         patch("builtins.input", side_effect=inputs):
        db.init_db()
        if duplicate:
            db.save_run({"timestamp": "2026-09-04T09:00:00",
                         "orders_submitted": [{"ticker": "AAPL", "qty": 3}]})
        actions.handle_action(dict(BUY), _ctx())
        return submit, db.load_chat_decisions(5)


# ── no soft gates: one question is enough ───────────────────────────────────

def test_yes_executes_when_nothing_is_flagged(tmp_path):
    submit, rows = _run_with(tmp_path, ["s"])
    submit.assert_called_once()
    assert rows[0]["user_choice"] == "confirmed"
    assert rows[0]["overridden"] == 0


def test_no_executes_nothing(tmp_path):
    submit, rows = _run_with(tmp_path, ["n"])
    submit.assert_not_called()
    assert rows[0]["user_choice"] == "rejected"
    assert rows[0]["outcome"] == "NOT_EXECUTED"


def test_eof_cancels(tmp_path):
    submit, rows = _run_with(tmp_path, [EOFError()])
    submit.assert_not_called()
    assert rows[0]["user_choice"] == "rejected"


# ── a soft gate demands the word ────────────────────────────────────────────

def test_yes_alone_is_not_enough_when_a_soft_gate_failed(tmp_path):
    submit, rows = _run_with(tmp_path, ["s", ""], duplicate=True)
    submit.assert_not_called()
    assert rows[0]["user_choice"] == "rejected"
    assert "not_bought_today" in (rows[0]["gates_failed"] or "")


def test_override_executes_and_is_recorded(tmp_path):
    submit, rows = _run_with(tmp_path, ["s", "override"], duplicate=True)
    submit.assert_called_once()
    assert rows[0]["user_choice"] == "confirmed"
    assert rows[0]["overridden"] == 1


def test_override_is_case_insensitive(tmp_path):
    submit, _ = _run_with(tmp_path, ["s", "OVERRIDE"], duplicate=True)
    submit.assert_called_once()


def test_yes_is_not_an_override(tmp_path):
    """'yes' confirms; it does not overrule a warning."""
    submit, rows = _run_with(tmp_path, ["s", "yes"], duplicate=True)
    submit.assert_not_called()
    assert rows[0]["user_choice"] == "rejected"


def test_eof_at_the_override_prompt_cancels(tmp_path):
    submit, rows = _run_with(tmp_path, ["s", EOFError()], duplicate=True)
    submit.assert_not_called()
    assert rows[0]["user_choice"] == "rejected"


def test_the_warning_is_shown_before_the_question(tmp_path, capsys):
    _run_with(tmp_path, ["s", "override"], duplicate=True)
    out = capsys.readouterr().out
    assert "ya compró AAPL hoy" in out
    assert "override" in out


def test_every_branch_leaves_exactly_one_journal_row(tmp_path):
    for inputs in (["n"], ["s"], ["s", ""], ["s", "override"]):
        _, rows = _run_with(tmp_path / str(inputs), inputs, duplicate=("override" in inputs))
        assert len(rows) == 1, f"{inputs} produced {len(rows)} rows"
        assert rows[0]["gates_json"], "the gate verdicts on screen must be stored"
