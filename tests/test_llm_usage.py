# tests/test_llm_usage.py — token/cost tracking (auditoría Fase A, prompt 6)
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import llm_usage


def _fake_response(tin=100, tout=50, cache=0):
    resp = MagicMock()
    resp.usage.input_tokens = tin
    resp.usage.output_tokens = tout
    resp.usage.cache_read_input_tokens = cache
    return resp


def setup_function(_fn):
    llm_usage.reset()


def test_record_usage_accumulates_totals_and_persists():
    with patch("db.database.save_llm_usage") as mock_save:
        llm_usage.record_usage("research", "claude-sonnet-4-6", _fake_response(1000, 300))
        llm_usage.record_usage("sentiment", "claude-haiku-4-5-20251001", _fake_response(500, 100, cache=200))

    s = llm_usage.run_summary()
    assert s["calls"] == 2
    assert s["input"] == 1500
    assert s["output"] == 400
    assert s["cache_read"] == 200
    # sonnet: 1000/1e6*3 + 300/1e6*15 = 0.0075 ; haiku: 500/1e6*1 + 100/1e6*5 = 0.001
    assert abs(s["cost_usd"] - 0.0085) < 1e-9
    assert mock_save.call_count == 2
    mock_save.assert_any_call("research", "claude-sonnet-4-6", 1000, 300, 0)


def test_record_usage_never_raises():
    broken = MagicMock()
    del broken.usage  # accessing .usage raises AttributeError
    llm_usage.record_usage("pm", "claude-sonnet-4-6", broken)  # must not raise
    assert llm_usage.run_summary()["calls"] == 0


def test_save_llm_usage_writes_row(tmp_path):
    import db.database as db

    with patch.object(db, "DB_PATH", str(tmp_path / "test.db")):
        db.save_llm_usage("research", "claude-sonnet-4-6", 123, 45, 6)
        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        row = conn.execute("SELECT agent, model, tokens_in, tokens_out, cache_read FROM llm_usage").fetchone()
        conn.close()

    assert row == ("research", "claude-sonnet-4-6", 123, 45, 6)
