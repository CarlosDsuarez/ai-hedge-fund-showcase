import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import db.database as dbmod


def _fresh_db(tmp_path, monkeypatch):
    """Point the DB layer at an isolated temp file and init the schema."""
    monkeypatch.setattr(dbmod, "DB_PATH", str(tmp_path / "test.db"))
    dbmod.init_db()
    return dbmod


def test_save_and_load_pending_round_trip(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.save_pending_signals("2026-06-12T10:00:00", [
        {"ticker": "AAPL", "composite_score": 0.40, "conviction": 4,
         "pm_action": "BUY", "size_pct": 7.5, "reason_deferred": "insufficient_budget"},
        {"ticker": "MSFT", "composite_score": 0.90, "conviction": 5,
         "pm_action": "BUY", "size_pct": 7.5, "reason_deferred": "insufficient_budget"},
    ])
    rows = db.load_pending_signals()
    assert [r["ticker"] for r in rows] == ["MSFT", "AAPL"]   # score DESC
    assert all(r["status"] == "PENDING" for r in rows)


def test_save_pending_dedupes_existing_ticker(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.save_pending_signals("t1", [{"ticker": "AAPL", "composite_score": 0.4, "conviction": 4}])
    db.save_pending_signals("t2", [{"ticker": "AAPL", "composite_score": 0.4, "conviction": 4}])
    assert len(db.load_pending_signals()) == 1


def test_mark_pending_executed_removes_from_pending(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.save_pending_signals("t1", [{"ticker": "AAPL", "composite_score": 0.4, "conviction": 4}])
    db.mark_pending_executed("AAPL")
    assert db.load_pending_signals() == []


def test_expire_pending_marks_old_rows_and_spares_fresh(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.save_pending_signals("t1", [
        {"ticker": "OLD", "composite_score": 0.5, "conviction": 4},
        {"ticker": "NEW", "composite_score": 0.5, "conviction": 4},
    ])
    # Back-date OLD's created_at by 20 days.
    with dbmod._conn() as conn:
        conn.execute("UPDATE pending_signals SET created_at = datetime('now','-20 days') WHERE ticker='OLD'")
    n = db.expire_pending_signals(14)
    assert n == 1
    tickers = [r["ticker"] for r in db.load_pending_signals()]
    assert tickers == ["NEW"]                      # OLD is EXPIRED, not PENDING
