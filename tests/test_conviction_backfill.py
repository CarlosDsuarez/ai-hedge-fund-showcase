# tests/test_conviction_backfill.py
"""conviction_log.return_30d/return_60d have been NULL for the project's
whole life — nothing ever wrote them. backfill_conviction_returns() closes
that gap: for rows old enough (>=30d / >=60d since the logged date), it looks
up the first close on/after the target date and derives the forward return.

No real network: data.fetcher.get_closes is patched to return a small
DatetimeIndex frame built by hand.
"""
import os
import sys
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import db.database as dbmod


def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(dbmod, "DB_PATH", str(tmp_path / "test.db"))
    dbmod.init_db()
    return dbmod


def _closes_frame(ticker: str, dates: list[str], prices: list[float]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame({ticker: prices}, index=idx)


def test_backfill_computes_return_30d_for_due_row(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    today = date(2026, 9, 15)
    entry_date = today - timedelta(days=35)   # due for 30d, not yet for 60d
    db.append_conviction_log(entry_date.isoformat(), "AAPL", 4, 100.0)

    target = entry_date + timedelta(days=30)
    closes = _closes_frame("AAPL", [target.isoformat()], [120.0])

    with patch("reports.conviction_backfill.get_closes", return_value=closes):
        from reports.conviction_backfill import backfill_conviction_returns
        n = backfill_conviction_returns(today=today)

    assert n == 1
    row = db.load_conviction_log()[0]
    assert row["return_30d"] == pytest.approx(0.20)   # 120/100 - 1
    assert row["return_60d"] is None


def test_backfill_computes_both_returns_when_both_due(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    today = date(2026, 9, 15)
    entry_date = today - timedelta(days=65)
    db.append_conviction_log(entry_date.isoformat(), "MSFT", 5, 200.0)

    t30 = (entry_date + timedelta(days=30)).isoformat()
    t60 = (entry_date + timedelta(days=60)).isoformat()
    closes = _closes_frame("MSFT", [t30, t60], [220.0, 210.0])

    with patch("reports.conviction_backfill.get_closes", return_value=closes):
        from reports.conviction_backfill import backfill_conviction_returns
        n = backfill_conviction_returns(today=today)

    assert n == 1
    row = db.load_conviction_log()[0]
    assert row["return_30d"] == pytest.approx(0.10)    # 220/200 - 1
    assert row["return_60d"] == pytest.approx(0.05)    # 210/200 - 1


def test_backfill_uses_first_close_on_or_after_target_when_target_is_a_gap_day(tmp_path, monkeypatch):
    """target+30d falls on a non-trading day — the first AVAILABLE close on or
    after it must be used, not an exact-date lookup that finds nothing."""
    db = _fresh_db(tmp_path, monkeypatch)
    today = date(2026, 9, 15)
    entry_date = today - timedelta(days=40)
    db.append_conviction_log(entry_date.isoformat(), "TSLA", 3, 300.0)

    target = entry_date + timedelta(days=30)   # this exact day has no bar below
    next_trading_day = target + timedelta(days=2)
    closes = _closes_frame("TSLA",
                           [(target - timedelta(days=1)).isoformat(), next_trading_day.isoformat()],
                           [295.0, 330.0])
    assert target.isoformat() not in closes.index.astype(str), "sanity: target is a gap day"

    with patch("reports.conviction_backfill.get_closes", return_value=closes):
        from reports.conviction_backfill import backfill_conviction_returns
        backfill_conviction_returns(today=today)

    row = db.load_conviction_log()[0]
    assert row["return_30d"] == pytest.approx(330.0 / 300.0 - 1)


def test_backfill_skips_rows_not_yet_due(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    today = date(2026, 9, 15)
    db.append_conviction_log((today - timedelta(days=5)).isoformat(), "NVDA", 5, 900.0)

    with patch("reports.conviction_backfill.get_closes") as mock_closes:
        from reports.conviction_backfill import backfill_conviction_returns
        n = backfill_conviction_returns(today=today)

    assert n == 0
    mock_closes.assert_not_called()


def test_backfill_skips_ticker_with_no_price_data(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    today = date(2026, 9, 15)
    entry_date = today - timedelta(days=35)
    db.append_conviction_log(entry_date.isoformat(), "ZZZZ", 4, 50.0)

    with patch("reports.conviction_backfill.get_closes", return_value=pd.DataFrame()):
        from reports.conviction_backfill import backfill_conviction_returns
        n = backfill_conviction_returns(today=today)

    assert n == 0
    row = db.load_conviction_log()[0]
    assert row["return_30d"] is None


def test_backfill_returns_zero_when_nothing_needs_backfill(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    from reports.conviction_backfill import backfill_conviction_returns
    assert backfill_conviction_returns(today=date(2026, 9, 15)) == 0
