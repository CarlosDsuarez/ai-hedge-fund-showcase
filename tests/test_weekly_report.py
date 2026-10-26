# tests/test_weekly_report.py
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime, timedelta
from unittest.mock import patch

import scheduler

def test_report_triggers_at_5pm():
    assert scheduler.REPORT_HOUR == 17


def _make_run_file(tmp_path, date_str: str, data: dict):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{date_str}_09-00.json").write_text(json.dumps(data))
    return str(tmp_path)


def test_scan_week_runs_includes_current_week(tmp_path):
    from reports.weekly_report import _scan_week_runs
    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    date_str = monday.strftime("%Y-%m-%d")
    _make_run_file(tmp_path, date_str, {"orders_submitted": [], "timestamp": date_str})
    runs = _scan_week_runs(str(tmp_path))
    assert len(runs) == 1
    assert runs[0]["timestamp"] == date_str


def test_scan_week_runs_excludes_prior_week(tmp_path):
    from reports.weekly_report import _scan_week_runs
    today = datetime.now()
    last_monday = (today - timedelta(days=today.weekday() + 7)).strftime("%Y-%m-%d")
    _make_run_file(tmp_path, last_monday, {"orders_submitted": []})
    runs = _scan_week_runs(str(tmp_path))
    assert len(runs) == 0


# ── _aggregate_week_trades: opened-this-week only (closed now comes from the
# ledger's trades table via _closed_this_week, not run-JSON scanning) ───────

def test_aggregate_week_trades_collects_orders():
    from reports.weekly_report import _aggregate_week_trades
    runs = [{"orders_submitted": [{"ticker": "AAPL", "qty": 10, "price": 180.0,
                                   "order": {"id": "id-1"}, "conviction": 4, "size_pct": 15.0}],
             "closed_positions": []}]
    opened = _aggregate_week_trades(runs)
    assert len(opened) == 1
    assert opened[0]["ticker"] == "AAPL"


def test_aggregate_week_trades_deduplicates_by_order_id():
    from reports.weekly_report import _aggregate_week_trades
    order = {"ticker": "AAPL", "qty": 10, "price": 180.0,
             "order": {"id": "same-id"}, "conviction": 4, "size_pct": 15.0}
    runs = [{"orders_submitted": [order], "closed_positions": []},
            {"orders_submitted": [order], "closed_positions": []}]
    opened = _aggregate_week_trades(runs)
    assert len(opened) == 1


# ── _get_open_positions: sourced from the broker's own book, not the last
# run's SL/TP-audit snapshot ─────────────────────────────────────────────────

def test_get_open_positions_reads_from_broker_and_merges_state():
    from reports.weekly_report import _get_open_positions
    broker_positions = [{
        "symbol": "AAPL", "qty": "10", "avg_entry_price": "180.0",
        "current_price": "189.0", "unrealized_plpc": "0.05",
        "unrealized_pl": "90.0",
    }]
    state = {"AAPL": {"sl_price": 165.0, "trail_stop": 175.0}}
    with patch("execution.alpaca.get_positions", return_value=broker_positions):
        positions = _get_open_positions(state)
    assert len(positions) == 1
    p = positions[0]
    assert p["ticker"] == "AAPL"
    assert p["entry_price"] == 180.0
    assert p["current_price"] == 189.0
    assert p["dollar_pnl"] == 90.0
    assert p["sl_price"] == 165.0
    assert p["trail_stop"] == 175.0


def test_get_open_positions_handles_broker_failure():
    from reports.weekly_report import _get_open_positions
    with patch("execution.alpaca.get_positions", side_effect=RuntimeError("down")):
        assert _get_open_positions({}) == []


def test_get_open_positions_empty_broker_book():
    from reports.weekly_report import _get_open_positions
    with patch("execution.alpaca.get_positions", return_value=[]):
        assert _get_open_positions({}) == []


# ── _get_account_snapshot: replaces the stale data/portfolio_state.json read ─

def test_get_account_snapshot_reads_equity_cash_and_week_pnl():
    from reports.weekly_report import _get_account_snapshot
    account = {"equity": "99754.89", "cash": "72555.23"}
    history = {
        "timestamp": [1717200000, 1717286400, 1717372800],  # Mon, Tue, Wed (UTC-ish)
        "equity": [100000.0, 100200.0, 99754.89],
    }
    with patch("execution.alpaca.get_account", return_value=account), \
         patch("execution.alpaca.get_portfolio_history", return_value=history):
        equity, cash, week_pnl = _get_account_snapshot()
    assert equity == 99754.89
    assert cash == 72555.23
    assert isinstance(week_pnl, float)


def test_get_account_snapshot_falls_back_to_zeros_on_broker_failure():
    from reports.weekly_report import _get_account_snapshot
    with patch("execution.alpaca.get_account", side_effect=RuntimeError("down")), \
         patch("execution.alpaca.get_portfolio_history", side_effect=RuntimeError("down")):
        equity, cash, week_pnl = _get_account_snapshot()
    assert equity == 0.0
    assert cash == 0.0
    assert week_pnl == 0.0


def test_get_account_snapshot_zero_week_pnl_on_empty_history():
    from reports.weekly_report import _get_account_snapshot
    account = {"equity": "100000", "cash": "50000"}
    with patch("execution.alpaca.get_account", return_value=account), \
         patch("execution.alpaca.get_portfolio_history", return_value={}):
        equity, cash, week_pnl = _get_account_snapshot()
    assert equity == 100000.0
    assert week_pnl == 0.0


# ── _format_msg1 ──────────────────────────────────────────────────────────────

def test_format_msg1_contains_equity_and_cash():
    from reports.weekly_report import _format_msg1
    msg = _format_msg1("Week of: May 26 – May 30", 99754.0, 72555.0, -316.0, 0.0, 0.0, [], [])
    assert "99,754" in msg
    assert "72,555" in msg


def test_format_msg1_lists_opened_trades():
    from reports.weekly_report import _format_msg1
    opened = [{"ticker": "META", "qty": 24, "price": 610.26,
               "conviction": 4, "order": {"created_at": "2026-05-26T13:01:00Z"}}]
    msg = _format_msg1("Week of: May 26 – May 30", 0, 0, 0, 0, 0.0, opened, [])
    assert "META" in msg
    assert "24" in msg
    assert "610.26" in msg


def test_format_msg1_shows_no_closes_when_empty():
    from reports.weekly_report import _format_msg1
    msg = _format_msg1("Week of: May 26 – May 30", 0, 0, 0, 0, 0.0, [], [])
    assert "No positions closed this week" in msg


def test_format_msg1_lists_closed_positions_with_pct_and_dollars():
    from reports.weekly_report import _format_msg1
    closed = [{"ticker": "NVDA", "unrealized_plpc": 8.2, "reason": "STOP_TRAIL", "dollar_pnl": 520.0}]
    msg = _format_msg1("Week of: May 26 – May 30", 0, 0, 0, 520.0, 0.0, [], closed)
    assert "NVDA" in msg
    assert "+8.2%" in msg
    assert "+520" in msg


def test_format_msg1_includes_realized_ledger_line_when_trades_exist(tmp_path, monkeypatch):
    from reports.weekly_report import _format_msg1
    import db.database as dbmod
    monkeypatch.setattr(dbmod, "DB_PATH", str(tmp_path / "test.db"))
    dbmod.init_db()
    dbmod.insert_fills([
        {"id": "b1", "symbol": "AAPL", "side": "buy", "qty": 10, "price": 100.0,
         "transaction_time": "2026-06-01T10:00:00"},
        {"id": "s1", "symbol": "AAPL", "side": "sell", "qty": 10, "price": 110.0,
         "transaction_time": "2026-06-11T10:00:00", "tag": "stop_trail"},
    ])
    dbmod.rebuild_trades()
    msg = _format_msg1("Week of: May 26 – May 30", 0, 0, 0, 0, 0.0, [], [])
    assert "Realized" in msg
    assert "100%" in msg   # 1/1 trades won


def test_format_msg1_omits_realized_ledger_line_with_no_scored_trades(tmp_path, monkeypatch):
    from reports.weekly_report import _format_msg1
    import db.database as dbmod
    monkeypatch.setattr(dbmod, "DB_PATH", str(tmp_path / "empty.db"))
    dbmod.init_db()
    msg = _format_msg1("Week of: May 26 – May 30", 0, 0, 0, 0, 0.0, [], [])
    assert "Realized (ledger" not in msg


# ── _format_msg2: Stop column (trail_stop, else sl_price) ────────────────────

def test_format_msg2_shows_trail_stop_when_present():
    from reports.weekly_report import _format_msg2
    positions = [{"ticker": "BRK-B", "entry_price": 484.11, "current_price": 473.76,
                  "unrealized_pnl_pct": -2.18, "dollar_pnl": -93.0,
                  "qty": 9, "sl_price": 454.0, "trail_stop": 460.0}]
    msg = _format_msg2(positions, [])
    assert "BRK-B" in msg
    assert "484.11" in msg
    assert "473.76" in msg
    assert "-2.2%" in msg
    assert "460.00" in msg    # trail_stop wins over sl_price
    assert "454.00" not in msg


def test_format_msg2_falls_back_to_sl_price_without_trail_stop():
    from reports.weekly_report import _format_msg2
    positions = [{"ticker": "AAPL", "entry_price": 180.0, "current_price": 189.0,
                  "unrealized_pnl_pct": 5.0, "dollar_pnl": 90.0,
                  "qty": 10, "sl_price": 165.0, "trail_stop": None}]
    msg = _format_msg2(positions, [])
    assert "165.00" in msg


def test_format_msg2_shows_stop_na_without_any_stop():
    from reports.weekly_report import _format_msg2
    positions = [{"ticker": "AAPL", "entry_price": 180.0, "current_price": 189.0,
                  "unrealized_pnl_pct": 5.0, "dollar_pnl": 90.0, "qty": 10,
                  "sl_price": None, "trail_stop": None}]
    msg = _format_msg2(positions, [])
    assert "Stop N/A" in msg


def test_format_msg2_aggregates_signals_from_runs():
    from reports.weekly_report import _format_msg2
    runs = [{"pm": {"recommendations": [
                {"ticker": "AAPL", "action": "BUY", "conviction": 4},
                {"ticker": "TSLA", "action": "SELL", "conviction": 3},
                {"ticker": "MSFT", "action": "HOLD", "conviction": 2},
             ]},
             "risk": {"sharpe_ratio": 1.23, "var_95": "-$4,500", "status": "OK"},
             "regime": {"regime": "BULL"}}]
    msg = _format_msg2([], runs)
    assert "BUY: 1" in msg
    assert "SELL: 1" in msg
    assert "HOLD: 1" in msg
    assert "BULL" in msg
    assert "1.23" in msg


def test_format_msg2_no_positions_message():
    from reports.weekly_report import _format_msg2
    assert "No open positions" in _format_msg2([], [])


# ── generate_weekly_report: full integration, everything broker/db mocked ───

def test_generate_weekly_report_sends_two_telegram_messages(tmp_path):
    from reports.weekly_report import generate_weekly_report

    runs_dir = tmp_path / "logs" / "runs"
    runs_dir.mkdir(parents=True)

    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    run_data = {
        "timestamp": monday.strftime("%Y-%m-%d"),
        "orders_submitted": [], "closed_positions": [],
        "pm": {"recommendations": []},
        "risk": {"sharpe_ratio": 1.0, "var_95": "N/A", "status": "OK", "violations": []},
        "regime": {"regime": "BULL"},
    }
    (runs_dir / f"{monday.strftime('%Y-%m-%d')}_09-00.json").write_text(json.dumps(run_data))
    (tmp_path / "logs" / "position_state.json").write_text(json.dumps({}))

    with patch("reports.weekly_report.send_message") as mock_send, \
         patch("reports.weekly_report._get_report_root", return_value=str(tmp_path)), \
         patch("execution.alpaca.get_account", return_value={"equity": "100000", "cash": "50000"}), \
         patch("execution.alpaca.get_portfolio_history", return_value={}), \
         patch("execution.alpaca.get_positions", return_value=[]), \
         patch("db.database.load_trades", return_value=[]):
        result = generate_weekly_report()

    assert mock_send.call_count == 2
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_weekly_report_survives_total_broker_outage(tmp_path):
    """Every broker call wrapped in try/except → fallbacks: the report must
    still generate and send, never crash, on a full Alpaca outage."""
    from reports.weekly_report import generate_weekly_report

    runs_dir = tmp_path / "logs" / "runs"
    runs_dir.mkdir(parents=True)
    (tmp_path / "logs" / "position_state.json").write_text(json.dumps({}))

    with patch("reports.weekly_report.send_message") as mock_send, \
         patch("reports.weekly_report._get_report_root", return_value=str(tmp_path)), \
         patch("execution.alpaca.get_account", side_effect=RuntimeError("down")), \
         patch("execution.alpaca.get_portfolio_history", side_effect=RuntimeError("down")), \
         patch("execution.alpaca.get_positions", side_effect=RuntimeError("down")), \
         patch("db.database.load_trades", side_effect=RuntimeError("down")):
        result = generate_weekly_report()

    assert mock_send.call_count == 2
    assert isinstance(result, str)
