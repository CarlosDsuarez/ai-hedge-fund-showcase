# tests/test_ledger.py
"""D. Ledger — fills/trades tables, FIFO trade reconstruction, broker-exit
reconciliation, tag→reason mapping, recent_stop_exits' trading-day window.

Every broker call is patched at execution.alpaca.<name>; the DB is a fresh
temp-file sqlite the way tests/test_pending_signals_db.py isolates it. No real
network, no shared state between tests.
"""
import os
import sys
from datetime import date, timedelta
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import db.database as dbmod


def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(dbmod, "DB_PATH", str(tmp_path / "test.db"))
    dbmod.init_db()
    return dbmod


def _fill(id, symbol, side, qty, price, ts, order_id=None, tag=None, cid=None):
    return {
        "id": id, "symbol": symbol, "side": side, "qty": qty, "price": price,
        "transaction_time": ts, "order_id": order_id or id,
        "client_order_id": cid, "tag": tag,
    }


# ── db.insert_fills / load_fills / last_fill_time ────────────────────────────

def test_insert_fills_dedupes_on_activity_id(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    fill = _fill("act-1", "AAPL", "buy", 10, 180.0, "2026-06-01T10:00:00Z")
    n1 = db.insert_fills([fill])
    n2 = db.insert_fills([fill])   # same activity id — must be a no-op
    assert [f["id"] for f in n1] == ["act-1"]   # the inserted rows, not a count
    assert n2 == []
    assert len(db.load_fills()) == 1


def test_insert_fills_returns_only_the_new_subset(tmp_path, monkeypatch):
    """reconcile_fills runs its broker-stop signal over exactly what this
    cycle inserted — the boundary row Alpaca re-sends (inclusive `after`)
    must not be in that subset."""
    db = _fresh_db(tmp_path, monkeypatch)
    old = _fill("a1", "AAPL", "sell", 10, 90.0, "2026-06-05T10:00:00Z", tag="bstop")
    db.insert_fills([old])
    new = _fill("a2", "MSFT", "buy", 5, 400.0, "2026-06-06T10:00:00Z")
    inserted = db.insert_fills([old, new])
    assert [f["id"] for f in inserted] == ["a2"]


def test_insert_fills_filters_by_symbol(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.insert_fills([
        _fill("a1", "AAPL", "buy", 10, 180.0, "2026-06-01T10:00:00Z"),
        _fill("a2", "MSFT", "buy", 5, 400.0, "2026-06-01T10:00:00Z"),
    ])
    assert len(db.load_fills(symbol="AAPL")) == 1
    assert len(db.load_fills()) == 2


def test_last_fill_time_returns_max_transaction_time(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    assert db.last_fill_time() is None
    db.insert_fills([
        _fill("a1", "AAPL", "buy", 10, 180.0, "2026-06-01T10:00:00Z"),
        _fill("a2", "AAPL", "sell", 10, 190.0, "2026-06-05T10:00:00Z"),
    ])
    assert db.last_fill_time() == "2026-06-05T10:00:00Z"


# ── db.rebuild_trades: FIFO arithmetic ───────────────────────────────────────

def test_rebuild_trades_matches_single_buy_single_sell(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.insert_fills([
        _fill("a1", "AAPL", "buy", 10, 100.0, "2026-06-01T10:00:00"),
        _fill("a2", "AAPL", "sell", 10, 110.0, "2026-06-11T10:00:00", tag="stop_trail"),
    ])
    n = db.rebuild_trades()
    assert n == 1
    trades = db.load_trades()
    assert len(trades) == 1
    t = trades[0]
    assert t["symbol"] == "AAPL"
    assert t["qty"] == 10
    assert t["entry_price"] == 100.0
    assert t["exit_price"] == 110.0
    assert t["pnl_usd"] == 100.0            # 10 * (110 - 100)
    assert t["pnl_pct"] == pytest.approx(0.10)
    assert t["holding_days"] == 10
    assert t["exit_reason"] == "STOP_TRAIL"


def test_rebuild_trades_two_lots_one_sell_produces_two_rows(tmp_path, monkeypatch):
    """A sell spanning two buy lots must emit one trades row per lot consumed,
    not a single blended-average row (decision 7)."""
    db = _fresh_db(tmp_path, monkeypatch)
    db.insert_fills([
        _fill("b1", "AAPL", "buy", 5, 100.0, "2026-06-01T10:00:00"),
        _fill("b2", "AAPL", "buy", 5, 120.0, "2026-06-03T10:00:00"),
        _fill("s1", "AAPL", "sell", 10, 130.0, "2026-06-10T10:00:00", tag="rank_decay"),
    ])
    n = db.rebuild_trades()
    assert n == 2
    trades = sorted(db.load_trades(), key=lambda t: t["entry_price"])
    assert trades[0]["entry_price"] == 100.0
    assert trades[0]["qty"] == 5
    assert trades[0]["pnl_usd"] == pytest.approx(150.0)   # 5 * (130-100)
    assert trades[0]["exit_reason"] == "RANK_DECAY"
    assert trades[1]["entry_price"] == 120.0
    assert trades[1]["qty"] == 5
    assert trades[1]["pnl_usd"] == pytest.approx(50.0)    # 5 * (130-120)


def test_rebuild_trades_partial_sell_leaves_remaining_lot(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.insert_fills([
        _fill("b1", "AAPL", "buy", 10, 100.0, "2026-06-01T10:00:00"),
        _fill("s1", "AAPL", "sell", 4, 110.0, "2026-06-05T10:00:00", tag="pm_sell"),
    ])
    db.rebuild_trades()
    trades = db.load_trades()
    assert len(trades) == 1
    assert trades[0]["qty"] == 4
    assert trades[0]["exit_reason"] == "PM_SELL"
    # The remaining 6 shares are still an open lot: a second sell should draw
    # from them without needing a new buy fill.
    db.insert_fills([_fill("s2", "AAPL", "sell", 6, 115.0, "2026-06-20T10:00:00", tag="time_cap")])
    db.rebuild_trades()
    trades = db.load_trades()
    assert len(trades) == 2
    second = [t for t in trades if t["exit_reason"] == "TIME_CAP"][0]
    assert second["qty"] == 6
    assert second["entry_price"] == 100.0


def test_rebuild_trades_sell_with_no_matching_lot_is_visible_but_unscored(tmp_path, monkeypatch):
    """A position opened before LEDGER_BACKFILL_FROM has no buy fill on record.
    Its sell must still produce a trades row (visible) with NULL entry/pnl
    (excluded from P&L stats) rather than being dropped or crashing FIFO."""
    db = _fresh_db(tmp_path, monkeypatch)
    db.insert_fills([_fill("s1", "TSLA", "sell", 8, 250.0, "2026-06-05T10:00:00", tag="bstop")])
    n = db.rebuild_trades()
    assert n == 1
    t = db.load_trades()[0]
    assert t["entry_price"] is None
    assert t["pnl_usd"] is None
    assert t["qty"] == 8
    assert t["exit_reason"] == "BROKER_STOP"


def test_rebuild_trades_is_idempotent(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.insert_fills([
        _fill("b1", "AAPL", "buy", 10, 100.0, "2026-06-01T10:00:00"),
        _fill("s1", "AAPL", "sell", 10, 110.0, "2026-06-11T10:00:00", tag="stop_initial"),
    ])
    db.rebuild_trades()
    first = db.load_trades()
    db.rebuild_trades()
    second = db.load_trades()
    assert len(first) == len(second) == 1
    assert first[0]["pnl_usd"] == second[0]["pnl_usd"]


def test_rebuild_trades_unknown_tag_maps_to_unknown(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.insert_fills([
        _fill("b1", "AAPL", "buy", 10, 100.0, "2026-06-01T10:00:00"),
        _fill("s1", "AAPL", "sell", 10, 110.0, "2026-06-11T10:00:00", tag="something_else"),
    ])
    db.rebuild_trades()
    assert db.load_trades()[0]["exit_reason"] == "UNKNOWN"


def test_rebuild_trades_main_tag_maps_to_legacy(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.insert_fills([
        _fill("b1", "AAPL", "buy", 10, 100.0, "2026-06-01T10:00:00"),
        _fill("s1", "AAPL", "sell", 10, 110.0, "2026-06-11T10:00:00", tag="main"),
    ])
    db.rebuild_trades()
    assert db.load_trades()[0]["exit_reason"] == "LEGACY"


def test_rebuild_trades_oco_tag_maps_to_legacy_oco(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.insert_fills([
        _fill("b1", "AAPL", "buy", 10, 100.0, "2026-06-01T10:00:00"),
        _fill("s1", "AAPL", "sell", 10, 115.0, "2026-06-11T10:00:00", tag="oco"),
    ])
    db.rebuild_trades()
    assert db.load_trades()[0]["exit_reason"] == "LEGACY_OCO"


def test_load_trades_since_filters_by_exit_time(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    db.insert_fills([
        _fill("b1", "AAPL", "buy", 10, 100.0, "2026-05-01T10:00:00"),
        _fill("s1", "AAPL", "sell", 10, 110.0, "2026-05-05T10:00:00", tag="stop_initial"),
        _fill("b2", "MSFT", "buy", 5, 300.0, "2026-06-01T10:00:00"),
        _fill("s2", "MSFT", "sell", 5, 310.0, "2026-06-10T10:00:00", tag="stop_trail"),
    ])
    db.rebuild_trades()
    since = db.load_trades(since="2026-06-01T00:00:00")
    assert len(since) == 1
    assert since[0]["symbol"] == "MSFT"


# ── db.record_exit / recent_stop_exits ───────────────────────────────────────

def test_recent_stop_exits_filters_by_reason_and_busday_window(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    today = date(2026, 9, 15)   # Tuesday
    db.record_exit("AAPL", "2026-09-14", "STOP_INITIAL", 10, 100.0, "r1")   # 1 busday ago
    db.record_exit("MSFT", "2026-08-20", "STOP_TRAIL", 5, 200.0, "r1")      # far outside window
    db.record_exit("TSLA", "2026-09-14", "PM_SELL", 3, 250.0, "r1")         # not a stop reason
    result = db.recent_stop_exits(10, today=today)
    assert "AAPL" in result
    assert "MSFT" not in result
    assert "TSLA" not in result


def test_recent_stop_exits_includes_broker_stop_reason(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    today = date(2026, 9, 15)
    db.record_exit("NVDA", "2026-09-10", "BROKER_STOP", 2, 900.0, "r1")
    result = db.recent_stop_exits(10, today=today)
    assert "NVDA" in result


def test_recent_stop_exits_excludes_legacy_oco(tmp_path, monkeypatch):
    """A historic OCO fill may be the take-profit leg — a winner. Cooling a
    winner down for 10 trading days is the wrong penalty, so LEGACY_OCO is
    never a stop for cooldown purposes; only the three real stop reasons are."""
    db = _fresh_db(tmp_path, monkeypatch)
    today = date(2026, 9, 15)
    db.record_exit("HON", "2026-09-14", "LEGACY_OCO", 10, 200.0, "r1")
    db.record_exit("UNK", "2026-09-14", "UNKNOWN", 10, 200.0, "r1")
    result = db.recent_stop_exits(10, today=today)
    assert "HON" not in result
    assert "UNK" not in result


def test_recent_stop_exits_excludes_exactly_at_window_edge(tmp_path, monkeypatch):
    """busday_count(exit_date, today) < days — the boundary is exclusive."""
    db = _fresh_db(tmp_path, monkeypatch)
    today = date(2026, 9, 15)   # Tuesday
    # Exactly 10 business days before 2026-09-15 is 2026-09-01 (Tue).
    db.record_exit("OLD", "2026-09-01", "STOP_INITIAL", 1, 50.0, "r1")
    result = db.recent_stop_exits(10, today=today)
    assert "OLD" not in result


# ── db.load_conviction_rows_needing_backfill ─────────────────────────────────

def test_load_conviction_rows_needing_backfill_selects_old_null_rows(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    today = date(2026, 9, 15)
    db.append_conviction_log((today - timedelta(days=40)).isoformat(), "AAPL", 4, 180.0)  # due for 30d
    db.append_conviction_log((today - timedelta(days=5)).isoformat(), "MSFT", 5, 400.0)   # too young
    rows = db.load_conviction_rows_needing_backfill(min_age_days=30, today=today)
    tickers = {r["ticker"] for r in rows}
    assert "AAPL" in tickers
    assert "MSFT" not in tickers


def test_load_conviction_rows_needing_backfill_skips_already_filled(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    today = date(2026, 9, 15)
    old_date = (today - timedelta(days=90)).isoformat()
    db.append_conviction_log(old_date, "AAPL", 4, 180.0)
    db.update_conviction_returns("AAPL", old_date, return_30d=0.05, return_60d=0.07)
    rows = db.load_conviction_rows_needing_backfill(min_age_days=30, today=today)
    assert rows == []


# ── execution.ledger: tag→reason mapping ─────────────────────────────────────

def test_exit_reason_by_tag_mapping():
    from execution.ledger import EXIT_REASON_BY_TAG
    assert EXIT_REASON_BY_TAG["bstop"] == "BROKER_STOP"
    # An OCO leg can be either the stop (loss) or the take-profit (win); the
    # tag cannot tell them apart, so it is its own reason and never a stop.
    assert EXIT_REASON_BY_TAG["oco"] == "LEGACY_OCO"
    assert EXIT_REASON_BY_TAG["stop_initial"] == "STOP_INITIAL"
    assert EXIT_REASON_BY_TAG["stop_trail"] == "STOP_TRAIL"
    assert EXIT_REASON_BY_TAG["rank_decay"] == "RANK_DECAY"
    assert EXIT_REASON_BY_TAG["pm_sell"] == "PM_SELL"
    assert EXIT_REASON_BY_TAG["time_cap"] == "TIME_CAP"
    assert EXIT_REASON_BY_TAG["drift"] == "DRIFT_TRIM"
    assert EXIT_REASON_BY_TAG["main"] == "LEGACY"


def test_tag_from_client_order_id_parses_first_token():
    from execution.ledger import _tag_from_client_order_id
    assert _tag_from_client_order_id("stop_initial-AAPL-sell-2026-09-15") == "stop_initial"
    assert _tag_from_client_order_id("bstop-AAPL-150.00-2026-09-15") == "bstop"
    assert _tag_from_client_order_id(None) is None
    assert _tag_from_client_order_id("") is None


# ── execution.ledger.reconcile_fills ──────────────────────────────────────────

def _activity(id, symbol, side, qty, price, ts, order_id=None):
    return {"id": id, "activity_type": "FILL", "symbol": symbol, "side": side,
            "qty": str(qty), "price": str(price), "transaction_time": ts,
            "order_id": order_id or id, "type": "fill"}


def test_reconcile_fills_never_raises_on_broker_error(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger
    with patch("execution.alpaca.get_activities", side_effect=RuntimeError("network down")):
        result = ledger.reconcile_fills()
    assert result["new_fills"] == 0
    assert result["trades_rebuilt"] == 0
    assert result["broker_exits"] == []
    assert "network down" in result["error"]


def test_reconcile_fills_resolves_sell_tag_via_get_order_once_per_order(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger
    activities = [
        _activity("a1", "AAPL", "buy", 10, 100.0, "2026-06-01T10:00:00Z", order_id="o-buy"),
        _activity("a2", "AAPL", "sell", 5, 110.0, "2026-06-10T10:00:00Z", order_id="o-sell"),
        _activity("a3", "AAPL", "sell", 5, 111.0, "2026-06-10T10:00:01Z", order_id="o-sell"),
    ]
    with patch("execution.alpaca.get_activities", return_value=activities), \
         patch("execution.alpaca.get_order",
               return_value={"client_order_id": "stop_trail-AAPL-sell-2026-06-10"}) as mock_get_order, \
         patch("execution.alpaca.get_positions", return_value=[]):
        result = ledger.reconcile_fills()

    assert mock_get_order.call_count == 1   # two fills, same order_id → one lookup
    assert result["new_fills"] == 3
    assert result["trades_rebuilt"] >= 1


def test_reconcile_fills_detects_broker_stop_exit_and_clears_state(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger
    from execution import order_manager

    state_file = tmp_path / "position_state.json"
    monkeypatch.setattr(order_manager, "STATE_FILE", str(state_file))
    order_manager.save_position_state({"AAPL": {"qty_remaining": 10, "entry_price": 100.0}})

    activities = [
        _activity("a1", "AAPL", "buy", 10, 100.0, "2026-06-01T10:00:00Z", order_id="o-buy"),
        _activity("a2", "AAPL", "sell", 10, 90.0, "2026-06-05T10:00:00Z", order_id="o-stop"),
    ]
    with patch("execution.alpaca.get_activities", return_value=activities), \
         patch("execution.alpaca.get_order",
               return_value={"client_order_id": "bstop-AAPL-90.00-2026-06-05"}), \
         patch("execution.alpaca.get_positions", return_value=[]):
        result = ledger.reconcile_fills()

    assert len(result["broker_exits"]) == 1
    assert result["broker_exits"][0]["ticker"] == "AAPL"
    assert result["broker_exits"][0]["reason"] == "BROKER_STOP"
    assert order_manager.load_position_state() == {}   # state record removed


def test_reconcile_fills_detects_untagged_exit_via_missing_position(tmp_path, monkeypatch):
    """A position tracked in state that has simply vanished from get_positions()
    (e.g. DELETE /v2/positions generated a tagless sell) must still surface as
    a broker exit with reason UNKNOWN — nothing may vanish silently."""
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger
    from execution import order_manager

    state_file = tmp_path / "position_state.json"
    monkeypatch.setattr(order_manager, "STATE_FILE", str(state_file))
    order_manager.save_position_state({"GOOGL": {"qty_remaining": 4, "entry_price": 150.0}})

    with patch("execution.alpaca.get_activities", return_value=[]), \
         patch("execution.alpaca.get_positions", return_value=[]):
        result = ledger.reconcile_fills()

    assert len(result["broker_exits"]) == 1
    exit_row = result["broker_exits"][0]
    assert exit_row["ticker"] == "GOOGL"
    assert exit_row["reason"] == "UNKNOWN"
    assert order_manager.load_position_state() == {}


def test_reconcile_fills_leaves_tracked_open_positions_alone(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger
    from execution import order_manager

    state_file = tmp_path / "position_state.json"
    monkeypatch.setattr(order_manager, "STATE_FILE", str(state_file))
    order_manager.save_position_state({"AAPL": {"qty_remaining": 10, "entry_price": 100.0}})

    with patch("execution.alpaca.get_activities", return_value=[]), \
         patch("execution.alpaca.get_positions",
               return_value=[{"symbol": "AAPL", "qty": "10"}]):
        result = ledger.reconcile_fills()

    assert result["broker_exits"] == []
    assert "AAPL" in order_manager.load_position_state()


def test_reconcile_fills_get_positions_failure_does_not_fabricate_exits(tmp_path, monkeypatch):
    """Review fix (round 1): a transient get_positions() failure must NOT be
    read as "every tracked position vanished". Before the fix, open_symbols
    fell back to an empty set on any get_positions() exception, so
    missing_symbols = set(state) - set() = every tracked ticker — fabricating
    an UNKNOWN exit for each and wiping position_state.json on a timeout."""
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger
    from execution import order_manager

    state_file = tmp_path / "position_state.json"
    monkeypatch.setattr(order_manager, "STATE_FILE", str(state_file))
    tracked_state = {
        "AAPL": {"qty_remaining": 10, "entry_price": 100.0},
        "MSFT": {"qty_remaining": 5, "entry_price": 300.0},
    }
    order_manager.save_position_state(dict(tracked_state))

    with patch("execution.alpaca.get_activities", return_value=[]), \
         patch("execution.alpaca.get_positions", side_effect=RuntimeError("timeout")):
        result = ledger.reconcile_fills()

    assert result["broker_exits"] == [], "a get_positions() failure must not fabricate exits"
    assert result.get("positions_unavailable") is True
    assert order_manager.load_position_state() == tracked_state, "state must be untouched"
    assert dbmod.recent_stop_exits(365) == {}, "no exits rows should have been recorded"


def test_reconcile_fills_dry_run_writes_nothing(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger
    from execution import order_manager

    state_file = tmp_path / "position_state.json"
    monkeypatch.setattr(order_manager, "STATE_FILE", str(state_file))
    order_manager.save_position_state({"AAPL": {"qty_remaining": 10, "entry_price": 100.0}})

    activities = [_activity("a1", "AAPL", "sell", 10, 90.0, "2026-06-05T10:00:00Z", order_id="o-stop")]
    with patch("execution.alpaca.get_activities", return_value=activities), \
         patch("execution.alpaca.get_order",
               return_value={"client_order_id": "bstop-AAPL-90.00-2026-06-05"}), \
         patch("execution.alpaca.get_positions", return_value=[]):
        result = ledger.reconcile_fills(dry_run=True)

    assert result["new_fills"] == 1
    assert result["trades_rebuilt"] == 0
    assert len(result["broker_exits"]) == 1   # reported...
    assert dbmod.load_fills() == []            # ...but nothing written
    assert "AAPL" in order_manager.load_position_state()   # state untouched


# ── C1: signal (1) runs on NEW fills only, and never on a name still open ────

def test_rebought_after_broker_stop_keeps_state(tmp_path, monkeypatch):
    """Reviewer probe (C1a). Backfill: HON stopped out in June, re-bought in
    July, still held today. The live position_state.json is full of such
    re-bought names (HON 5x, JPM 6x, GOOGL 5x); before the fix the first live
    run recorded a broker exit for each, deleted its (migrated, ladder-raised)
    state record, and fired a spurious "Positions Closed" alert."""
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger, order_manager
    monkeypatch.setattr(order_manager, "STATE_FILE", str(tmp_path / "ps.json"))
    order_manager.save_position_state({"HON": {"qty_remaining": 10, "entry_price": 200.0,
                                               "initial_stop": 190.0, "trail_stop": 195.0}})
    activities = [
        _activity("a1", "HON", "buy", 10, 180.0, "2026-06-01T10:00:00Z", order_id="o1"),
        _activity("a2", "HON", "sell", 10, 170.0, "2026-06-10T10:00:00Z", order_id="o-oco"),
        _activity("a3", "HON", "buy", 10, 200.0, "2026-07-01T10:00:00Z", order_id="o2"),
    ]
    with patch("execution.alpaca.get_activities", return_value=activities), \
         patch("execution.alpaca.get_order", return_value={"client_order_id": "oco-HON-sell-2026-06-05"}), \
         patch("execution.alpaca.get_positions", return_value=[{"symbol": "HON", "qty": "10"}]):
        result = ledger.reconcile_fills()
    assert result["broker_exits"] == [], "held position reported as broker exit"
    assert order_manager.load_position_state()["HON"]["trail_stop"] == 195.0


def test_boundary_row_refetched_next_run_does_not_rerecord(tmp_path, monkeypatch):
    """Reviewer probe (C1b). Alpaca's `after` is inclusive on the boundary row,
    so the last stop fill comes back on every later run. It is not new — it
    must not be re-detected, re-recorded or re-alerted."""
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger, order_manager
    monkeypatch.setattr(order_manager, "STATE_FILE", str(tmp_path / "ps.json"))
    order_manager.save_position_state({"AAPL": {"qty_remaining": 10, "entry_price": 100.0}})
    acts = [_activity("a2", "AAPL", "sell", 10, 90.0, "2026-06-05T10:00:00Z", order_id="o-stop")]
    with patch("execution.alpaca.get_activities", return_value=acts), \
         patch("execution.alpaca.get_order", return_value={"client_order_id": "bstop-AAPL-90.00-10-x"}), \
         patch("execution.alpaca.get_positions", return_value=[]):
        r1 = ledger.reconcile_fills()
        # Second run: Alpaca returns the boundary row again (inclusive `after`)
        r2 = ledger.reconcile_fills()
    assert len(r1["broker_exits"]) == 1
    assert r2["broker_exits"] == []
    with dbmod._conn() as c:
        n = c.execute("select count(*) from exits").fetchone()[0]
    assert n == 1


def test_reconcile_fills_stop_fill_on_open_name_is_a_reentry_not_an_exit(tmp_path, monkeypatch):
    """A NEW bstop fill for a name get_positions() still lists is a re-entry
    (stopped and bought back inside the window), not an exit."""
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger, order_manager
    monkeypatch.setattr(order_manager, "STATE_FILE", str(tmp_path / "ps.json"))
    order_manager.save_position_state({"JPM": {"qty_remaining": 12, "entry_price": 330.0}})
    acts = [
        _activity("a1", "JPM", "sell", 12, 300.0, "2026-09-10T10:00:00Z", order_id="o-stop"),
        _activity("a2", "JPM", "buy", 12, 330.0, "2026-09-14T10:00:00Z", order_id="o-buy"),
    ]
    with patch("execution.alpaca.get_activities", return_value=acts), \
         patch("execution.alpaca.get_order", return_value={"client_order_id": "bstop-JPM-300.00-12-x"}), \
         patch("execution.alpaca.get_positions", return_value=[{"symbol": "JPM", "qty": "12"}]):
        result = ledger.reconcile_fills()
    assert result["broker_exits"] == []
    assert "JPM" in order_manager.load_position_state()


def test_reconcile_fills_new_stop_fill_still_detected_when_positions_unavailable(tmp_path, monkeypatch):
    """Signal (1) does not depend on get_positions(): a fresh stop fill with
    no later buy is an exit even when the broker position read failed."""
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger, order_manager
    monkeypatch.setattr(order_manager, "STATE_FILE", str(tmp_path / "ps.json"))
    order_manager.save_position_state({"AAPL": {"qty_remaining": 10, "entry_price": 100.0}})
    acts = [_activity("a2", "AAPL", "sell", 10, 90.0, "2026-06-05T10:00:00Z", order_id="o-stop")]
    with patch("execution.alpaca.get_activities", return_value=acts), \
         patch("execution.alpaca.get_order", return_value={"client_order_id": "bstop-AAPL-90.00-10-x"}), \
         patch("execution.alpaca.get_positions", side_effect=RuntimeError("timeout")):
        result = ledger.reconcile_fills()
    assert [e["ticker"] for e in result["broker_exits"]] == ["AAPL"]
    assert result["broker_exits"][0]["reason"] == "BROKER_STOP"
    assert order_manager.load_position_state() == {}


def test_reconcile_fills_dry_run_reports_only_new_fills(tmp_path, monkeypatch):
    """dry_run must report what a real run would find: the boundary row is
    already on record, so it is neither a new fill nor a broker exit."""
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger, order_manager
    monkeypatch.setattr(order_manager, "STATE_FILE", str(tmp_path / "ps.json"))
    dbmod.insert_fills([_fill("a1", "AAPL", "sell", 10, 90.0, "2026-06-05T10:00:00Z", tag="bstop")])
    acts = [
        _activity("a1", "AAPL", "sell", 10, 90.0, "2026-06-05T10:00:00Z", order_id="o-stop"),
        _activity("a2", "MSFT", "buy", 5, 400.0, "2026-06-06T10:00:00Z", order_id="o-buy"),
    ]
    with patch("execution.alpaca.get_activities", return_value=acts), \
         patch("execution.alpaca.get_order", return_value={"client_order_id": "bstop-AAPL-90.00-10-x"}), \
         patch("execution.alpaca.get_positions", return_value=[]):
        result = ledger.reconcile_fills(dry_run=True)
    assert result["new_fills"] == 1
    assert result["broker_exits"] == []
    assert len(dbmod.load_fills()) == 1   # nothing written


# ── I5: legacy OCO legs ───────────────────────────────────────────────────────

def test_reconcile_fills_legacy_oco_sell_on_closed_name_is_a_broker_exit(tmp_path, monkeypatch):
    """First live run: a stale state record whose last sell was an OCO leg
    (stop or take-profit — unknowable from the tag) is reconciled as a broker
    exit with reason LEGACY_OCO, never BROKER_STOP."""
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger, order_manager
    monkeypatch.setattr(order_manager, "STATE_FILE", str(tmp_path / "ps.json"))
    order_manager.save_position_state({"HON": {"qty_remaining": 10, "entry_price": 180.0}})
    acts = [
        _activity("a1", "HON", "buy", 10, 180.0, "2026-06-01T10:00:00Z", order_id="o1"),
        _activity("a2", "HON", "sell", 10, 200.0, "2026-06-10T10:00:00Z", order_id="o-oco"),
    ]
    with patch("execution.alpaca.get_activities", return_value=acts), \
         patch("execution.alpaca.get_order", return_value={"client_order_id": "oco-HON-sell-2026-06-05"}), \
         patch("execution.alpaca.get_positions", return_value=[]):
        result = ledger.reconcile_fills()
    assert len(result["broker_exits"]) == 1
    assert result["broker_exits"][0]["reason"] == "LEGACY_OCO"
    assert order_manager.load_position_state() == {}
    assert dbmod.recent_stop_exits(365, today=date(2026, 6, 11)) == {}, "LEGACY_OCO never cools down"


# ── broker_exits rows carry the realized pnl_pct (Telegram renders it) ───────

def test_reconcile_fills_broker_exit_carries_pnl_pct_from_trades(tmp_path, monkeypatch):
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger, order_manager
    monkeypatch.setattr(order_manager, "STATE_FILE", str(tmp_path / "ps.json"))
    order_manager.save_position_state({"AAPL": {"qty_remaining": 10, "entry_price": 100.0}})
    acts = [
        _activity("a1", "AAPL", "buy", 10, 100.0, "2026-06-01T10:00:00Z", order_id="o-buy"),
        _activity("a2", "AAPL", "sell", 10, 90.0, "2026-06-05T10:00:00Z", order_id="o-stop"),
    ]
    with patch("execution.alpaca.get_activities", return_value=acts), \
         patch("execution.alpaca.get_order", return_value={"client_order_id": "bstop-AAPL-90.00-10-x"}), \
         patch("execution.alpaca.get_positions", return_value=[]):
        result = ledger.reconcile_fills()
    assert result["broker_exits"][0]["pnl_pct"] == pytest.approx(-0.10)


def test_reconcile_fills_pnl_pct_is_none_without_a_matched_trade(tmp_path, monkeypatch):
    """A vanished position with no sell fill in the window (signal 2 only)
    has nothing to score against — None, not a fake 0.0."""
    _fresh_db(tmp_path, monkeypatch)
    from execution import ledger, order_manager
    monkeypatch.setattr(order_manager, "STATE_FILE", str(tmp_path / "ps.json"))
    order_manager.save_position_state({"GOOGL": {"qty_remaining": 4, "entry_price": 150.0}})
    with patch("execution.alpaca.get_activities", return_value=[]), \
         patch("execution.alpaca.get_positions", return_value=[]):
        result = ledger.reconcile_fills()
    assert result["broker_exits"][0]["reason"] == "UNKNOWN"
    assert result["broker_exits"][0]["pnl_pct"] is None
