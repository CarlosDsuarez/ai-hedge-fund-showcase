"""execution/order_manager — the single exit engine.

Initial stop STOP_ATR_MULT×ATR14 at entry, chandelier trail TRAIL_ATR_MULT×ATR14
off the highest close (ATR frozen at entry), no TP partials, rank decay after
RANK_DECAY_DAYS consecutive weak runs, a MAX_HOLDING_DAYS hard cap, PM SELL at
conviction >= PM_SELL_MIN_CONVICTION (values: config.py placeholders).

Position state lives in the DB now; logs/position_state.json is a mirror
(plus a one-time migration source). The suite-wide fixture in conftest.py
pins both to per-test temp files.
"""
import json
import os
import sys
from datetime import date
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import db.database as db
import execution.order_manager as om


def _pos(symbol, price, qty=10, avg_entry=None, market_value=None):
    """A position dict shaped like Alpaca's GET /v2/positions row."""
    return {
        "symbol": symbol, "current_price": str(price), "qty": str(qty),
        "qty_available": str(qty),
        "avg_entry_price": str(avg_entry if avg_entry is not None else price),
        "market_value": str(market_value if market_value is not None else price * qty),
        "unrealized_plpc": "0.0",
    }


def _record(entry=100.0, atr=2.0, qty=10, **overrides):
    rec = {
        "entry_price": entry, "entry_time": "2026-09-01T10:00:00", "atr_14": atr,
        "initial_stop": round(entry - config.STOP_ATR_MULT * atr, 2),
        "trail_stop": round(entry - config.STOP_ATR_MULT * atr, 2),
        "highest_close": entry, "qty_total": qty, "qty_remaining": qty,
        "decay_count": 0,
    }
    rec.update(overrides)
    return rec


# ── state: DB primary, JSON mirror, one-time migration ───────────────────────



def test_register_new_position_rejects_a_missing_atr():
    with pytest.raises(ValueError):
        om.register_new_position("AAPL", 100.0, 10, 0.0)


def test_save_mirrors_to_json_with_sl_price_for_legacy_readers():
    """dashboard/app.py, chat/context.py and the weekly report read sl_price."""
    om.save_position_state({"AAPL": _record(trail_stop=97.5)})

    mirrored = json.load(open(om.STATE_FILE))
    assert mirrored["AAPL"]["sl_price"] == pytest.approx(97.5)
    assert mirrored["AAPL"]["trail_stop"] == pytest.approx(97.5)
    assert "sl_price" not in db.load_position_state()["AAPL"], "the DB record stays clean"


def test_load_prefers_the_db_over_a_stale_json_file():
    om.save_position_state({"AAPL": _record()})
    with open(om.STATE_FILE, "w") as f:
        json.dump({"MSFT": _record(entry=400.0)}, f)   # stale mirror

    assert set(om.load_position_state()) == {"AAPL"}


def _legacy(**over):
    rec = {
        "entry_price": 100.0, "atr_14": 2.0, "sl_price": 97.0, "sl_is_breakeven": False,
        "tp1_price": 102.0, "tp1_filled": False, "tp2_price": 104.0, "tp2_filled": False,
        "tp3_price": 106.0, "tp3_filled": False, "qty_total": 10, "qty_remaining": 10,
        "entry_time": "2026-08-20T10:00:00",
    }
    rec.update(over)
    return rec






def test_migration_falls_back_to_sl_price_without_an_atr():
    with open(om.STATE_FILE, "w") as f:
        json.dump({"AAPL": _legacy(atr_14=0.0)}, f)
    rec = om.load_position_state()["AAPL"]
    assert rec["initial_stop"] == pytest.approx(97.0)
    assert rec["trail_stop"] == pytest.approx(97.0)






# ── pure helpers ─────────────────────────────────────────────────────────────



def test_update_trail_uses_the_atr_stored_at_entry():
    rec = _record(entry=100.0, atr=2.0)
    with patch("data.fetcher.get_atr", side_effect=AssertionError("must not refetch ATR")):
        out = om.update_trail(rec, 120.0)
    assert out["trail_stop"] == pytest.approx(114.0)




def test_rank_percentile_matches_pandas_pct_rank():
    scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    assert om.rank_percentile("A", scores) == pytest.approx(0.25)
    assert om.rank_percentile("D", scores) == pytest.approx(1.0)
    ties = {"A": 1.0, "B": 1.0, "C": 2.0, "D": 3.0}
    assert om.rank_percentile("A", ties) == pytest.approx(0.375)   # average rank of the tie
    assert om.rank_percentile("ZZZ", scores) is None
    assert om.rank_percentile("A", {}) is None


# ── run_exit_engine ──────────────────────────────────────────────────────────

def _engine(positions, *, state=None, score_map=None, pm_sells=None, metadata=None,
            sell=(10, 0.0), today=None, dry_run=False):
    """Drive the engine with the broker and the price feed stubbed out."""
    if state is not None:
        om.save_position_state(state)
    sells = []

    def _sell(ticker, qty, *, tag="main"):
        sells.append({"ticker": ticker, "qty": qty, "tag": tag})
        filled, price = sell
        return (min(filled, qty), price) if filled else (0, 0.0)

    with patch("execution.trade_executor.sell_and_confirm", side_effect=_sell), \
         patch("data.fetcher.get_atr", return_value=2.0):
        exited, closed = om.run_exit_engine(
            positions, score_map=score_map or {}, pm_sells=pm_sells or {},
            metadata=metadata or {}, today=today or date(2026, 9, 16), dry_run=dry_run)
    return exited, closed, sells


def test_stop_initial_sells_the_whole_position_with_the_reason_tag():
    state = {"AAPL": _record(entry=100.0, atr=2.0)}          # initial = trail = 95
    exited, closed, sells = _engine([_pos("AAPL", 94.0, qty=10)], state=state, sell=(10, 93.9))

    assert exited == {"AAPL"}
    assert sells == [{"ticker": "AAPL", "qty": 10, "tag": "stop_initial"}]
    assert closed == [{"ticker": "AAPL", "reason": "STOP_INITIAL", "qty": 10,
                       "price": 93.9, "pnl_pct": pytest.approx(-0.061)}]
    assert om.load_position_state() == {}, "the record is deleted after the fill"


def test_stop_trail_label_when_the_chandelier_has_risen():
    state = {"AAPL": _record(entry=100.0, atr=2.0, highest_close=120.0, trail_stop=114.0)}
    exited, closed, sells = _engine([_pos("AAPL", 113.5)], state=state, sell=(10, 113.4))

    assert closed[0]["reason"] == "STOP_TRAIL"
    assert sells[0]["tag"] == "stop_trail"
    assert closed[0]["pnl_pct"] == pytest.approx(0.134)


def test_engine_ratchets_the_trail_and_persists_it():
    state = {"AAPL": _record(entry=100.0, atr=2.0)}
    exited, closed, sells = _engine([_pos("AAPL", 110.0)], state=state)

    assert exited == set() and sells == []
    rec = om.load_position_state()["AAPL"]
    assert rec["highest_close"] == pytest.approx(110.0)
    assert rec["trail_stop"] == pytest.approx(104.0)
    assert json.load(open(om.STATE_FILE))["AAPL"]["sl_price"] == pytest.approx(104.0)


def test_exit_writes_an_exits_row():
    state = {"AAPL": _record(entry=100.0, atr=2.0)}
    _engine([_pos("AAPL", 94.0)], state=state, sell=(10, 94.0))

    with db._conn() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM exits").fetchall()]
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["reason"] == "STOP_INITIAL"
    assert rows[0]["qty"] == 10 and rows[0]["price"] == 94.0
    assert rows[0]["run_timestamp"]


def test_unfilled_sell_keeps_the_position_and_its_state():
    state = {"AAPL": _record(entry=100.0, atr=2.0)}
    exited, closed, sells = _engine([_pos("AAPL", 94.0)], state=state, sell=(0, 0.0))

    assert sells, "the sell was attempted"
    assert exited == set() and closed == []
    assert "AAPL" in om.load_position_state()
    with db._conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM exits").fetchone()[0] == 0




def test_pm_sell_executes_at_conviction_4_not_3():
    state = {"AAPL": _record(entry=100.0, atr=2.0)}
    exited, closed, sells = _engine([_pos("AAPL", 105.0)], state=state,
                                    pm_sells={"AAPL": 3})
    assert sells == [] and exited == set()

    state = {"AAPL": _record(entry=100.0, atr=2.0)}
    exited, closed, sells = _engine([_pos("AAPL", 105.0)], state=state,
                                    pm_sells={"AAPL": 4}, sell=(10, 105.0))
    assert closed[0]["reason"] == "PM_SELL"
    assert sells[0]["tag"] == "pm_sell"


def test_pm_sell_on_a_ticker_not_held_sells_nothing():
    state = {"AAPL": _record(entry=100.0, atr=2.0)}
    exited, closed, sells = _engine([_pos("AAPL", 105.0)], state=state,
                                    pm_sells={"MSFT": 5})
    assert sells == [] and exited == set()


def test_rank_decay_counts_resets_and_fires_at_5():
    weak = {"AAPL": -1.0, "B": 0.5, "C": 0.7, "D": 0.9}    # AAPL pctl 0.25 < 0.50
    strong = {"AAPL": 2.0, "B": 0.5, "C": 0.7, "D": 0.9}   # AAPL pctl 1.00

    om.save_position_state({"AAPL": _record(entry=100.0, atr=2.0)})
    for n in range(1, 5):
        exited, closed, sells = _engine([_pos("AAPL", 105.0)], score_map=weak)
        assert sells == [], f"run {n}: decay must not fire before {config.RANK_DECAY_DAYS}"
        assert om.load_position_state()["AAPL"]["decay_count"] == n

    _engine([_pos("AAPL", 105.0)], score_map=strong)              # one strong run resets
    assert om.load_position_state()["AAPL"]["decay_count"] == 0

    for _ in range(4):
        _engine([_pos("AAPL", 105.0)], score_map=weak)
    exited, closed, sells = _engine([_pos("AAPL", 105.0)], score_map=weak, sell=(10, 105.0))
    assert closed[0]["reason"] == "RANK_DECAY"
    assert sells[0]["tag"] == "rank_decay"


def test_rank_decay_is_skipped_on_an_empty_or_all_zero_score_map():
    """A failed strategy run scores everything 0.0; that is not five weak days."""
    om.save_position_state({"AAPL": _record(entry=100.0, atr=2.0, decay_count=3)})
    for score_map in ({}, None, {"AAPL": 0.0, "B": 0.0, "C": 0.0}):
        exited, closed, sells = _engine([_pos("AAPL", 105.0)], score_map=score_map)
        assert sells == []
        assert om.load_position_state()["AAPL"]["decay_count"] == 3, f"counter touched on {score_map!r}"


def test_stop_outranks_pm_sell_and_decay_and_only_one_action_fires():
    state = {"AAPL": _record(entry=100.0, atr=2.0, decay_count=4)}
    weak = {"AAPL": -1.0, "B": 0.5, "C": 0.7, "D": 0.9}
    exited, closed, sells = _engine([_pos("AAPL", 94.0)], state=state, score_map=weak,
                                    pm_sells={"AAPL": 5}, sell=(10, 94.0))
    assert len(sells) == 1
    assert closed[0]["reason"] == "STOP_INITIAL"




def test_stateless_position_below_the_initial_atr_stop_is_sold():
    """Entry 100 with a stored ATR: a price below entry − STOP_ATR_MULT×ATR sells."""
    exited, closed, sells = _engine([_pos("AAPL", 94.0, avg_entry=100.0)], sell=(10, 94.0))
    assert closed[0]["reason"] == "STOP_INITIAL"


def test_stateless_position_without_atr_is_left_alone():
    with patch("execution.trade_executor.sell_and_confirm") as sell, \
         patch("data.fetcher.get_atr", return_value=0.0):
        exited, closed = om.run_exit_engine([_pos("AAPL", 50.0, avg_entry=100.0)],
                                            score_map={}, pm_sells={}, metadata={})
    sell.assert_not_called()
    assert exited == set()
    assert om.load_position_state() == {}


def test_short_positions_are_ignored():
    state = {"AAPL": _record(entry=100.0, atr=2.0)}
    short = _pos("AAPL", 94.0, qty=-10, market_value=-940.0)
    exited, closed, sells = _engine([short], state=state)
    assert sells == [] and exited == set()




def test_state_is_persisted_after_every_mutation():
    """A crash after a broker close must not resurrect the position from
    stale disk state (tests/test_alpaca_retry_fills pinned this for the
    ladder; the engine keeps the property)."""
    state = {"AAPL": _record(entry=100.0, atr=2.0), "MSFT": _record(entry=500.0, atr=10.0)}
    om.save_position_state(state)
    saves = []
    with patch.object(om, "save_position_state", side_effect=lambda s: saves.append(dict(s))), \
         patch("execution.trade_executor.sell_and_confirm", return_value=(10, 94.0)), \
         patch("data.fetcher.get_atr", return_value=2.0):
        exited, closed = om.run_exit_engine(
            [_pos("AAPL", 94.0), _pos("MSFT", 500.0)],
            score_map={}, pm_sells={}, metadata={}, today=date(2026, 9, 16))

    assert exited == {"AAPL"}
    assert "AAPL" not in saves[0], "first save (right after the AAPL fill) already lacks AAPL"
    assert "MSFT" in saves[0]


def test_a_raise_on_one_ticker_does_not_discard_the_sells_that_already_happened():
    """Fault isolation per position: MSFT's row is malformed (current_price
    is not a number) — AAPL's fill, already executed at the broker, must still
    come back in (exited, closed) so it reaches the alerts, run_data and the
    stop audit's exclusion set; NVDA after it must still be evaluated."""
    state = {"AAPL": _record(entry=100.0, atr=2.0),
             "MSFT": _record(entry=500.0, atr=10.0),
             "NVDA": _record(entry=200.0, atr=4.0)}
    broken = {**_pos("MSFT", 480.0), "current_price": "not-a-price"}
    exited, closed, sells = _engine([_pos("AAPL", 94.0), broken, _pos("NVDA", 189.0)],
                                    state=state, sell=(10, 94.0))

    assert exited == {"AAPL", "NVDA"}
    assert [c["ticker"] for c in closed] == ["AAPL", "NVDA"]
    assert [s["ticker"] for s in sells] == ["AAPL", "NVDA"]
    assert "MSFT" in om.load_position_state(), "the failed ticker's state is untouched"


def test_brk_b_symbol_is_normalised_from_the_broker():
    state = {"BRK-B": _record(entry=400.0, atr=8.0)}
    exited, closed, sells = _engine([_pos("BRK.B", 379.0)], state=state, sell=(10, 379.0))
    assert exited == {"BRK-B"}
    assert sells[0]["ticker"] == "BRK-B"
