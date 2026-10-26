"""agents/risk_sl_tp_agent.run_stop_audit — one GTC broker stop per position.

The OCO audit this replaces computed its own SL/TP (3-8% / 10-20%) in
parallel with the software ladder, so a position had two exit policies that
disagreed. Now the broker order IS the engine's trailing level: placed when
missing, replaced when the chandelier rises, never a take-profit leg.

Decision table (design C, task 3):
  (a) no open sell orders                          → submit_stop_order
  (b) one resting stop at the wanted price and qty → protected, nothing sent
  (c) one resting stop that differs                → replace_order; on failure
      release + wait for the old order to die + submit fresh
  (d) anything else resting (legacy OCO, partials) → release + wait + submit
  want_stop >= price × 0.995                        → do NOT place (it would
      fire on arrival); flagged unprotected, the engine closes it in software
  get_open_orders() is None                         → fail closed, touch nothing
"""
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.risk_sl_tp_agent as agent
import execution.order_manager as om


def _pos(symbol, price, qty=10, qty_available=None, market_value=None):
    """A broker position row. qty_available is left for _audit() to derive
    from the resting orders (Alpaca: qty - held_for_orders) unless a test
    sets it explicitly — a fixture that says "10 available" next to a resting
    sell for 10 describes a broker state that cannot exist."""
    row = {
        "symbol": symbol, "current_price": str(price), "qty": str(qty),
        "avg_entry_price": "100.0",
        "market_value": str(price * qty if market_value is None else market_value),
    }
    if qty_available is not None:
        row["qty_available"] = str(qty_available)
    return row


def _held_for_orders(symbol, open_orders):
    return sum(int(float(o.get("qty") or 0)) for o in open_orders or []
               if o.get("symbol") == symbol and o.get("side") == "sell")


def _rec(trail=95.0, qty=10, **overrides):
    rec = {"entry_price": 100.0, "entry_time": "2026-09-01T10:00:00", "atr_14": 2.0,
           "initial_stop": 95.0, "trail_stop": trail, "highest_close": 100.0,
           "qty_total": qty, "qty_remaining": qty, "decay_count": 0}
    rec.update(overrides)
    return rec


def _stop_order(order_id="stop-1", symbol="AAPL", stop_price="95.00", qty="10",
                order_class="simple", status="new"):
    return {"id": order_id, "client_order_id": f"bstop-{symbol}", "symbol": symbol,
            "qty": qty, "side": "sell", "type": "stop", "stop_price": stop_price,
            "status": status, "order_class": order_class, "legs": None}


def _audit(positions, state, *, open_orders, exited=(), dry_run=False,
           submit=None, replace=None, waits=None):
    """Run the audit with every broker call recorded."""
    calls = {"submit": [], "replace": [], "replace_ticker": [], "release": [], "wait": []}
    submit = submit or (lambda t, q, p: {"id": f"new-{t}", "status": "accepted"})
    replace = replace or (lambda oid, *, ticker=None, stop_price, qty: {"id": f"repl-{oid}", "status": "accepted"})
    waits = waits or {}

    def _submit(t, q, p):
        calls["submit"].append((t, q, p))
        return submit(t, q, p)

    def _replace(oid, *, ticker=None, stop_price, qty):
        calls["replace"].append((oid, stop_price, qty))
        calls["replace_ticker"].append(ticker)
        return replace(oid, ticker=ticker, stop_price=stop_price, qty=qty)

    def _release(t):
        calls["release"].append(t)
        return [o["id"] for o in open_orders or [] if o["side"] == "sell"]

    def _wait(oid, timeout_s=30.0, poll_s=2.0):
        calls["wait"].append((oid, timeout_s, poll_s))
        return {"id": oid, "status": waits.get(oid, "canceled")}

    for p in positions:
        if "qty_available" not in p:
            held = _held_for_orders(p["symbol"], open_orders)
            p["qty_available"] = str(max(int(float(p["qty"])) - held, 0))

    om.save_position_state(state)
    with patch.object(agent, "get_open_orders", return_value=open_orders), \
         patch.object(agent, "submit_stop_order", side_effect=_submit), \
         patch.object(agent, "replace_order", side_effect=_replace), \
         patch.object(agent, "release_reserved_shares", side_effect=_release), \
         patch("execution.alpaca.wait_for_fill", side_effect=_wait):
        result = agent.run_stop_audit(positions, om.load_position_state(),
                                      exited=set(exited), dry_run=dry_run)
    return result, calls


def _names(result, bucket):
    return [r["ticker"] for r in result[bucket]]


# ── decision table ───────────────────────────────────────────────────────────

def test_no_open_sells_places_a_stop_at_the_trail_and_persists_the_order():
    result, calls = _audit([_pos("AAPL", 105.0)], {"AAPL": _rec(trail=97.456)}, open_orders=[])

    assert calls["submit"] == [("AAPL", 10, 97.46)]
    assert calls["replace"] == [] and calls["release"] == []
    assert _names(result, "placed") == ["AAPL"]
    rec = om.load_position_state()["AAPL"]
    assert rec["broker_stop_order_id"] == "new-AAPL"
    assert rec["broker_stop_price"] == pytest.approx(97.46)
    assert rec["broker_stop_qty"] == 10


def test_matching_resting_stop_is_left_alone():
    """The resting stop holds every share (qty_available=0 at Alpaca) — that
    is what "protected" looks like, not a reason to report the position bare."""
    resting = [_stop_order(stop_price="97.46", qty="10")]
    result, calls = _audit([_pos("AAPL", 105.0, qty_available=0)], {"AAPL": _rec(trail=97.456)},
                           open_orders=resting)

    assert calls == {"submit": [], "replace": [], "replace_ticker": [], "release": [], "wait": []}
    assert _names(result, "protected") == ["AAPL"]
    assert result["unprotected"] == []
    assert om.load_position_state()["AAPL"]["broker_stop_order_id"] == "stop-1"


def test_price_tolerance_is_half_a_cent():
    resting = [_stop_order(stop_price="97.45", qty="10")]
    result, _ = _audit([_pos("AAPL", 105.0)], {"AAPL": _rec(trail=97.454)}, open_orders=resting)
    assert _names(result, "protected") == ["AAPL"]


def test_risen_trail_replaces_the_resting_stop():
    """Reproduces the review finding: resting stop @95 for 10, qty_available=0,
    trail risen to 104 → the stop must be REPLACED, not reported unprotected."""
    resting = [_stop_order(order_id="stop-old", stop_price="95.00", qty="10")]
    result, calls = _audit([_pos("AAPL", 110.0, qty_available=0)], {"AAPL": _rec(trail=104.0)},
                           open_orders=resting)

    assert result["unprotected"] == []
    assert calls["replace"] == [("stop-old", 104.0, 10)]
    # The ticker rides along so the replacement carries a bstop client_order_id.
    assert calls["replace_ticker"] == ["AAPL"]
    assert calls["submit"] == []
    assert _names(result, "replaced") == ["AAPL"]
    assert om.load_position_state()["AAPL"]["broker_stop_order_id"] == "repl-stop-old", \
        "PATCH returns a NEW order id; that is the one to remember"


def test_changed_quantity_replaces_the_resting_stop():
    """A drift trim ran before the audit (the trim's release cancelled the
    stop, so 7 are available and nothing rests) — the stop must cover what is
    left. Modelled here as the stale stop still resting over 10 with 0
    available: Alpaca caps want_qty at the position (7), and the stop is
    resized, never left as-is."""
    resting = [_stop_order(order_id="stop-old", stop_price="95.00", qty="10")]
    result, calls = _audit([_pos("AAPL", 105.0, qty=7, qty_available=0)],
                           {"AAPL": _rec(trail=95.0, qty=7)}, open_orders=resting)
    assert calls["replace"] == [("stop-old", 95.0, 7)]
    assert _names(result, "replaced") == ["AAPL"]


def test_stop_at_or_above_market_is_not_placed():
    """A stop within 0.5% of the market would fire on arrival. The software
    engine sells such a position itself; the audit only reports it."""
    result, calls = _audit([_pos("AAPL", 95.2)], {"AAPL": _rec(trail=95.0)}, open_orders=[])

    assert calls["submit"] == []
    assert _names(result, "unprotected") == ["AAPL"]
    assert result["unprotected"][0]["reason"] == "stop at market — software close pending"


def test_legacy_oco_is_released_before_the_stop_is_placed():
    """An OCO holds the whole position (qty_available=0). The audit must
    reclaim those shares — release, wait, place for the full 10 — not report
    "nothing available" and leave the TP leg alive forever."""
    oco = [
        {"id": "oco-tp", "symbol": "AAPL", "qty": "10", "side": "sell", "type": "limit",
         "stop_price": None, "status": "new", "order_class": "oco"},
        {"id": "oco-sl", "symbol": "AAPL", "qty": "10", "side": "sell", "type": "stop",
         "stop_price": "92.00", "status": "held", "order_class": "oco"},
    ]
    result, calls = _audit([_pos("AAPL", 105.0, qty_available=0)], {"AAPL": _rec(trail=95.0)},
                           open_orders=oco)

    assert calls["release"] == ["AAPL"]
    assert {w[0] for w in calls["wait"]} == {"oco-tp", "oco-sl"}
    assert calls["replace"] == []
    assert calls["submit"] == [("AAPL", 10, 95.0)]
    assert _names(result, "placed") == ["AAPL"]


def test_single_oco_stop_leg_counts_as_legacy_not_as_ours():
    leg = [_stop_order(order_id="oco-sl", stop_price="95.00", qty="10", order_class="oco")]
    result, calls = _audit([_pos("AAPL", 105.0)], {"AAPL": _rec(trail=95.0)}, open_orders=leg)
    assert calls["release"] == ["AAPL"]
    assert calls["submit"] == [("AAPL", 10, 95.0)]


def test_patch_failure_falls_back_to_release_wait_submit():
    resting = [_stop_order(order_id="stop-old", stop_price="95.00", qty="10")]

    def _boom(oid, *, ticker=None, stop_price, qty):
        raise RuntimeError("422 order not replaceable")

    result, calls = _audit([_pos("AAPL", 110.0)], {"AAPL": _rec(trail=104.0)},
                           open_orders=resting, replace=_boom)

    assert calls["replace"] == [("stop-old", 104.0, 10)]
    assert calls["release"] == ["AAPL"]
    assert calls["wait"] == [("stop-old", 10, 1)]
    assert calls["submit"] == [("AAPL", 10, 104.0)]
    assert _names(result, "replaced") == ["AAPL"]
    assert om.load_position_state()["AAPL"]["broker_stop_order_id"] == "new-AAPL"


def test_old_stop_that_filled_during_the_replace_is_never_re_placed():
    """Placing a sell over shares the fired stop just sold would open a short."""
    resting = [_stop_order(order_id="stop-old", stop_price="95.00", qty="10")]

    def _boom(oid, *, ticker=None, stop_price, qty):
        raise RuntimeError("422")

    result, calls = _audit([_pos("AAPL", 110.0)], {"AAPL": _rec(trail=104.0)},
                           open_orders=resting, replace=_boom, waits={"stop-old": "filled"})

    assert calls["submit"] == []
    assert _names(result, "errors") == ["AAPL"]
    assert "filled" in result["errors"][0]["error"]


def test_old_order_still_pending_cancel_after_the_wait_blocks_the_submit():
    resting = [_stop_order(order_id="stop-old", stop_price="95.00", qty="10")]

    def _boom(oid, *, ticker=None, stop_price, qty):
        raise RuntimeError("422")

    result, calls = _audit([_pos("AAPL", 110.0)], {"AAPL": _rec(trail=104.0)},
                           open_orders=resting, replace=_boom,
                           waits={"stop-old": "pending_cancel"})
    assert calls["submit"] == [], "shares still reserved: a submit would 403"
    assert _names(result, "errors") == ["AAPL"]


# ── guards ───────────────────────────────────────────────────────────────────

def test_fails_closed_when_the_open_order_state_is_unknown():
    result, calls = _audit([_pos("AAPL", 105.0), _pos("MSFT", 400.0)],
                           {"AAPL": _rec(), "MSFT": _rec(trail=380.0)}, open_orders=None)

    assert calls == {"submit": [], "replace": [], "replace_ticker": [], "release": [], "wait": []}
    assert result["open_orders_unavailable"] is True
    assert sorted(_names(result, "errors")) == ["AAPL", "MSFT"]
    assert not result["placed"] and not result["protected"] and not result["replaced"]


def test_dry_run_places_nothing():
    result, calls = _audit([_pos("AAPL", 105.0)], {"AAPL": _rec(trail=95.0)},
                           open_orders=[], dry_run=True)
    assert calls == {"submit": [], "replace": [], "replace_ticker": [], "release": [], "wait": []}
    assert _names(result, "placed") == ["AAPL"]
    assert result["placed"][0]["dry_run"] is True
    assert "broker_stop_order_id" not in om.load_position_state()["AAPL"]


def test_positions_exited_this_cycle_are_skipped():
    """DELETE /v2/positions leaves the row visible with qty_available=0 until
    the liquidation fills; auditing it would drive a guaranteed 403."""
    result, calls = _audit([_pos("AAPL", 105.0), _pos("MSFT", 400.0)],
                           {"AAPL": _rec(), "MSFT": _rec(trail=380.0)},
                           open_orders=[], exited={"AAPL"})
    assert calls["submit"] == [("MSFT", 10, 380.0)]
    assert "AAPL" not in {t for b in result.values() if isinstance(b, list) for r in b for t in [r["ticker"]]}


def test_position_without_a_state_record_is_reported_not_guessed():
    result, calls = _audit([_pos("AAPL", 105.0)], {}, open_orders=[])
    assert calls["submit"] == []
    assert _names(result, "unprotected") == ["AAPL"]
    assert "state" in result["unprotected"][0]["reason"]


def test_nothing_available_and_nothing_resting_means_nothing_to_protect():
    """qty_available=0 with no resting sell of ours to reclaim (a liquidation
    the broker is still settling, seen through a stale positions snapshot)."""
    result, calls = _audit([_pos("AAPL", 105.0, qty_available=0)], {"AAPL": _rec()},
                           open_orders=[])
    assert calls["submit"] == [] and calls["release"] == []
    assert _names(result, "unprotected") == ["AAPL"]
    assert "no shares" in result["unprotected"][0]["reason"]


def test_partial_reservation_is_topped_up_to_the_full_position():
    """A resting stop for 7 with 3 more shares bought since: the replacement
    covers 7 (reclaimed) + 3 (available) = 10."""
    resting = [_stop_order(order_id="stop-old", stop_price="95.00", qty="7")]
    result, calls = _audit([_pos("AAPL", 105.0, qty=10, qty_available=3)], {"AAPL": _rec(trail=95.0)},
                           open_orders=resting)
    assert calls["replace"] == [("stop-old", 95.0, 10)]


def test_reclaimed_quantity_never_exceeds_the_position():
    """A stale resting stop for 10 over a position trimmed to 7 (0 available)."""
    resting = [_stop_order(order_id="stop-old", stop_price="95.00", qty="10")]
    result, calls = _audit([_pos("AAPL", 105.0, qty=7, qty_available=0)],
                           {"AAPL": _rec(trail=95.0, qty=7)}, open_orders=resting)
    assert calls["replace"] == [("stop-old", 95.0, 7)]


def test_shorts_are_ignored():
    result, calls = _audit([_pos("AAPL", 105.0, qty=-10, market_value=-1050.0)],
                           {"AAPL": _rec()}, open_orders=[])
    assert calls["submit"] == []
    assert all(not v for k, v in result.items() if isinstance(v, list))


def test_brk_b_matches_the_brokers_dotted_symbol():
    resting = [_stop_order(symbol="BRK.B", stop_price="380.00", qty="10")]
    result, calls = _audit([_pos("BRK.B", 400.0, qty_available=0)], {"BRK-B": _rec(trail=380.0)},
                           open_orders=resting)
    assert _names(result, "protected") == ["BRK-B"]
    assert calls["submit"] == []


def test_chat_status_answer_is_a_dry_run_and_names_each_position():
    """chat/executor's "risk_sl_tp" answer must report protection without
    placing anything — a status question is not an order."""
    from chat.executor import _format_stop_audit, _positions_for_stop_audit

    ctx = {"live_positions": [
        {"symbol": "BRK-B", "qty": 10, "current_price": 400.0, "market_value": 4000.0},
        {"symbol": "AAPL", "qty": 5, "current_price": 190.0, "market_value": 950.0},
    ]}
    positions = _positions_for_stop_audit(ctx)
    assert positions[0]["symbol"] == "BRK.B"
    state = {"BRK-B": _rec(trail=380.0), "AAPL": _rec(trail=182.5, decay_count=2)}
    resting = [_stop_order(symbol="BRK.B", stop_price="380.00", qty="10")]
    with patch.object(agent, "get_open_orders", return_value=resting), \
         patch.object(agent, "submit_stop_order") as submit, \
         patch.object(agent, "replace_order") as replace:
        result = agent.run_stop_audit(positions, state, exited=set(), dry_run=True)
        text = _format_stop_audit(result, state, positions, "es")

    submit.assert_not_called()
    replace.assert_not_called()
    assert "Protegidas: 1/2" in text
    assert "BRK-B" in text and "380.00" in text
    assert "AAPL" in text and "182.50" in text and "decay 2/5" in text
    assert "la corrida lo colocaría" in text


def test_find_broker_stop_ignores_other_symbols_and_buys():
    orders = [_stop_order(symbol="MSFT"),
              {**_stop_order(symbol="AAPL"), "side": "buy"},
              _stop_order(order_id="mine", symbol="AAPL")]
    found = agent.find_broker_stop("AAPL", orders)
    assert found["id"] == "mine"
    assert agent.find_broker_stop("NVDA", orders) is None
