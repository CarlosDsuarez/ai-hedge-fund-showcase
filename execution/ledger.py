"""execution/ledger.py — reconciles the broker's own fill history against
local state, so realized P&L is finally measurable.

Before this module existed there was no record of what actually happened to a
position once Alpaca closed it on its own: an OCO stop firing, or a manual
`DELETE /v2/positions/{sym}`, left `logs/position_state.json` claiming a
position that no longer existed, so those exits were never reconciled.

`reconcile_fills()` is the single entry point: pull fills since the last
reconciliation, insert them (idempotent), rebuild the FIFO trades table from
scratch, and detect any position that is gone from the broker but still
tracked locally — the definition of a "broker exit" this module exists to
catch. It runs at the very top of main.run_exits(), before get_positions(),
because a broker-side stop changes what "open positions" even means for every
check that follows in that cycle.

Never raises: a broken reconciliation must not take down the exit cycle that
depends on it running first.
"""
from datetime import datetime

import config
from db import database as db

# Tag → exit reason, parsed from the first '-'-delimited token of a sell
# fill's client_order_id (the format _client_order_id(), submit_stop_order()
# and replace_order() all produce: "{tag}-{symbol}-...").  Alpaca-generated
# ids — a UUID — carry no tag at all and fall through to UNKNOWN, same as any
# tag not listed here. Alpaca assigns one to: the legs of an OCO/bracket
# (the parent carries ours, each leg gets its own), a `DELETE
# /v2/positions/{sym}` close, and any PATCH replacement sent without a
# client_order_id (replace_order used to — 179 of the first 182 sell fills
# reconciled on 2026-09-21 were untagged). UNKNOWN therefore means "no
# reason on record", not "the pipeline lost track": the fill, price and
# P&L are real.
#
# "oco" is the pre-Sprint-1 bracket: EITHER leg (the stop or the take-profit)
# fills under that tag, so the fill cannot say whether it was a loss or a
# win. It is therefore its own reason, LEGACY_OCO — never BROKER_STOP, which
# would put every legacy winner into the re-entry cooldown
# (db.recent_stop_exits counts STOP_INITIAL / STOP_TRAIL / BROKER_STOP only).
EXIT_REASON_BY_TAG = {
    "bstop": "BROKER_STOP",
    "oco": "LEGACY_OCO",
    "stop_initial": "STOP_INITIAL",
    "stop_trail": "STOP_TRAIL",
    "rank_decay": "RANK_DECAY",
    "pm_sell": "PM_SELL",
    "time_cap": "TIME_CAP",
    "drift": "DRIFT_TRIM",
    "main": "LEGACY",
}


def _tag_from_client_order_id(client_order_id: "str | None") -> "str | None":
    """First token of a client_order_id, or None when there isn't one."""
    if not client_order_id:
        return None
    return client_order_id.split("-", 1)[0]


def _fills_from_activities(activities: list[dict], alpaca) -> list[dict]:
    """Normalize raw /v2/account/activities/FILL rows into fills-table shape.

    Rows carry no client_order_id, so a sell fill's tag needs one
    get_order(order_id) round trip — cached per order_id since a partially
    filled order produces several activity rows for the same order.
    """
    order_tags: dict[str, str | None] = {}
    fills = []
    for act in activities:
        side = (act.get("side") or "").lower()
        order_id = act.get("order_id")
        client_order_id = None
        if side == "sell" and order_id:
            if order_id not in order_tags:
                try:
                    order = alpaca.get_order(order_id)
                    order_tags[order_id] = order.get("client_order_id")
                except Exception:
                    order_tags[order_id] = None
            client_order_id = order_tags[order_id]
        fills.append({
            "id": act.get("id"),
            "symbol": (act.get("symbol") or "").replace(".", "-"),
            "side": side,
            "qty": float(act.get("qty") or 0),
            "price": float(act.get("price") or 0),
            "transaction_time": act.get("transaction_time"),
            "order_id": order_id,
            "client_order_id": client_order_id,
            "tag": _tag_from_client_order_id(client_order_id),
        })
    return fills


# Sell-fill reasons that mean "the broker closed this on its own": the
# engine's GTC stop, or a leg of the pre-Sprint-1 OCO bracket.
_BROKER_SIDE_REASONS = ("BROKER_STOP", "LEGACY_OCO")


def _last_fill_by_symbol(fills: list[dict], side: "str | None" = None) -> dict:
    """{symbol: its latest fill} over `fills`, optionally only one side."""
    latest: dict[str, dict] = {}
    for f in fills:
        if side and f["side"] != side:
            continue
        prev = latest.get(f["symbol"])
        if not prev or (f["transaction_time"] or "") >= (prev["transaction_time"] or ""):
            latest[f["symbol"]] = f
    return latest


def _detect_broker_exits(new_fills: list[dict], state: dict, open_symbols: set,
                         positions_available: bool = True,
                         window_fills: "list[dict] | None" = None) -> list[dict]:
    """Symbols the broker closed that the pipeline never recorded.

    Two independent signals, either one is enough: (1) among the fills that
    are NEW this cycle, the symbol's most recent fill is a sell tagged
    BROKER_STOP or LEGACY_OCO (the GTC stop or a legacy OCO leg fired), or
    (2) a symbol still tracked in position_state.json is simply absent from
    get_positions() — the DELETE /v2/positions path generates a sell with no
    tag at all, so it can never satisfy (1).

    `new_fills` is exactly what db.insert_fills inserted (or would insert):
    Alpaca's `after` is inclusive on the boundary row, so the previous run's
    last stop fill comes back on every later fetch, and running (1) over the
    whole window re-detected — and re-recorded, and re-alerted — the same
    exit every day. `window_fills` (the whole fetch, default `new_fills`) is
    only consulted for the qty/price/time detail of a signal-(2) exit.

    Signal (1) also requires the name NOT to be open: a stop fill on a
    symbol get_positions() still lists is a stop-and-re-buy inside the
    window (the live state file is full of them — HON 5x, JPM 6x), not an
    exit, and recording it would delete a live record's ladder-raised stop.
    Independently of the broker read, a buy fill AFTER the sell in the same
    window is the same re-entry and is skipped even when positions are
    unavailable — that is what protects the first backfill run.

    `positions_available=False` means get_positions() itself failed this
    cycle — signal (2) is then skipped entirely rather than computed against
    an empty open_symbols set, which would otherwise read as "every tracked
    position vanished" and fabricate an UNKNOWN exit (plus a state wipe) for
    the whole portfolio on a transient Alpaca timeout. Signal (1) is
    unaffected: it comes from `new_fills`, not from get_positions().

    Each row carries `fill_id` (the matched sell fill's activity id, None
    for a signal-(2)-only exit) so reconcile_fills can attach the realized
    pnl_pct from the trades table after rebuild.
    """
    window_fills = window_fills if window_fills is not None else new_fills
    last_new_fill = _last_fill_by_symbol(new_fills)
    broker_stop_symbols = {
        sym for sym, f in last_new_fill.items()
        if f["side"] == "sell"
        and EXIT_REASON_BY_TAG.get(f["tag"], "UNKNOWN") in _BROKER_SIDE_REASONS
        and not (positions_available and sym in open_symbols)
    }
    missing_symbols = (set(state) - open_symbols) if positions_available else set()

    last_sell_by_symbol = _last_fill_by_symbol(window_fills, side="sell")
    exits = []
    for symbol in sorted(broker_stop_symbols | missing_symbols):
        f = last_sell_by_symbol.get(symbol)
        if f:
            reason = EXIT_REASON_BY_TAG.get(f["tag"], "UNKNOWN")
            qty, price, exit_time, fill_id = f["qty"], f["price"], f["transaction_time"], f["id"]
        else:
            record = state.get(symbol, {})
            reason = "UNKNOWN"
            qty = record.get("qty_remaining", 0)
            price = None
            exit_time = datetime.now().isoformat()
            fill_id = None
        exits.append({
            "ticker": symbol, "reason": reason, "qty": qty,
            "price": price if price is not None else 0.0,
            "exit_time": exit_time, "fill_id": fill_id,
        })
    return exits


def _realized_pnl_pct(exit_fill_id: "str | None", trades: list[dict]) -> "float | None":
    """Qty-weighted pnl_pct of the trades rows closed by `exit_fill_id`.

    A sell spanning two buy lots is two trades rows (db.rebuild_trades); the
    position's realized return is their qty-weighted blend. None when no row
    matched or none is scored (a pre-ledger position with NULL entry).
    """
    if not exit_fill_id:
        return None
    rows = [t for t in trades
            if t.get("exit_fill_id") == exit_fill_id and t.get("pnl_pct") is not None]
    total_qty = sum(float(t["qty"]) for t in rows)
    if total_qty <= 0:
        return None
    return round(sum(float(t["qty"]) * float(t["pnl_pct"]) for t in rows) / total_qty, 4)


def reconcile_fills(after: "str | None" = None, dry_run: bool = False) -> dict:
    """Pull broker fills since `after`, rebuild trades, surface broker exits.

    `after` defaults to db.last_fill_time(), falling back to
    config.LEDGER_BACKFILL_FROM on an empty ledger (first run — also the run
    that has to reconcile the 28 pre-existing unreconciled OCO exits).

    `dry_run=True` fetches and reports exactly what a real run would find —
    including the get_order() lookups, which are reads — but writes nothing:
    no fills inserted, no trades rebuilt, no exits recorded, no state file
    touched.

    Returns {"new_fills", "trades_rebuilt", "broker_exits"} on success, or
    the same shape with new_fills=0/trades_rebuilt=0/broker_exits=[] plus an
    "error" key on any failure. Never raises. Each broker_exits row carries
    "pnl_pct" — the realized return from the rebuilt trades table, or None
    when nothing matched (dry run, a vanished position with no sell fill in
    the window, a pre-ledger lot) — so the Telegram close line can say
    "n/a" instead of a made-up +0.0%.
    """
    try:
        from execution import alpaca
        from execution.order_manager import load_position_state, save_position_state

        since = after or db.last_fill_time() or config.LEDGER_BACKFILL_FROM
        activities = alpaca.get_activities("FILL", after=since)
        fills = _fills_from_activities(activities, alpaca)

        state = load_position_state()
        positions_available = True
        try:
            open_positions = alpaca.get_positions()
        except Exception as e:
            print(f"  [Ledger] get_positions failed ({e}) — skipping the "
                  f"missing-from-broker exit signal this cycle; state untouched")
            open_positions = []
            positions_available = False
        open_symbols = {
            (p.get("symbol") or "").replace(".", "-") for p in open_positions
        }

        if dry_run:
            # Same subset a real run would insert (the boundary row and any
            # re-fetched history are already on record), without inserting.
            known_ids = {f["id"] for f in db.load_fills()}
            new_fills = [f for f in fills if f["id"] not in known_ids]
            broker_exits = _detect_broker_exits(new_fills, state, open_symbols,
                                                positions_available=positions_available,
                                                window_fills=fills)
            for exit_row in broker_exits:
                exit_row["pnl_pct"] = None   # trades are not rebuilt on a dry run
            result = {
                "new_fills": len(new_fills),
                "trades_rebuilt": 0,
                "broker_exits": broker_exits,
            }
            if not positions_available:
                result["positions_unavailable"] = True
            return result

        new_fills = db.insert_fills(fills)
        trades_rebuilt = db.rebuild_trades()

        # Signal (1) runs over the inserted subset only — see
        # _detect_broker_exits for why the whole window must not be used.
        broker_exits = _detect_broker_exits(new_fills, state, open_symbols,
                                            positions_available=positions_available,
                                            window_fills=fills)

        trades = db.load_trades() if broker_exits else []
        run_ts = datetime.now().isoformat()
        for exit_row in broker_exits:
            exit_row["pnl_pct"] = _realized_pnl_pct(exit_row.get("fill_id"), trades)
            db.record_exit(exit_row["ticker"], exit_row["exit_time"],
                           exit_row["reason"], exit_row["qty"],
                           exit_row["price"], run_ts)
            state.pop(exit_row["ticker"], None)
        if broker_exits:
            save_position_state(state)

        result = {
            "new_fills": len(new_fills),
            "trades_rebuilt": trades_rebuilt,
            "broker_exits": broker_exits,
        }
        if not positions_available:
            result["positions_unavailable"] = True
        return result
    except Exception as e:
        print(f"  [Ledger] reconcile_fills failed: {e}")
        return {"new_fills": 0, "trades_rebuilt": 0, "broker_exits": [], "error": str(e)}
