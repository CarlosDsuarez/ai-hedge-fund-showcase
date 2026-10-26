"""Broker stop audit: one GTC stop per position, at the exit engine's level.

The OCO audit this file used to hold computed its own SL (3-8%) and TP
(10-20%) from ATR-20 and beta, in parallel with the software TP ladder — two
exit policies per position that disagreed with each other, and a broker TP
that sold winners the moment they reached +10%. Design C keeps exactly one
exit policy: execution/order_manager.run_exit_engine owns the level, and the
broker order placed here is a copy of it so a gap-down between runs is
still caught. No take-profit leg, ever.

The audit never decides a level. It reads trail_stop from the position
record and reconciles the broker's open orders to it: place when nothing
rests, leave alone when the resting stop matches, replace when the
chandelier rose (PATCH, then cancel + fresh submit if the PATCH is refused),
and clear out anything else that reserves the shares (a legacy OCO, a
partial close) before placing. It fails closed when the open-order state is
unknown: acting blind is how a second sell gets a 403 — or a fill.

Module-level imports from execution.alpaca on purpose: the suite patches
`agents.risk_sl_tp_agent.<name>` by attribute.
"""
from execution.alpaca import (
    get_open_orders,
    release_reserved_shares,
    replace_order,
    submit_stop_order,
    wait_until_dead,
)
from execution.order_manager import save_position_state

# Within this fraction of the market a GTC stop would trigger on arrival
# (Alpaca rejects or fills it immediately). The software engine sells such a
# position itself on its next pass; the audit only reports it.
IMMEDIATE_TRIGGER_BAND = 0.995

# The resting order is considered "ours and current" when its stop price is
# within half a cent of the wanted level (Alpaca quotes to the cent).
STOP_PRICE_TOLERANCE = 0.005

_WAIT_TIMEOUT_S = 10
_WAIT_POLL_S = 1


def _open_sells(alpaca_symbol: str, open_orders: list) -> list:
    return [o for o in open_orders
            if o.get("symbol") == alpaca_symbol and o.get("side") == "sell"]


def find_broker_stop(ticker: str, open_orders: list) -> "dict | None":
    """The single plain GTC stop resting on `ticker`, or None.

    An OCO's stop leg carries order_class "oco" and is NOT the engine's
    order: it cannot be PATCHed on its own and its sibling TP leg still
    reserves the shares. It is treated as legacy and cleared, not adopted.
    """
    alpaca_symbol = ticker.replace("-", ".")
    for o in _open_sells(alpaca_symbol, open_orders):
        if o.get("type") == "stop" and (o.get("order_class") or "simple") != "oco":
            return o
    return None


def _matches(order: dict, want_stop: float, want_qty: int) -> bool:
    try:
        price_ok = abs(float(order.get("stop_price") or 0) - want_stop) <= STOP_PRICE_TOLERANCE
        qty_ok = int(float(order.get("qty") or 0)) == want_qty
    except (TypeError, ValueError):
        return False
    return price_ok and qty_ok


def _wait_until_dead(order_ids: list) -> "str | None":
    """Wait for the cancelled orders' shares to be released (execution.alpaca.
    wait_until_dead). Returns the reason the caller must NOT place a new
    stop — a fill (the position is gone; selling again would open a short),
    an order still holding shares, or a read error — or None when clear."""
    released = wait_until_dead(order_ids, timeout_s=_WAIT_TIMEOUT_S, poll_s=_WAIT_POLL_S)
    if released["filled"]:
        return (f"{released['filled'][0]} filled during the replace — the position is gone; "
                f"ledger will reconcile")
    if released["pending"]:
        oid, status = released["pending"][0]
        return f"{oid} still {status}"
    if released["errors"]:
        oid, err = released["errors"][0]
        return f"{oid}: {err}"
    return None


def run_stop_audit(positions: list, state: dict, *, exited: set,
                   dry_run: bool = False) -> dict:
    """Reconcile every long position's broker stop to its engine trail level.

    positions : Alpaca GET /v2/positions rows.
    state     : {ticker: record} from order_manager.load_position_state();
                mutated in place and persisted whenever a broker order is
                placed or replaced (broker_stop_order_id/price/qty).
    exited    : tickers the engine sold this cycle. They are still listed by
                the broker until the liquidation fills, with every share
                held_for_orders — auditing one is a guaranteed 403.

    Returns {"protected", "placed", "replaced", "unprotected", "errors"}
    (lists of {"ticker", ...}), plus "open_orders_unavailable": True when the
    audit failed closed.
    """
    result: dict = {"protected": [], "placed": [], "replaced": [], "unprotected": [], "errors": []}

    longs = [
        p for p in positions
        if float(p.get("market_value") or 0) > 0
        and (p.get("symbol") or "").replace(".", "-") not in exited
    ]

    open_orders = get_open_orders()
    if open_orders is None:
        print("  [Stop] Open-order state unavailable — audit skipped (fail-closed)")
        result["open_orders_unavailable"] = True
        for p in longs:
            result["errors"].append({"ticker": (p.get("symbol") or "").replace(".", "-"),
                                     "error": "open-order state unknown"})
        return result

    for p in longs:
        alpaca_symbol = p.get("symbol") or ""
        ticker = alpaca_symbol.replace(".", "-")
        rec = state.get(ticker)
        price = float(p.get("current_price") or 0)
        if not rec or not rec.get("trail_stop"):
            result["unprotected"].append({"ticker": ticker, "reason": "no state record"})
            continue
        want_stop = round(float(rec["trail_stop"]), 2)
        sells = _open_sells(alpaca_symbol, open_orders)
        resting = find_broker_stop(ticker, open_orders)

        # Alpaca reports qty_available = qty - held_for_orders, and a resting
        # GTC sell stop (or a legacy OCO) holds the WHOLE position — so on any
        # day after the first placement qty_available is 0 for every protected
        # name. The shares those resting sells hold are exactly the ones this
        # audit is about to keep, PATCH or cancel, so they count as ours to
        # protect: want_qty = available + reclaimable, capped at the position.
        qty_total = int(float(p.get("qty") or 0))
        qty_available = int(float(p.get("qty_available", qty_total) or 0))
        reclaimable = sum(int(float(o.get("qty") or 0)) for o in sells)
        want_qty = min(qty_available + reclaimable, qty_total)
        if want_qty <= 0:
            result["unprotected"].append({
                "ticker": ticker, "stop": want_stop,
                "reason": "no shares available and nothing of ours resting to reclaim"})
            continue
        if price > 0 and want_stop >= price * IMMEDIATE_TRIGGER_BAND:
            result["unprotected"].append({
                "ticker": ticker, "stop": want_stop, "price": price,
                "reason": "stop at market — software close pending"})
            continue

        try:
            if not sells:                                             # (a)
                order = _place(ticker, want_qty, want_stop, dry_run)
                _remember(state, ticker, rec, order, want_stop, want_qty, dry_run)
                result["placed"].append(_row(ticker, want_stop, want_qty, order, dry_run))
                continue

            if len(sells) == 1 and resting is not None:
                if _matches(resting, want_stop, want_qty):                # (b)
                    _remember(state, ticker, rec, resting, want_stop, want_qty, dry_run)
                    result["protected"].append(_row(ticker, want_stop, want_qty, resting, dry_run))
                    continue
                order, via = _replace(ticker, resting, want_qty, want_stop, dry_run)   # (c)
                _remember(state, ticker, rec, order, want_stop, want_qty, dry_run)
                result["replaced"].append({**_row(ticker, want_stop, want_qty, order, dry_run),
                                           "replaced_order_id": resting.get("id"), "via": via})
                continue

            order = _clear_and_place(ticker, sells, want_qty, want_stop, dry_run)     # (d)
            _remember(state, ticker, rec, order, want_stop, want_qty, dry_run)
            result["placed"].append({**_row(ticker, want_stop, want_qty, order, dry_run),
                                     "cleared": [o.get("id") for o in sells]})
        except Exception as e:
            print(f"  [Stop] {ticker}: {e}")
            result["errors"].append({"ticker": ticker, "stop": want_stop, "error": str(e)})

    return result


# ── broker actions ───────────────────────────────────────────────────────────

def _row(ticker, stop, qty, order, dry_run) -> dict:
    return {"ticker": ticker, "stop": stop, "qty": qty,
            "order_id": (order or {}).get("id"), "dry_run": dry_run}


def _place(ticker: str, qty: int, stop: float, dry_run: bool) -> dict:
    if dry_run:
        print(f"  [DryRun] Would place GTC stop {ticker} qty={qty} @ ${stop:.2f}")
        return {}
    order = submit_stop_order(ticker, qty, stop)
    print(f"  [Stop] {ticker}: placed GTC stop qty={qty} @ ${stop:.2f} "
          f"(id={order.get('id')}, status={order.get('status')})")
    return order


def _replace(ticker: str, resting: dict, qty: int, stop: float,
             dry_run: bool) -> "tuple[dict, str]":
    """PATCH the resting stop; if the broker refuses, cancel it and place a
    fresh one — but only once the old order is provably dead. A PATCH that
    half-applied, or a stop that fired in the meantime, must not end with a
    second sell on shares that are no longer there."""
    old_id = resting.get("id")
    old_price = float(resting.get("stop_price") or 0)
    if dry_run:
        print(f"  [DryRun] Would replace stop {ticker} {old_id}: "
              f"${old_price:.2f}/{resting.get('qty')} → ${stop:.2f}/{qty}")
        return {}, "patch"
    try:
        order = replace_order(old_id, ticker=ticker, stop_price=stop, qty=qty)
        print(f"  [Stop] {ticker}: replaced {old_id} ${old_price:.2f} → ${stop:.2f} "
              f"qty={qty} (new id={order.get('id')})")
        return order, "patch"
    except Exception as e:
        print(f"  [Stop] {ticker}: PATCH refused ({e}) — cancelling {old_id} and re-placing")
    release_reserved_shares(ticker)
    blocker = _wait_until_dead([old_id])
    if blocker:
        raise RuntimeError(f"old stop not released: {blocker}")
    return _place(ticker, qty, stop, dry_run), "cancel+submit"


def _clear_and_place(ticker: str, sells: list, qty: int, stop: float, dry_run: bool) -> dict:
    """(d): whatever is resting is not the engine's stop — a legacy OCO, a
    partial close, two orders where there should be one. Clear it all, wait
    for the shares to come back, then place."""
    ids = [o.get("id") for o in sells if o.get("id")]
    if dry_run:
        print(f"  [DryRun] Would cancel {len(ids)} resting sell order(s) on {ticker} "
              f"({', '.join(ids)}) and place GTC stop qty={qty} @ ${stop:.2f}")
        return {}
    release_reserved_shares(ticker)
    blocker = _wait_until_dead(ids)
    if blocker:
        raise RuntimeError(f"resting sell orders not released: {blocker}")
    return _place(ticker, qty, stop, dry_run)


def _remember(state: dict, ticker: str, rec: dict, order: dict, stop: float,
              qty: int, dry_run: bool) -> None:
    """Persist which broker order protects the position. Skipped in dry run
    (nothing was placed) and when the record already says exactly this."""
    if dry_run:
        return
    order_id = (order or {}).get("id")
    if (rec.get("broker_stop_order_id") == order_id
            and rec.get("broker_stop_price") == stop
            and rec.get("broker_stop_qty") == qty):
        return
    rec["broker_stop_order_id"] = order_id
    rec["broker_stop_price"] = stop
    rec["broker_stop_qty"] = qty
    state[ticker] = rec
    save_position_state(state)
