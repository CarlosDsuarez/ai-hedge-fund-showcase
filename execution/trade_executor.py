"""Broker-side execution primitives shared by the pipeline and the terminal chat.

main.py keeps what is its own — the deployment budget, the funnel, the dry-run
switch, the conviction log. What lives here is the part both callers need
identically: resolve a price, submit, wait, reconcile against the *real* fill,
and write the local exit-engine record (initial stop + chandelier trail).

The chat cannot simply import main.py to reuse `_execute_buy`: main.py runs
`init_db()`, parses argv, and calls `get_account()` + `get_positions()` at
import time (main.py:78), so importing it from the REPL would fire a live
broker round-trip on every chat start.

execution.alpaca is imported *inside* the functions on purpose. The suite
patches `execution.alpaca.<name>` by attribute, and a module-level
`from execution.alpaca import submit_order` would bind a name those patches
cannot reach.
"""
import math
from dataclasses import dataclass, field

import config

# Broker states that mean the order is over and nothing more will fill.
_TERMINAL_REJECTS = ("canceled", "rejected", "expired")


@dataclass
class BuyResult:
    """What actually happened, as opposed to what was asked for."""
    status: str                       # FILLED|PARTIAL|PENDING|SKIPPED|REJECTED|ERROR
    qty_requested: int = 0
    filled_qty: int = 0
    fill_price: float = 0.0
    spent: float = 0.0
    reason_code: str | None = None  # NO_PRICE_DATA|QTY_LT_1|ORDER_REJECTED|ORDER_ERROR
    detail: str = ""
    order: dict = field(default_factory=dict)
    levels: dict | None = None      # register_new_position record, when written
    assumed_fill: bool = False         # qty was assumed after a poll timeout


def resolve_price(ticker: str) -> float:
    """Latest price from the broker's own feed, with the data vendor as fallback.

    Sizing against a stale or divergent quote is how an order ends up a share
    short of its target, so the broker's feed goes first. Raises whatever the
    feeds raise — callers decide whether that is fatal.
    """
    from data.alpaca_data import get_latest_price

    price = get_latest_price(ticker)
    if not price:
        from data.fetcher import get_stock_price

        hist = get_stock_price(ticker, period="5d")
        price = (float(hist["Close"].iloc[-1])
                 if hist is not None and not hist.empty else 0.0)
    return float(price or 0.0)


def _is_insufficient_qty(exc: Exception) -> bool:
    resp = getattr(exc, "response", None)
    return (resp is not None and getattr(resp, "status_code", None) == 403
            and "insufficient qty" in (getattr(resp, "text", "") or "").lower())


def sell_and_confirm(ticker: str, qty: int, *, tag: str = "main") -> tuple:
    """Submit a sell and report what actually executed.

    Returns (filled_qty, fill_price); (0, 0.0) means the position did not
    change. Buys have always reconciled against the real fill, but sells did
    not: submit_order returning was treated as success, so an unfilled trim was
    indistinguishable from a completed one and still cleared the position's
    weight block.

    Every position now carries a GTC broker stop that holds all its shares.
    release_reserved_shares only fires the cancel, so the sell waits for the
    broker to release the shares (wait_until_dead) before submitting, and
    retries once after a second wait if the broker still answers 403
    "insufficient qty". A stop that FILLED while being cancelled means the
    position is already gone: the sell is abandoned rather than opening a
    short, and the ledger books the broker fill on the next reconciliation.
    """
    from execution.alpaca import (
        release_reserved_shares, submit_order, wait_for_fill, wait_until_dead,
    )
    try:
        # Any resting stop/OCO reserves these shares; releasing first is what
        # keeps the sell from being refused 403 — once the cancel has landed.
        canceled = release_reserved_shares(ticker) or []
        if canceled:
            released = wait_until_dead(canceled)
            if released["filled"]:
                print(f"  [Sell] {ticker}: resting order {released['filled']} FILLED while "
                      f"being cancelled — position already sold by the broker; no market sell")
                return 0, 0.0
            if not released["dead"]:
                print(f"  [Sell] {ticker}: cancel not yet acknowledged for "
                      f"{released['pending'] or released['errors']} — trying the sell anyway")
        try:
            order = submit_order(ticker, "sell", qty, tag=tag)
        except Exception as exc:
            if not _is_insufficient_qty(exc):
                raise
            # The cancel reported dead but the broker still had the shares
            # locked for a beat: wait again and retry exactly once.
            print(f"  [Sell] {ticker}: 403 insufficient qty right after the release — "
                  f"waiting for the cancel to settle and retrying once")
            wait_until_dead(canceled)
            order = submit_order(ticker, "sell", qty, tag=tag)
        if order.get("id"):
            order = wait_for_fill(order["id"])
        status = order.get("status", "unknown")
        filled_qty = int(float(order.get("filled_qty") or 0))
        fill_price = float(order.get("filled_avg_price") or 0)
        if status in _TERMINAL_REJECTS or filled_qty == 0:
            print(f"  [Sell] {ticker}: {status}, filled {filled_qty}/{qty} — treating as no-op")
            return 0, 0.0
        if filled_qty < qty:
            print(f"  [Sell] {ticker}: PARTIAL {filled_qty}/{qty} @ ${fill_price:.2f}")
        return filled_qty, fill_price
    except Exception as e:
        print(f"  [Sell] {ticker}: error {e}")
        return 0, 0.0


def buy_and_register(ticker: str, spend: float, *, tag: str = "main",
                     atr_14: float | None = None,
                     price: float | None = None,
                     note: str | None = None,
                     register: bool = True,
                     assume_fill_on_timeout: bool = False) -> BuyResult:
    """Turn dollars into a filled position, honestly reported.

    `assume_fill_on_timeout` exists because the two callers need opposite
    behaviour on a poll timeout. The batch pipeline has to account the
    requested qty or its remaining budget drifts, so it assumes the fill. The
    chat must not: an assumed fill would register a position and place a
    broker stop over shares that may not exist.
    """
    qty_requested = 0
    try:
        if price is None:
            price = resolve_price(ticker)
        price = float(price or 0.0)
        if not price:
            print(f"  [Skip] {ticker} no price data")
            return BuyResult(status="SKIPPED", reason_code="NO_PRICE_DATA",
                             detail="sin cotización")

        qty = math.floor(spend / price)
        if qty < 1:
            print(f"  [Skip] {ticker} qty<1 at ${price:.2f}")
            return BuyResult(
                status="SKIPPED", reason_code="QTY_LT_1",
                detail=f"${spend:,.0f} no alcanza 1 acción a ${price:.2f}")
        qty_requested = qty

        print(f"  [Order] BUY {ticker} qty={qty} @ ~${price:.2f} "
              f"({note or f'tag={tag}'})")

        from execution.alpaca import submit_order, wait_for_fill

        order = submit_order(ticker, "buy", qty, tag=tag)
        if order.get("id"):
            order = wait_for_fill(order["id"])

        status = order.get("status", "unknown")
        filled_qty = int(float(order.get("filled_qty") or 0))
        fill_price = float(order.get("filled_avg_price") or 0) or price
        assumed = False

        if status in _TERMINAL_REJECTS:
            print(f"  [Order] {ticker}: {status} — nothing filled")
            return BuyResult(status="REJECTED", qty_requested=qty,
                             reason_code="ORDER_REJECTED",
                             detail=f"estado={status}", order=order)

        if filled_qty == 0:
            if not assume_fill_on_timeout:
                print(f"  [Order] {ticker}: fill pending after timeout (status={status}) "
                      f"— nothing registered")
                return BuyResult(status="PENDING", qty_requested=qty,
                                 fill_price=price, order=order,
                                 detail=f"estado={status}")
            # A day market order on a liquid name fills; account for the
            # requested qty and flag it.
            print(f"  [Order] {ticker}: fill pending after timeout (status={status}) "
                  f"— accounting requested qty={qty}")
            filled_qty, fill_price, assumed = qty, price, True
        elif filled_qty < qty:
            print(f"  [Order] {ticker}: PARTIAL fill {filled_qty}/{qty} @ ${fill_price:.2f}")

        result = BuyResult(
            status="PARTIAL" if filled_qty < qty else "FILLED",
            qty_requested=qty, filled_qty=filled_qty, fill_price=fill_price,
            spent=filled_qty * fill_price, order=order, assumed_fill=assumed,
        )
        if register:
            result.levels = register_levels(ticker, fill_price, filled_qty, atr_14)
        return result

    except Exception as e:
        print(f"  [Order] Error {ticker}: {e}")
        return BuyResult(status="ERROR", qty_requested=qty_requested,
                         reason_code="ORDER_ERROR",
                         detail=f"{type(e).__name__}: {e}")


def register_levels(ticker: str, fill_price: float, filled_qty: int,
                    atr_14: float | None) -> dict | None:
    """Write the local exit-engine record for a fresh position.

    Local state first, broker stop second — always. The reverse order fails
    silently: a position with no record is registered on the fly by the next
    run_exit_engine pass off avg_entry_price, and until then the broker stop
    is the only thing protecting it.

    The stop is config.STOP_ATR_MULT × ATR14 below the fill and nothing else:
    no R:R gate (there is no take-profit to measure against any more) and no
    ATR_MULTIPLIER_SL env override (it silently replaced the validated value).

    `atr_14` normally comes from the research row; when that row has none (a
    failed research chunk) the live ATR is fetched — data.fetcher.get_atr(14,
    "3mo"), the same feed the engine uses to register on the fly — so the fill
    is not left without a record and a broker stop until the next run.

    Returns the record, or None when no ATR could be found anywhere.
    """
    if not atr_14 or atr_14 <= 0:
        try:
            from data.fetcher import get_atr

            atr_14 = get_atr(ticker, 14, "3mo")
        except Exception as e:
            print(f"  [Levels] {ticker}: live ATR unavailable — {e}")
            atr_14 = 0.0
        if not atr_14 or atr_14 <= 0:
            print(f"  [Levels] {ticker}: no ATR from research nor the live feed — "
                  f"the exit engine registers it on the fly next run")
            return None
        print(f"  [Levels] {ticker}: research row had no ATR — using live ATR {atr_14:.2f}")
    try:
        from execution.order_manager import register_new_position

        record = register_new_position(ticker, fill_price, filled_qty, atr_14)
        print(f"  [Levels] {ticker}: stop={record.get('initial_stop')} "
              f"({config.STOP_ATR_MULT}×ATR {atr_14:.2f}) | trail "
              f"{config.TRAIL_ATR_MULT}×ATR under the highest close")
        return record
    except Exception as e:
        print(f"  [Levels] {ticker}: {e}")
        return None
