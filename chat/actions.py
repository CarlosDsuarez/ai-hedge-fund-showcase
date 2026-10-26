"""Confirmation, guards and execution for trades asked for at the terminal.

The chat used to submit an order the moment the router parsed one: no
PAPER_TRADING check (main.py:1376 gates its whole execution phase on it), no
limits, and a "✓ Ejecutado" printed on the POST returning rather than on a
fill. This module is the missing half of that: what the pipeline enforces
before it spends money, applied to a trade a person asked for out loud.

Two kinds of gate, and the distinction is the whole design:

  HARD   (chat/pretrade.py) — no paper account, unknown order state, a position
         already past -8%, no quote, not even one share, the pipeline holding
         the run lock. These are refusals, not warnings.
  SOFT   (chat/pretrade.py caps + agents/entry_gates.py) — the budget, the
         per-position and sector caps, and every opinion about whether the name
         deserves capital. Shown with the shortfall, and overridable by typing
         the word.

Every decision lands in db.chat_decisions — confirmed, rejected and blocked
alike — with the gate verdicts that were on screen when the user answered.
"""
import json
import math
import uuid

from chat import pretrade
from utils.env import load_project_env

# This checkout's .env only (see utils/env.py). The PAPER_TRADING read below
# means nothing if the file was never loaded.
load_project_env()

_CONFIRM_WORDS = ("s", "si", "sí", "y", "yes")
_OVERRIDE_WORD = "override"

# Broker states that mean the order is over and nothing more will fill.
_TERMINAL_REJECTS = ("canceled", "rejected", "expired")

_KIND_BY_TYPE = {
    "close_position": "close",
    "cancel_order": "cancel_order",
    "submit_buy": "buy",
    "submit_oco": "oco",
    "sell_partial": "sell_partial",
}

_UNPROTECTED_WARNING = "el resto queda SIN protección hasta la próxima corrida"


def _paper_trading() -> bool:
    """Read the switch at CALL time, never at import.

    A module imported before load_project_env() ran would otherwise cache
    "false" — or worse, a stale "true" — for the life of the process.
    """
    return pretrade.paper_trading_enabled()


def _int_or_none(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _position(ticker: str, ctx: dict) -> dict:
    return next((p for p in ctx.get("live_positions", [])
                 if p.get("symbol") == ticker), {})


def _resolve_partial_qty(action: dict, ctx: dict) -> None:
    """Turn 'la mitad de AAPL' into a share count, in place."""
    if action.get("qty"):
        return
    pct = action.get("pct")
    held = _int_or_none(_position(action.get("ticker", ""), ctx).get("qty")) or 0
    if pct and held > 0:
        pct = float(pct)
        if pct > 1:            # the router may say 30 rather than 0.30
            pct = pct / 100.0
        action["qty"] = max(int(math.floor(held * pct)), 1) if pct > 0 else 0


def _bought_today(ticker: str) -> bool:
    """True when the last pipeline run already bought this name.

    It matters because the two are NOT deduplicated by the broker: the
    client_order_id is per (tag, symbol, side, day), and the pipeline tags
    "main" while the chat tags "chat". A second buy of the same name today
    really would go through.
    """
    try:
        from db.database import load_recent_runs

        runs = load_recent_runs(1)
        if not runs:
            return False
        return any(o.get("ticker") == ticker
                   for o in (runs[0].get("orders_submitted") or []))
    except Exception:
        return False


# ── journal ─────────────────────────────────────────────────────────────────

def _gates_payload(gates: list) -> tuple:
    rows = [{"name": g.name, "passed": g.passed, "severity": g.severity,
             "reason_code": g.reason_code, "detail": g.detail,
             "shortfall": g.shortfall} for g in gates]
    failed = ",".join(g.name for g in gates if not g.passed)
    return json.dumps(rows, ensure_ascii=False), failed


def _record(decision_id: str, journal: dict, **outcome) -> None:
    from db.database import record_chat_decision, update_chat_decision_outcome

    record_chat_decision(decision_id, journal)
    if outcome:
        update_chat_decision_outcome(decision_id, **outcome)


def _update(decision_id: str, **outcome) -> None:
    from db.database import update_chat_decision_outcome

    update_chat_decision_outcome(decision_id, **outcome)


def _build_journal(action: dict, ctx: dict, proposal, kind: str, gates: list,
                   question, session_id) -> dict:
    gates_json, gates_failed = _gates_payload(gates)
    journal = {
        "session_id": session_id,
        "question": question,
        "kind": kind,
        "ticker": action.get("ticker") or "?",
        "paper_trading": int(_paper_trading()),
        "source": "proposal" if proposal is not None else action.get("source", "manual"),
        "proposed_qty": _int_or_none(action.get("qty")),
        "sl_price": action.get("sl_price"),
        "tp_price": action.get("tp_price"),
        "gates_json": gates_json,
        "gates_failed": gates_failed,
        "user_choice": "aborted",
        "overridden": 0,
    }
    if proposal is not None:
        size = getattr(proposal, "size", None)
        journal.update({
            "ref_price": proposal.ref_price,
            "size_pct": getattr(size, "alloc_pct", None),
            "atr_14": proposal.atr_14,
            "conviction": proposal.conviction,
            "composite_score": proposal.composite,
            "regime": proposal.regime,
            "sizing_json": json.dumps({
                "n": getattr(size, "n", None),
                "n_floor": getattr(size, "n_floor", None),
                "peer_count": getattr(size, "peer_count", None),
                "deployable_frac": getattr(size, "deployable_frac", None),
                "binding": getattr(size, "binding", None),
                "caps": [{"name": c.name, "limit": c.limit, "headroom": c.headroom}
                         for c in getattr(size, "caps", [])],
            }, ensure_ascii=False),
        })
    return journal


# ── the flow ────────────────────────────────────────────────────────────────

def handle_action(action: dict, ctx: dict, *, question: str | None = None,
                  session_id: str | None = None) -> None:
    """Gate, confirm and execute one trade action. Never raises."""
    if not action or not action.get("type"):
        return

    proposal = action.pop("_proposal", None)
    kind = _KIND_BY_TYPE.get(action.get("type"), action.get("type"))
    if kind == "sell_partial":
        _resolve_partial_qty(action, ctx)

    ticker = action.get("ticker") or ""
    if proposal is not None:
        gates = list(proposal.chat_gates) + list(proposal.entry_gates)
    else:
        gates = pretrade.evaluate_chat_gates(
            kind=kind, ticker=ticker, ctx=ctx,
            equity=float(ctx.get("account", {}).get("equity") or 0.0),
            held_today=_bought_today(ticker))

    decision_id = uuid.uuid4().hex
    journal = _build_journal(action, ctx, proposal, kind, gates, question, session_id)
    desc = _describe_order(action, ctx)

    hard = pretrade.hard_failures(gates)
    if hard:
        print()
        print("⛔ Bloqueado — no se envía la orden:")
        for g in hard:
            print(f"   · {g.detail or g.reason_code}")
        print(f"  NO enviado: {desc}\n")
        _record(decision_id, journal, outcome="BLOCKED",
                error="; ".join(g.reason_code or g.name for g in hard))
        return

    soft = pretrade.soft_failures(gates)

    print()
    print("→ Esto requiere ejecutar una orden. ¿Confirmas?")
    print(f"  {desc}")
    if soft:
        print("  Avisos (no bloquean, pero exigen confirmación extra):")
        for g in soft:
            print(f"   ⚠ {g.detail or g.reason_code}")
    print("  [s] Sí, ejecutar  [n] No, cancelar")

    try:
        choice = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "n"

    if choice not in _CONFIRM_WORDS:
        print("  Orden cancelada.\n")
        journal["user_choice"] = "rejected"
        _record(decision_id, journal, outcome="NOT_EXECUTED")
        return

    overridden = 0
    if soft:
        print(f"  Hay {len(soft)} aviso(s). Escribe '{_OVERRIDE_WORD}' para ejecutar "
              f"igualmente, o Enter para cancelar.")
        try:
            second = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            second = ""
        if second != _OVERRIDE_WORD:
            print("  Orden cancelada (sin override).\n")
            journal["user_choice"] = "rejected"
            _record(decision_id, journal, outcome="NOT_EXECUTED")
            return
        overridden = 1

    # Recorded BEFORE the broker call: a crash mid-order still leaves the
    # intent and the gate verdicts on record instead of an unexplained position.
    journal["user_choice"] = "confirmed"
    journal["overridden"] = overridden
    _record(decision_id, journal)

    if kind == "buy":
        _execute_buy(decision_id, action, ctx, proposal)
    elif kind == "sell_partial":
        _execute_trim(decision_id, action, ctx)
    else:
        _execute_simple(decision_id, action, ctx)


# ── buy: fill, local levels, then broker protection ─────────────────────────

def _execute_buy(decision_id: str, action: dict, ctx: dict, proposal) -> None:
    from execution.trade_executor import buy_and_register

    qty = _int_or_none(action.get("qty")) or 0
    if qty < 1:
        print("  ✗ Cantidad inválida — nada enviado.\n")
        _update(decision_id, outcome="NOT_EXECUTED", error="qty < 1")
        return

    price = getattr(getattr(proposal, "size", None), "price", None) or None
    atr = getattr(proposal, "atr_14", None) or None
    spend = qty * float(price) if price else None
    if spend is None:
        # A hand-typed "compra 5 de AAPL" has no proposal to size from.
        from execution.trade_executor import resolve_price
        try:
            price = resolve_price(action["ticker"])
        except Exception as e:
            print(f"  ✗ Sin precio para {action['ticker']}: {e}\n")
            _update(decision_id, outcome="ERROR", error=f"no price: {e}")
            return
        if not price:
            print(f"  ✗ Sin precio para {action['ticker']}.\n")
            _update(decision_id, outcome="ERROR", error="no price")
            return
        spend = qty * float(price)

    # assume_fill_on_timeout stays False here: an assumed fill would register a
    # position and place an OCO over shares that may not exist.
    result = buy_and_register(action["ticker"], spend, tag="chat", atr_14=atr,
                              price=float(price), assume_fill_on_timeout=False)

    if result.status == "ERROR":
        print(f"  ✗ Error al ejecutar: {result.detail}\n")
        _update(decision_id, outcome="ERROR", error=result.detail)
        return
    if result.status in ("SKIPPED", "REJECTED"):
        print(f"  ✗ NO ejecutada — {result.reason_code}: {result.detail}\n")
        _update(decision_id, outcome="NOT_EXECUTED", error=result.detail)
        return
    if result.status == "PENDING":
        print("  ⏳ Enviada pero SIN fill confirmado. No la des por ejecutada — "
              "no se registraron niveles ni OCO.\n")
        _update(decision_id, outcome="PENDING",
                order_id=result.order.get("id"), filled_qty=0)
        return

    order_id = result.order.get("id")
    label = "◑ Fill PARCIAL" if result.status == "PARTIAL" else "✓ Ejecutado"
    print(f"  {label}: {result.filled_qty}/{result.qty_requested} @ "
          f"${result.fill_price:.2f}  ID: {order_id or 'N/A'}")

    _save_metadata(action["ticker"], result, ctx, proposal)

    oco_id, protected, note = _protect(action["ticker"])
    print(f"  {note}\n")

    _update(decision_id,
            outcome="PARTIAL" if result.status == "PARTIAL" else "EXECUTED",
            filled_qty=result.filled_qty,
            fill_price=round(result.fill_price, 4),
            order_id=order_id, oco_order_id=oco_id, protected=int(protected))


def _save_metadata(ticker: str, result, ctx: dict, proposal) -> None:
    """Write position_metadata the way main.py does.

    Without it main.py sees the ticker held but untracked: holding_days never
    accrues and the exit engine has no entry date to register the position
    against if its state record is ever lost.
    """
    try:
        from datetime import datetime

        from db.database import load_position_metadata, save_position_metadata

        equity = float(ctx.get("account", {}).get("equity") or 0.0)
        meta = load_position_metadata() or {}
        meta[ticker] = {
            "entry_price": round(result.fill_price, 2),
            "entry_date": datetime.now().isoformat(),
            "holding_days": 0,
            "conviction": getattr(proposal, "conviction", 0) or 0,
            "size_pct": (result.spent / equity * 100.0) if equity > 0 else 0.0,
        }
        save_position_metadata(meta)
    except Exception as e:
        print(f"  [Meta] no se pudo registrar metadata de {ticker}: {e}")


def _protect(ticker: str) -> tuple:
    """Place (or refresh) the single GTC broker stop for this position, now.

    Reuses risk_sl_tp_agent.run_stop_audit rather than calling
    submit_stop_order directly: that path already sizes off qty_available,
    clears any legacy OCO that still reserves the shares, refuses a stop that
    would fire on arrival, and fails closed when the open-order state is
    unknown. A position with no engine record (a buy whose ATR was missing)
    is registered on the fly first, exactly as the pipeline would do it.

    Returns (stop_order_id, protected, message_for_the_user).
    """
    try:
        from datetime import datetime

        from agents.risk_sl_tp_agent import run_stop_audit
        from execution.alpaca import get_positions
        from execution.order_manager import (
            load_position_state, register_on_the_fly, save_position_state, update_trail,
        )

        alpaca_sym = ticker.replace("-", ".")
        pos = next((p for p in get_positions()
                    if p.get("symbol") == alpaca_sym), None)
        if pos is None:
            return None, False, (f"⚠ {ticker} aún no aparece en el bróker — "
                                 f"{_UNPROTECTED_WARNING}")

        state = load_position_state()
        if ticker not in state:
            from data.fetcher import get_atr

            atr = get_atr(ticker, 14, "3mo")
            if not atr or atr <= 0:
                return None, False, (f"⚠ sin ATR para {ticker}: no hay nivel de stop — "
                                     f"{_UNPROTECTED_WARNING}")
            rec = register_on_the_fly(ticker, pos, atr, datetime.now().isoformat())
            state[ticker] = update_trail(rec, float(pos.get("current_price") or 0))
            save_position_state(state)

        audit = run_stop_audit([pos], state, exited=set(), dry_run=False)
        if audit.get("open_orders_unavailable"):
            return None, False, (f"⚠ estado de órdenes desconocido — "
                                 f"{_UNPROTECTED_WARNING}")

        for bucket, verb in (("protected", "ya tiene stop"),
                             ("placed", "stop colocado"),
                             ("replaced", "stop actualizado")):
            row = next((r for r in audit[bucket] if r["ticker"] == ticker), None)
            if row:
                return (row.get("order_id"), True,
                        f"✓ {ticker} {verb}: stop GTC ${row['stop']:.2f} "
                        f"sobre {row['qty']} acciones")
        row = next((r for r in audit["unprotected"] + audit["errors"]
                    if r["ticker"] == ticker), None)
        detail = (row or {}).get("reason") or (row or {}).get("error") or "sin resultado"
        return None, False, f"⚠ no se pudo proteger ({detail}) — {_UNPROTECTED_WARNING}"
    except Exception as e:
        return None, False, f"⚠ protección falló ({e}) — {_UNPROTECTED_WARNING}"


# ── partial sell ────────────────────────────────────────────────────────────

def _execute_trim(decision_id: str, action: dict, ctx: dict) -> None:
    """Sell part of a position and re-protect what is left.

    close_partial_position would release every resting sell for the symbol —
    the broker stop included — and return the order, not the fill. So the trim
    goes through sell_and_confirm (fill reconciled), then shrinks the local
    record, then re-protects. Step four is not optional: without it the
    remainder sits unprotected and "trim ejecutado" would imply "sigue
    protegido".
    """
    from execution.trade_executor import sell_and_confirm

    ticker = action["ticker"]
    qty = _int_or_none(action.get("qty")) or 0
    if qty < 1:
        print("  ✗ Cantidad inválida — nada enviado.\n")
        _update(decision_id, outcome="NOT_EXECUTED", error="qty < 1")
        return

    filled, price = sell_and_confirm(ticker, qty, tag="chat")
    if filled == 0:
        print(f"  ✗ NO ejecutada — nada se vendió de {ticker}.\n")
        _update(decision_id, outcome="NOT_EXECUTED", error="filled_qty=0")
        return

    label = "◑ Venta PARCIAL" if filled < qty else "✓ Vendido"
    print(f"  {label}: {filled}/{qty} de {ticker} @ ${price:.2f}")

    remaining = _shrink_local_state(ticker, filled, price, ctx)
    if remaining is None:
        print("  Posición cerrada por completo — sin remanente que proteger.\n")
        _update(decision_id, outcome="EXECUTED", filled_qty=filled,
                fill_price=round(price, 4), protected=1)
        return

    oco_id, protected, note = _protect(ticker)
    print(f"  {note}\n")
    _update(decision_id, outcome="PARTIAL" if filled < qty else "EXECUTED",
            filled_qty=filled, fill_price=round(price, 4),
            oco_order_id=oco_id, protected=int(protected))


def _shrink_local_state(ticker: str, sold_qty: int, price: float, ctx: dict):
    """Reduce the engine record and the position metadata. Returns the record or None."""
    record = None
    try:
        from execution.order_manager import reduce_position_qty

        record = reduce_position_qty(ticker, sold_qty)
    except Exception as e:
        print(f"  [Estado] no se pudo reducir el registro de {ticker}: {e} — "
              f"el stop del bróker puede cubrir acciones que ya no existen")

    try:
        from db.database import load_position_metadata, save_position_metadata

        meta = load_position_metadata() or {}
        if ticker in meta:
            equity = float(ctx.get("account", {}).get("equity") or 0.0)
            held = _int_or_none(_position(ticker, ctx).get("qty")) or 0
            left = max(held - sold_qty, 0)
            if left <= 0:
                meta.pop(ticker, None)
            elif equity > 0:
                meta[ticker]["size_pct"] = left * price / equity * 100.0
            save_position_metadata(meta)
    except Exception as e:
        print(f"  [Meta] no se pudo actualizar metadata de {ticker}: {e}")

    return record


# ── close / cancel / oco ────────────────────────────────────────────────────

def _execute_simple(decision_id: str, action: dict, ctx: dict) -> None:
    try:
        order, fillable = _execute(action, ctx)
    except Exception as e:
        print(f"  ✗ Error al ejecutar: {e}\n")
        _update(decision_id, outcome="ERROR", error=f"{type(e).__name__}: {e}")
        return
    _report(decision_id, action, order, fillable)


def _report(decision_id: str, action: dict, order: dict, fillable: bool) -> None:
    """Print what actually happened. Never claims a fill it did not see."""
    order_id = order.get("id") or order.get("order_id")
    is_oco = action.get("type") == "submit_oco"

    if not fillable:
        # A cancel or a resting OCO: there is nothing to fill, so "submitted"
        # is the whole truth.
        print(f"  ✓ Enviado. Status: {order.get('status', 'submitted')}  "
              f"ID: {order_id or 'N/A'}\n")
        _update(decision_id, outcome="EXECUTED",
                order_id=None if is_oco else order_id,
                oco_order_id=order_id if is_oco else None,
                protected=1 if is_oco else None)
        return

    status, filled_qty, fill_price, order = _reconcile(order)
    requested = _int_or_none(order.get("qty")) or _int_or_none(action.get("qty")) or 0

    if status in _TERMINAL_REJECTS:
        print(f"  ✗ NO ejecutada — status={status}. No se llenó nada.\n")
        outcome = "NOT_EXECUTED"
    elif filled_qty == 0:
        print(f"  ⏳ Enviada pero SIN fill confirmado (status={status}).")
        print("     No la des por ejecutada — revisa tus órdenes abiertas.\n")
        outcome = "PENDING"
    elif requested and filled_qty < requested:
        print(f"  ◑ Fill PARCIAL {filled_qty}/{requested} @ ${fill_price:.2f}  "
              f"ID: {order_id or 'N/A'}\n")
        outcome = "PARTIAL"
    else:
        print(f"  ✓ Ejecutado: {filled_qty} @ ${fill_price:.2f}  "
              f"ID: {order_id or 'N/A'}\n")
        outcome = "EXECUTED"

    _update(decision_id, outcome=outcome, filled_qty=filled_qty,
            fill_price=round(fill_price, 4) if fill_price else None,
            order_id=order_id)


def _reconcile(order: dict) -> tuple:
    """Poll the order to a terminal state. Returns (status, filled_qty, fill_price, order).

    A failure to poll is reported, not swallowed into a fake fill: the caller
    then reports PENDING, which is the honest answer.
    """
    order_id = order.get("id")
    if order_id:
        try:
            from execution.alpaca import wait_for_fill

            order = wait_for_fill(order_id)
        except Exception as e:
            print(f"  [Fill] no se pudo confirmar el fill: {e}")

    status = order.get("status", "unknown")
    filled_qty = _int_or_none(order.get("filled_qty")) or 0
    try:
        fill_price = float(order.get("filled_avg_price") or 0)
    except (TypeError, ValueError):
        fill_price = 0.0
    return status, filled_qty, fill_price, order


def _execute(action: dict, ctx: dict) -> tuple:
    """Submit the order. Returns (order_dict, fillable). Raises on failure.

    `fillable` is False for orders with no fill of their own to wait on: a
    cancellation, and an OCO that rests at the broker until a level is hit.
    """
    action_type = action.get("type")
    ticker = action.get("ticker", "")
    qty = action.get("qty")
    sl_price = action.get("sl_price")
    tp_price = action.get("tp_price")

    if action_type == "close_position":
        from execution.alpaca import close_position

        return close_position(ticker), True

    elif action_type == "cancel_order":
        from execution.alpaca import cancel_order

        order_id = _resolve_order_id(ticker, ctx)
        if not order_id:
            raise ValueError(f"No open order found for {ticker}")
        return cancel_order(order_id), False

    elif action_type == "submit_buy":
        from execution.alpaca import submit_order

        if qty is None:
            raise ValueError("qty is required for submit_buy but was not provided")
        return submit_order(ticker, "buy", int(qty), tag="chat"), True

    elif action_type == "submit_oco":
        from execution.alpaca import submit_oco_order

        if qty is None or sl_price is None or tp_price is None:
            raise ValueError("qty, sl_price, tp_price are required for submit_oco")
        return submit_oco_order(ticker, int(qty), float(sl_price),
                                float(tp_price), tag="chatoco"), False

    else:
        raise ValueError(f"Unknown action type: {action_type}")


def _resolve_order_id(ticker: str, ctx: dict) -> str | None:
    """Find order ID from ctx open_orders by ticker symbol."""
    alpaca_symbol = ticker.replace("-", ".")
    for order in ctx.get("open_orders", []):
        if order.get("symbol") in (ticker, alpaca_symbol):
            return order.get("id")
    return None


def _describe_order(action: dict, ctx: dict) -> str:
    """Build a human-readable order description."""
    action_type = action.get("type")
    ticker = action.get("ticker", "?")
    qty = action.get("qty")
    sl_price = action.get("sl_price")
    tp_price = action.get("tp_price")

    if action_type == "close_position":
        pos = _position(ticker, ctx)
        qty_str = f"{pos.get('qty')} shares" if pos else "all shares"
        return f"CLOSE {qty_str} of {ticker} @ market"

    elif action_type == "sell_partial":
        held = _position(ticker, ctx).get("qty", "?")
        return f"SELL {qty} of {held} shares of {ticker} @ market (trim)"

    elif action_type == "cancel_order":
        alpaca_symbol = ticker.replace("-", ".")
        order = next(
            (o for o in ctx.get("open_orders", []) if o.get("symbol") in (ticker, alpaca_symbol)),
            None
        )
        if order:
            return (f"CANCEL order: {ticker} {order.get('side')} "
                    f"qty={order.get('qty')} (id: {order.get('id')})")
        return f"CANCEL order for {ticker} (not found in open orders)"

    elif action_type == "submit_buy":
        return f"BUY {qty} shares of {ticker} @ market"

    elif action_type == "submit_oco":
        return f"OCO SELL {qty} shares of {ticker}  SL=${sl_price}  TP=${tp_price}"

    return f"{action_type} {ticker}"
