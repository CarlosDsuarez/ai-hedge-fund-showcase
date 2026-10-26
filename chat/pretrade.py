"""The gates that belong to the chat, not to the strategy.

agents/entry_gates.py answers "does this name deserve capital" — an opinion,
and a person at the terminal is allowed to overrule an opinion in writing.
This module answers "can this order safely go out at all", and some of those
answers are not opinions:

  HARD (never overridable)
    paper_trading      the switch main.py:1376 gates its whole execution phase
                       on, which the chat used to ignore entirely
    order_state_known  get_open_orders() returning None means "unknown", and
                       selling or protecting on an unknown order state risks a
                       403 or an oversell
    no_critical        a position already past -8% halts new entries; the
                       pipeline treats this as a hard block (main.py:930)
    price_available    no quote, no sizing
    min_one_share      the budget does not reach a single share
    pipeline_idle      the scheduled run holds logs/main.lock; two writers
                       deploying the same budget is how it gets spent twice

  SOFT (shown with the shortfall, overridable in writing)
    budget / position / sector / buying-power headroom, the minimum ticket,
    and a same-day duplicate of a name the pipeline already bought.
"""
import fcntl
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from agents.entry_gates import GateResult
from utils.env import load_project_env

load_project_env()

_LOCK_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "main.lock")

CRITICAL_LOSS_PCT = -0.08   # chat-only hard gate; the pipeline's CRITICAL block is retired (design C)


def paper_trading_enabled() -> bool:
    """Read at call time — a module imported before .env loaded must not cache this."""
    return os.getenv("PAPER_TRADING", "false").strip().lower() == "true"


def pipeline_is_running() -> bool:
    """True when the scheduled run holds the lock.

    Probed with a non-blocking exclusive flock: acquiring it means nobody else
    had it, and it is released immediately. A failure to probe reports False —
    refusing to trade because the lock file could not be opened would be a
    worse failure than the race it guards against.
    """
    try:
        os.makedirs(os.path.dirname(_LOCK_FILE), exist_ok=True)
        with open(_LOCK_FILE, "a") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False
    except Exception:
        return False


def critical_positions(ctx: dict) -> list:
    """Positions already past the -8% bar that halts new entries."""
    return [p.get("symbol") for p in ctx.get("live_positions", [])
            if float(p.get("unrealized_plpc", 0.0)) < CRITICAL_LOSS_PCT]


def order_state_known() -> bool:
    """get_open_orders() fails closed with None; acting on that is the bug it prevents."""
    try:
        from execution.alpaca import get_open_orders

        return get_open_orders() is not None
    except Exception:
        return False


def _hard(name, passed, code=None, detail=""):
    return GateResult(name=name, passed=passed, severity="hard",
                      reason_code=None if passed else code,
                      detail="" if passed else detail)


def _soft(name, passed, code=None, detail="", shortfall=None):
    return GateResult(name=name, passed=passed, severity="soft",
                      reason_code=None if passed else code,
                      detail="" if passed else detail,
                      shortfall=None if passed else shortfall)


BUY_KINDS = ("buy",)
SELL_KINDS = ("close", "sell_partial", "oco")


def evaluate_chat_gates(*, kind: str, ticker: str, ctx: dict, size=None,
                        equity: float = 0.0, held_today: bool = False) -> list:
    """The chat's own gates for one proposed action, hard ones first.

    Which gates apply depends on what is being done, and getting that wrong
    would be its own bug: blocking a close because a position is down 8% would
    forbid exactly the trade that fixes it, and demanding a quote before
    cancelling an order would refuse a cancellation for no reason.
    """
    results = []

    results.append(_hard(
        "paper_trading", paper_trading_enabled(), "LIVE_ACCOUNT",
        "PAPER_TRADING no está activo — el chat no opera cuentas reales"))

    results.append(_hard(
        "pipeline_idle", not pipeline_is_running(), "PIPELINE_RUNNING",
        "la corrida automática tiene el lock — espera a que termine"))

    if kind in SELL_KINDS:
        # Selling on an unknown order state risks a 403 (shares still reserved
        # by a resting OCO) or an oversell into an unintended short.
        results.append(_hard(
            "order_state_known", order_state_known(), "ORDER_STATE_UNKNOWN",
            "no se pudo leer las órdenes abiertas — vender o proteger a ciegas "
            "arriesga un 403 o una venta en corto"))

    if kind in BUY_KINDS:
        critical = critical_positions(ctx)
        results.append(_hard(
            "no_critical_position", not critical, "CRITICAL_BLOCK",
            f"posición con pérdida > 8% pendiente de revisión: {', '.join(critical)}"))

        if size is not None:
            price = getattr(size, "price", 0.0) or 0.0
            results.append(_hard(
                "price_available", price > 0, "NO_PRICE_DATA", "sin cotización"))

            qty = getattr(size, "qty", 0) or 0
            results.append(_hard(
                "min_one_share", qty >= 1, "QTY_LT_1",
                f"${getattr(size, 'spend', 0.0):,.0f} no alcanza 1 acción "
                f"a ${price:.2f}"))

            for cap in size.caps:
                if cap.name == "target":
                    continue
                results.append(_soft(
                    f"cap_{cap.name}", cap.headroom > 0, "CAP_EXHAUSTED",
                    f"{cap.label}: límite ${cap.limit:,.0f}, "
                    f"disponible ${cap.headroom:,.0f}",
                    shortfall=max(-cap.headroom, 0.0)))

            floor = config.MIN_POSITION_PCT * max(equity, 0.0)
            ok = size.spend >= floor if floor > 0 else True
            results.append(_soft(
                "min_ticket", ok, "BELOW_MIN_TICKET",
                f"${size.spend:,.0f} < mínimo ${floor:,.0f}",
                shortfall=max(floor - size.spend, 0.0)))

        results.append(_soft(
            "not_bought_today", not held_today, "DUPLICATE_TODAY",
            f"el pipeline ya compró {ticker} hoy — los client_order_id de 'main' "
            f"y 'chat' no colisionan, así que esta orden SÍ se ejecutaría además"))

        results.append(_soft(
            "entries_open", not config.ENTRIES_FROZEN, "ENTRIES_FROZEN",
            "entradas congeladas hasta el primer rebalanceo mensual "
            "(config.ENTRIES_FROZEN) — el pipeline no compra"))

    return results


def hard_failures(results) -> list:
    return [r for r in results if not r.passed and r.severity == "hard"]


def soft_failures(results) -> list:
    return [r for r in results if not r.passed and r.severity == "soft"]
