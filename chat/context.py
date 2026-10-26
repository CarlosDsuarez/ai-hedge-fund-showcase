"""
chat/context.py — builds a ContextSnapshot dict for the AI chat layer.

Combines live Alpaca data (account, positions, open orders) with two on-disk
JSON caches (portfolio_state.json, position_state.json).

The public function load_context() NEVER raises; failures set availability
flags to False and fall back to empty structures.
"""

import json
import os
from datetime import datetime, UTC

# Project root is one level above this file (chat/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PORTFOLIO_STATE_PATH = os.path.join(_PROJECT_ROOT, "data", "portfolio_state.json")
_POSITION_STATE_PATH = os.path.join(_PROJECT_ROOT, "logs", "position_state.json")


def _normalize_symbol(symbol: str) -> str:
    """Replace dots with dashes (Alpaca BRK.B → internal BRK-B)."""
    return symbol.replace(".", "-")


def _load_alpaca() -> tuple[dict, list, list, bool]:
    """
    Call Alpaca API for account, positions, and open orders.

    Returns (account_dict, positions_list, orders_list, success).
    On any exception returns empty structures and success=False.
    """
    try:
        # Import here to avoid module-level side effects if keys are missing
        from execution.alpaca import get_account, get_open_orders, get_positions

        raw_account = get_account()
        raw_positions = get_positions()
        raw_orders = get_open_orders() or []  # None = state unknown; empty is fine for read-only context

        account = {
            "equity": round(float(raw_account.get("equity", 0)), 2),
            "buying_power": round(float(raw_account.get("buying_power", 0)), 2),
            "cash": round(float(raw_account.get("cash", 0)), 2),
        }

        positions = []
        for p in raw_positions:
            positions.append(
                {
                    "symbol": _normalize_symbol(p.get("symbol", "")),
                    "qty": int(float(p.get("qty", 0))),
                    "current_price": round(float(p.get("current_price", 0)), 2),
                    "market_value": round(float(p.get("market_value", 0)), 2),
                    "unrealized_pl": round(float(p.get("unrealized_pl", 0)), 2),
                    "unrealized_plpc": round(float(p.get("unrealized_plpc", 0)), 4),
                }
            )

        orders = []
        for o in raw_orders:
            orders.append(
                {
                    "id": o.get("id", ""),
                    "symbol": _normalize_symbol(o.get("symbol", "")),
                    "qty": int(float(o.get("qty", 0))),
                    "side": o.get("side", ""),
                    "status": o.get("status", ""),
                    "type": o.get("type", ""),
                    "order_class": o.get("order_class") or "simple",
                }
            )

        return account, positions, orders, True

    except Exception as e:
        print(f"  [context] Alpaca unavailable: {e}")
        empty_account = {"equity": 0.0, "buying_power": 0.0, "cash": 0.0}
        return empty_account, [], [], False


def _load_cache() -> tuple[dict, dict, bool]:
    """
    Load portfolio_state.json and position_state.json from disk.

    Returns (portfolio_state, position_state, at_least_one_loaded).
    """
    portfolio_state: dict = {}
    position_state: dict = {}
    loaded_any = False

    for path, store in [
        (_PORTFOLIO_STATE_PATH, "portfolio"),
        (_POSITION_STATE_PATH, "position"),
    ]:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if store == "portfolio":
                portfolio_state = data
            else:
                position_state = data
            loaded_any = True
        except Exception:
            pass  # leave as empty dict; loaded_any stays False if neither loads

    return portfolio_state, position_state, loaded_any


def load_context() -> dict:
    """
    Loads a live context snapshot. Never raises.

    Returns a ContextSnapshot dict with keys:
        account, live_positions, open_orders, position_state, portfolio_state,
        alpaca_available, cache_available, loaded_at

    Note: `cache_available` is True when at least one of the two cache files
    (portfolio_state.json or position_state.json) loaded successfully. Check
    `ctx['position_state']` and `ctx['portfolio_state']` individually — either
    may be an empty dict even when `cache_available` is True.

    Note: `alpaca_available` reflects whether account/positions loaded. Open
    orders may be empty even when True, as get_open_orders() handles its own
    errors gracefully.
    """
    account, live_positions, open_orders, alpaca_available = _load_alpaca()
    portfolio_state, position_state, cache_available = _load_cache()

    return {
        "account": account,
        "live_positions": live_positions,
        "open_orders": open_orders,
        "position_state": position_state,
        "portfolio_state": portfolio_state,
        "alpaca_available": alpaca_available,
        "cache_available": cache_available,
        "loaded_at": datetime.now(UTC).isoformat(),
    }
