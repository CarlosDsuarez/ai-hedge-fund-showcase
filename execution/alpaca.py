import os
import time
from datetime import date, datetime

import requests
from utils.env import load_project_env

from utils.retry import with_retry

load_project_env()

PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class LiveTradingDisabled(RuntimeError):
    """Raised when broker access is attempted without PAPER_TRADING=true."""


def _get_base_url() -> str:
    """Paper endpoint only. Fails closed: there is no live-trading path."""
    if os.getenv("PAPER_TRADING", "false").lower() != "true":
        raise LiveTradingDisabled("Broker access requires PAPER_TRADING=true (paper account only).")
    return PAPER_BASE_URL


def _get_headers() -> dict:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
        "Content-Type": "application/json",
    }


class OrderStateUnknown(RuntimeError):
    """Open-order state could not be read, so what reserves a symbol's shares is unknown.

    Raised instead of selling blind: a sell issued while an OCO still holds the
    shares is rejected 403, and a sell issued after that OCO already filled
    would oversell into an unintended short.
    """


# Every 403 seen during a run, for the end-of-run health alert. A Forbidden on a
# write is not a transient blip — it means an intended sell/close/trim did NOT
# happen — and the run used to end "complete" with no trace of it.
FORBIDDEN_EVENTS: list[dict] = []


def _record_forbidden(method: str, endpoint: str, symbol: str | None, body: str) -> None:
    FORBIDDEN_EVENTS.append({
        "method": method,
        "endpoint": endpoint,
        "symbol": symbol,
        "detail": body[:200],
    })


def _raise_enriched(resp, method: str, endpoint: str, symbol: str | None) -> None:
    """raise_for_status, but with the response body in the message.

    requests' default text is "403 Client Error: Forbidden for url: …", which
    says nothing about why. Alpaca puts the actual reason in the body
    ("insufficient qty available for order (requested: 5, available: 0)"), and
    discarding it is what kept 142 Forbidden responses undiagnosable.
    """
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        body = (resp.text or "")[:500]
        if resp.status_code == 403:
            _record_forbidden(method, endpoint, symbol, body)
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} for url: {resp.url} — {body}",
            response=resp,
            request=getattr(exc, "request", None),
        ) from None


def _transient(exc: Exception) -> bool:
    """Retry only transient failures: network errors, timeouts, 429, 5xx.

    4xx (except 429) are permanent — retrying a rejected request spams the API.
    POST retries are safe because orders carry a deterministic client_order_id:
    if the first attempt actually landed, the retry is rejected as duplicate
    and submit_order reconciles by fetching the existing order.
    """
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def _get(endpoint: str, params: dict | None = None, symbol: str | None = None) -> dict:
    def _do():
        resp = requests.get(
            f"{_get_base_url()}{endpoint}",
            headers=_get_headers(),
            params=params,
            timeout=10,
        )
        _raise_enriched(resp, "GET", endpoint, symbol)
        return resp.json()
    return with_retry(_do, f"GET {endpoint}", retriable=_transient)


def _post(endpoint: str, data: dict, symbol: str | None = None) -> dict:
    def _do():
        resp = requests.post(f"{_get_base_url()}{endpoint}", headers=_get_headers(), json=data, timeout=10)
        _raise_enriched(resp, "POST", endpoint, symbol or data.get("symbol"))
        return resp.json()
    return with_retry(_do, f"POST {endpoint}", retriable=_transient)


def _delete(endpoint: str, params: dict | None = None, symbol: str | None = None) -> dict:
    def _do():
        resp = requests.delete(
            f"{_get_base_url()}{endpoint}",
            headers=_get_headers(),
            params=params,
            timeout=10,
        )
        _raise_enriched(resp, "DELETE", endpoint, symbol)
        return resp.json() if resp.content else {}
    return with_retry(_do, f"DELETE {endpoint}", retriable=_transient)


def _patch(endpoint: str, data: dict, symbol: str | None = None) -> dict:
    def _do():
        resp = requests.patch(f"{_get_base_url()}{endpoint}", headers=_get_headers(), json=data, timeout=10)
        _raise_enriched(resp, "PATCH", endpoint, symbol or data.get("symbol"))
        return resp.json()
    return with_retry(_do, f"PATCH {endpoint}", retriable=_transient)


def get_account() -> dict:
    return _get("/v2/account")


def get_positions() -> list:
    return _get("/v2/positions")


def get_activities(activity_type: str = "FILL", after: str | None = None,
                   page_size: int = 100) -> list:
    """Every fill (and partial fill) since `after`, oldest first.

    GET /v2/account/activities/{type} returns at most page_size (max 100) rows
    per call in `direction=asc` order; `page_token` is the `id` of the last row
    of the previous page. Stopping on a short page (rather than counting pages
    up front) is what lets `after` reach back to LEDGER_BACKFILL_FROM without
    the caller knowing how many months of fills that implies.

    Rows carry no client_order_id — execution/ledger.py resolves the tag for
    sell fills with one get_order(order_id) call per distinct order.
    """
    results: list = []
    page_size = min(page_size, 100)
    params: dict = {"direction": "asc", "page_size": page_size}
    if after:
        params["after"] = after
    page_token: str | None = None
    while True:
        call_params = dict(params)
        if page_token:
            call_params["page_token"] = page_token
        page = _get(f"/v2/account/activities/{activity_type}", params=call_params)
        if not page:
            break
        results.extend(page)
        if len(page) < page_size:
            break
        page_token = page[-1].get("id")
        if not page_token:
            break
    return results


def get_portfolio_history(period: str = "1M", timeframe: str = "1D") -> dict:
    """{timestamp[], equity[], profit_loss[], profit_loss_pct[], base_value, timeframe}."""
    return _get("/v2/account/portfolio/history", params={"period": period, "timeframe": timeframe})


def _client_order_id(tag: str, symbol: str, side: str, seq: "int | None" = None) -> str:
    """Deterministic id so a retry/re-run of the same intent is rejected by Alpaca as duplicate (max 48 chars).

    `seq` is the escape hatch for the one case where determinism is wrong:
    re-protecting a position whose earlier OCO of the same day is already dead.
    """
    suffix = f"-{seq}" if seq else ""
    return f"{tag}-{symbol}-{side}-{date.today().isoformat()}{suffix}"[:48]


def _is_duplicate_cid(exc: requests.HTTPError) -> bool:
    """True when Alpaca refused the order because the client_order_id is taken."""
    resp = getattr(exc, "response", None)
    return (resp is not None and resp.status_code == 422
            and "client_order_id" in (resp.text or ""))


def submit_order(ticker: str, side: str, qty: int, tag: str = "main") -> dict:
    alpaca_ticker = ticker.replace("-", ".")
    cid = _client_order_id(tag, alpaca_ticker, side)
    data = {
        "symbol": alpaca_ticker,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "client_order_id": cid,
    }
    try:
        return _post("/v2/orders", data, symbol=alpaca_ticker)
    except requests.HTTPError as exc:
        # Duplicate client_order_id (422) = the order already landed (earlier
        # attempt whose response we lost, or a re-run). Reconcile: return the
        # existing order instead of failing.
        if exc.response is not None and exc.response.status_code == 422 \
                and "client_order_id" in exc.response.text:
            print(f"  [Order] {ticker}: duplicate client_order_id — reconciling with existing order")
            return get_order_by_client_id(cid)
        # 403 = the broker refused outright. Usually "insufficient qty available"
        # because an open OCO still reserves the shares. Say so out loud with the
        # full body: the order did NOT go through, and the caller's one-line
        # `[TRIM] Error {e}` used to be the only trace.
        if exc.response is not None and exc.response.status_code == 403:
            print(f"  [Order] 403 FORBIDDEN — {ticker} {side} qty={qty} tag={tag} "
                  f"NOT submitted: {exc.response.text[:500]}")
        raise


def get_order(order_id: str) -> dict:
    return _get(f"/v2/orders/{order_id}")


def get_order_by_client_id(client_order_id: str) -> dict:
    return _get("/v2/orders:by_client_order_id", params={"client_order_id": client_order_id})


_FILL_TERMINAL = {"filled", "canceled", "rejected", "expired", "done_for_day"}


def wait_for_fill(order_id: str, timeout_s: float = 30.0, poll_s: float = 2.0) -> dict:
    """Poll an order until it reaches a terminal state (or timeout).

    Returns the last order dict seen; caller must read `status` and
    `filled_qty` — a day market order can time out still pending, and
    positions must be registered with the REAL filled quantity.
    """
    deadline = time.monotonic() + timeout_s
    order = get_order(order_id)
    while order.get("status") not in _FILL_TERMINAL and time.monotonic() < deadline:
        time.sleep(poll_s)
        order = get_order(order_id)
    return order


# States in which the shares an order held are free again: wait_for_fill's
# own terminal set plus "replaced", which a PATCH leaves the original in.
_DEAD_ORDER_STATES = {"canceled", "replaced", "filled", "rejected", "expired", "done_for_day"}


def wait_until_dead(order_ids: list, timeout_s: float = 10.0, poll_s: float = 1.0) -> dict:
    """Poll each order until the broker has released its shares.

    cancel_order only FIRES the cancel; Alpaca acknowledges it asynchronously
    and until then the shares stay held_for_orders. Any sell placed in that
    window is refused 403 "insufficient qty available". Every caller that
    cancels a resting sell in order to sell (or re-protect) the same shares
    has to wait here first.

    Returns {"dead": all released, "filled": [ids that FILLED — the position
    is gone, do not sell it again], "pending": [(id, status)] still holding
    shares after the timeout, "errors": [(id, error)]}.
    """
    out: dict = {"dead": True, "filled": [], "pending": [], "errors": []}
    for oid in order_ids:
        try:
            status = (wait_for_fill(oid, timeout_s=timeout_s, poll_s=poll_s)
                      .get("status") or "").lower()
        except Exception as e:
            out["errors"].append((oid, str(e)))
            out["dead"] = False
            continue
        if status == "filled":
            out["filled"].append(oid)
        elif status not in _DEAD_ORDER_STATES:
            out["pending"].append((oid, status or "unknown"))
            out["dead"] = False
    return out


def close_position(ticker: str) -> dict:
    alpaca_ticker = ticker.replace("-", ".")
    release_reserved_shares(ticker)
    return _delete(f"/v2/positions/{alpaca_ticker}", symbol=alpaca_ticker)


def get_open_orders() -> list | None:
    """Fetch all open orders (up to 500).

    Returns None on any error (fail-closed): callers must treat None as
    "state unknown" and block actions that depend on dedup — returning []
    here would silently disable duplicate-order protection exactly when
    the API is down.
    """
    try:
        result = _get("/v2/orders", params={"status": "open", "limit": 500})
        return result if isinstance(result, list) else []
    except Exception as e:
        print(f"  [Warning] get_open_orders failed: {e}. Order state UNKNOWN — callers must fail closed.")
        return None


def submit_oco_order(ticker: str, qty: int, sl_price: float, tp_price: float,
                     tag: str = "oco") -> dict:
    """Submit an OCO sell order (take-profit limit + stop-loss stop) for an existing long position.

    Manual chat action only (chat/actions.py "submit_oco"). The pipeline no
    longer places OCOs — the exit engine protects each position with a single
    GTC stop (submit_stop_order) and run_stop_audit clears any resting OCO on
    its next pass. The client_order_id is deterministic per (tag, symbol,
    side, day), so a second OCO on the same symbol the same day is refused
    422 as a duplicate; a chat that can place and cancel them makes that
    reachable.

    So the duplicate is now inspected rather than propagated. An id still held
    by a resting order means the position is already protected and that order
    is returned. An id belonging to a filled or canceled one is dead weight,
    and a fresh id is tried so the protection actually gets placed.
    """
    alpaca_ticker = ticker.replace("-", ".")
    data = {
        "symbol": alpaca_ticker,
        "qty": str(qty),
        "side": "sell",
        "type": "limit",
        "time_in_force": "gtc",
        "order_class": "oco",
        "take_profit": {"limit_price": str(round(tp_price, 2))},
        "stop_loss": {"stop_price": str(round(sl_price, 2))},
    }
    last_exc = None
    for seq in (None, 2, 3):
        cid = _client_order_id(tag, alpaca_ticker, "sell", seq=seq)
        try:
            return _post("/v2/orders", {**data, "client_order_id": cid},
                         symbol=alpaca_ticker)
        except requests.HTTPError as exc:
            if not _is_duplicate_cid(exc):
                raise
            last_exc = exc
            try:
                existing = get_order_by_client_id(cid)
            except Exception:
                existing = {}
            status = (existing.get("status") or "").lower()
            if existing.get("id") and status not in _FILL_TERMINAL:
                print(f"  [OCO] {ticker}: an OCO with id {cid} is already resting "
                      f"(status={status or 'unknown'}) — reusing it")
                return existing
            print(f"  [OCO] {ticker}: id {cid} belongs to a {status or 'missing'} "
                  f"order — retrying with a fresh id")
    raise RuntimeError(
        f"{ticker}: could not place OCO — every client_order_id for today is taken"
    ) from last_exc


def release_reserved_shares(ticker: str) -> list:
    """Cancel the open sell orders that reserve this symbol's shares.

    An OCO placed by submit_oco_order holds the whole position: Alpaca then
    rejects any second sell with 403 "insufficient qty available for order".
    Every exit, close and trim has to clear those reservations first.

    Raises OrderStateUnknown when the open-order list cannot be read —
    get_open_orders() fails closed by returning None, and selling on an unknown
    order state risks either another 403 or an oversell.
    """
    alpaca_ticker = ticker.replace("-", ".")
    open_orders = get_open_orders()
    if open_orders is None:
        raise OrderStateUnknown(
            f"{ticker}: open-order state UNKNOWN (get_open_orders failed) — "
            f"refusing to sell blind; the shares may still be reserved by an OCO"
        )
    canceled = []
    for order in open_orders:
        if order.get("symbol") != alpaca_ticker or order.get("side") != "sell":
            continue
        order_id = order.get("id")
        if not order_id:
            continue
        try:
            cancel_order(order_id)
            canceled.append(order_id)
        except Exception as e:
            # Leave it to the sell itself to fail loudly — but name the order
            # that is still holding the shares, or the 403 looks inexplicable.
            print(f"  [Release] {ticker}: could not cancel reserving order {order_id}: {e}")
    if canceled:
        print(f"  [Release] {ticker}: canceled {len(canceled)} open sell order(s) "
              f"reserving shares before selling")
    return canceled


def close_partial_position(ticker: str, qty: int) -> dict:
    alpaca_ticker = ticker.replace("-", ".")
    release_reserved_shares(ticker)
    # Routed through _delete so it inherits transient retry and body-enriched
    # errors; it used to call requests.delete directly and had neither.
    return _delete(
        f"/v2/positions/{alpaca_ticker}",
        params={"qty": str(qty)},
        symbol=alpaca_ticker,
    )


def cancel_order(order_id: str) -> dict:
    return _delete(f"/v2/orders/{order_id}")


def _stop_client_order_id(alpaca_ticker: str, stop_price: float, qty: int) -> str:
    """`bstop-<sym>-<price>-<qty>-<stamp>`, the id every broker-side stop
    carries — placed or replaced. execution/ledger.EXIT_REASON_BY_TAG reads
    the first token to book the fill as BROKER_STOP."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"bstop-{alpaca_ticker}-{stop_price:.2f}-{int(qty)}-{stamp}"[:48]


def submit_stop_order(ticker: str, qty: int, stop_price: float) -> dict:
    """Submit the single broker-side GTC stop-loss (design C: no TP leg).

    The client_order_id folds in the stop price, the qty and a seconds
    stamp, not just (tag, symbol, day) like `_client_order_id`: the stop
    audit replaces this order whenever the trailing level rises, re-places
    it after a chat trim at the same level with fewer shares, and may
    cancel-and-re-place at the identical level and qty when a PATCH is
    refused — each of those must be a fresh id, never a 422 duplicate that
    leaves the position bare until tomorrow. The `bstop` tag stays first
    because execution/ledger.EXIT_REASON_BY_TAG reads the first token.

    Should an id still collide, the resting order it belongs to is returned
    (the position is protected by it) rather than raising.
    """
    alpaca_ticker = ticker.replace("-", ".")
    cid = _stop_client_order_id(alpaca_ticker, stop_price, qty)
    data = {
        "symbol": alpaca_ticker,
        "qty": str(qty),
        "side": "sell",
        "type": "stop",
        "time_in_force": "gtc",
        "stop_price": str(round(stop_price, 2)),
        "client_order_id": cid,
    }
    try:
        return _post("/v2/orders", data, symbol=alpaca_ticker)
    except requests.HTTPError as exc:
        if _is_duplicate_cid(exc):
            print(f"  [Stop] {ticker}: duplicate client_order_id — reusing the resting order")
            return get_order_by_client_id(cid)
        raise


def replace_order(order_id: str, *, ticker: str, stop_price: float, qty: int) -> dict:
    """PATCH /v2/orders/{id} — Alpaca cancels the original and returns a NEW
    order (fresh id; the original's status becomes "replaced"). Callers must
    persist the id this returns, not the one they passed in.

    The replacement gets its own `bstop` client_order_id. Without one Alpaca
    assigns a UUID, the ledger finds no tag when that stop fires, and the
    exit is booked UNKNOWN instead of BROKER_STOP — which also drops it from
    db.recent_stop_exits and the re-entry cooldown.
    """
    alpaca_ticker = ticker.replace("-", ".")
    return _patch(f"/v2/orders/{order_id}", {
        "stop_price": str(round(stop_price, 2)),
        "qty": str(qty),
        "client_order_id": _stop_client_order_id(alpaca_ticker, stop_price, qty),
    }, symbol=alpaca_ticker)
