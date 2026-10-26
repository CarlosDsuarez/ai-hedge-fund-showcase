# tests/test_alpaca_stop_orders.py
"""New execution/alpaca.py surface for Task 2 (D. Ledger):

- get_activities: paginated fetch of broker fills (GET /v2/account/activities/FILL)
- get_portfolio_history: GET /v2/account/portfolio/history
- submit_stop_order / replace_order: the single broker-side GTC stop the new
  exit engine (task 3) will place and move as the trailing level rises.

No real network calls — every test patches requests.get/post/patch directly,
mirroring tests/test_alpaca_oco.py.
"""
import os
import sys
from datetime import date
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _resp(status_code=200, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = b"1" if json_body is not None else b""
    resp.json.return_value = json_body if json_body is not None else {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


# ── get_activities: pagination ───────────────────────────────────────────────

def test_get_activities_single_page_stops_when_short():
    from execution.alpaca import get_activities

    page = [{"id": f"a{i}", "symbol": "AAPL", "side": "buy", "qty": "1",
             "price": "100", "transaction_time": f"2026-06-0{i}T10:00:00Z",
             "order_id": f"o{i}"} for i in range(1, 4)]  # 3 rows, page_size default 100
    with patch("execution.alpaca.requests.get", return_value=_resp(json_body=page)) as mock_get:
        result = get_activities("FILL", page_size=100)

    assert result == page
    # A single short page (< page_size) must not trigger a second request.
    assert mock_get.call_count == 1
    called_url = mock_get.call_args.args[0]
    assert "/v2/account/activities/FILL" in called_url


def test_get_activities_paginates_with_page_token_until_short_page():
    from execution.alpaca import get_activities

    page1 = [{"id": f"a{i}", "symbol": "AAPL", "side": "buy", "qty": "1",
              "price": "100", "transaction_time": "2026-06-01T10:00:00Z"}
             for i in range(2)]
    page2 = [{"id": f"b{i}", "symbol": "MSFT", "side": "sell", "qty": "1",
              "price": "200", "transaction_time": "2026-06-02T10:00:00Z"}
             for i in range(1)]
    responses = [_resp(json_body=page1), _resp(json_body=page2)]
    with patch("execution.alpaca.requests.get", side_effect=responses) as mock_get:
        result = get_activities("FILL", page_size=2)

    assert mock_get.call_count == 2
    assert len(result) == 3
    # Second call must carry page_token = id of the last row of page 1.
    second_call_params = mock_get.call_args_list[1].kwargs["params"]
    assert second_call_params["page_token"] == "a1"


def test_get_activities_empty_page_stops_pagination():
    from execution.alpaca import get_activities

    with patch("execution.alpaca.requests.get", return_value=_resp(json_body=[])) as mock_get:
        result = get_activities("FILL")

    assert result == []
    assert mock_get.call_count == 1


def test_get_activities_passes_after_param():
    from execution.alpaca import get_activities

    with patch("execution.alpaca.requests.get", return_value=_resp(json_body=[])) as mock_get:
        get_activities("FILL", after="2026-05-15")

    params = mock_get.call_args.kwargs["params"]
    assert params["after"] == "2026-05-15"
    assert params["direction"] == "asc"


def test_get_activities_caps_page_size_at_100():
    from execution.alpaca import get_activities

    with patch("execution.alpaca.requests.get", return_value=_resp(json_body=[])) as mock_get:
        get_activities("FILL", page_size=500)

    assert mock_get.call_args.kwargs["params"]["page_size"] == 100


# ── get_portfolio_history ────────────────────────────────────────────────────

def test_get_portfolio_history_sends_period_and_timeframe():
    from execution.alpaca import get_portfolio_history

    body = {"timestamp": [1, 2], "equity": [100.0, 101.0],
            "profit_loss": [0.0, 1.0], "profit_loss_pct": [0.0, 0.01],
            "base_value": 100.0, "timeframe": "1D"}
    with patch("execution.alpaca.requests.get", return_value=_resp(json_body=body)) as mock_get:
        result = get_portfolio_history(period="1M", timeframe="1D")

    assert result == body
    params = mock_get.call_args.kwargs["params"]
    assert params == {"period": "1M", "timeframe": "1D"}
    assert "/v2/account/portfolio/history" in mock_get.call_args.args[0]


# ── submit_stop_order ────────────────────────────────────────────────────────

def test_submit_stop_order_sends_correct_payload():
    from execution.alpaca import submit_stop_order

    with patch("execution.alpaca.requests.post", return_value=_resp(json_body={"id": "s1"})) as mock_post:
        result = submit_stop_order("AAPL", 10, 150.256)

    payload = mock_post.call_args.kwargs["json"]
    assert payload["symbol"] == "AAPL"
    assert payload["qty"] == "10"
    assert payload["side"] == "sell"
    assert payload["type"] == "stop"
    assert payload["time_in_force"] == "gtc"
    assert payload["stop_price"] == "150.26"
    assert result["id"] == "s1"


def test_submit_stop_order_client_id_embeds_price_qty_and_a_time_nonce():
    """The id carries price, qty and a seconds-resolution stamp: a same-day
    replace at a new level, a re-placement after a chat trim (same level,
    fewer shares), and even a cancel-then-re-place at the identical level and
    qty must all be fresh ids, never a 422 duplicate. The `bstop` tag stays
    first so execution/ledger.EXIT_REASON_BY_TAG still reads it."""
    from execution.alpaca import submit_stop_order

    today = date.today().strftime("%Y%m%d")
    with patch("execution.alpaca.requests.post", return_value=_resp(json_body={"id": "s1"})) as mock_post:
        submit_stop_order("AAPL", 10, 150.0)
    cid = mock_post.call_args.kwargs["json"]["client_order_id"]
    assert cid.startswith(f"bstop-AAPL-150.00-10-{today}T")
    assert cid.split("-", 1)[0] == "bstop"
    assert len(cid) <= 48


def test_submit_stop_order_after_a_same_day_trim_is_not_a_duplicate():
    """Review finding: stop @104 for 10 → chat trims 3 → stop cancelled →
    _protect re-places @104 for 7. With price-only ids that was the same id
    and a 422 — the position stayed unprotected until the next day."""
    from execution.alpaca import submit_stop_order

    with patch("execution.alpaca.requests.post", return_value=_resp(json_body={"id": "s1"})) as mock_post:
        submit_stop_order("AAPL", 10, 104.0)
        first = mock_post.call_args.kwargs["json"]["client_order_id"]
        submit_stop_order("AAPL", 7, 104.0)
        second = mock_post.call_args.kwargs["json"]["client_order_id"]
    assert first != second


def test_submit_stop_order_reconciles_a_duplicate_id_with_the_resting_order():
    """Belt and braces: should the id ever collide (same second, same level,
    same qty), the resting order is returned instead of raising — the
    position IS protected by it."""
    from execution.alpaca import submit_stop_order

    dup = _resp(status_code=422, text='{"message": "client_order_id must be unique"}')
    with patch("execution.alpaca.requests.post", return_value=dup), \
         patch("execution.alpaca.get_order_by_client_id",
               return_value={"id": "resting", "status": "accepted"}) as by_cid:
        result = submit_stop_order("AAPL", 10, 150.0)
    assert result["id"] == "resting"
    by_cid.assert_called_once()


def test_submit_stop_order_longest_symbol_stays_within_48_chars():
    from execution.alpaca import submit_stop_order

    with patch("execution.alpaca.requests.post", return_value=_resp(json_body={"id": "s1"})) as mock_post:
        submit_stop_order("BRK-B", 12345, 9999.99)
    cid = mock_post.call_args.kwargs["json"]["client_order_id"]
    assert len(cid) <= 48
    assert cid.startswith("bstop-BRK.B-9999.99-12345-")


def test_submit_stop_order_translates_dash_ticker_for_alpaca():
    from execution.alpaca import submit_stop_order

    with patch("execution.alpaca.requests.post", return_value=_resp(json_body={"id": "s1"})) as mock_post:
        submit_stop_order("BRK-B", 5, 400.0)
    payload = mock_post.call_args.kwargs["json"]
    assert payload["symbol"] == "BRK.B"


# ── wait_until_dead ──────────────────────────────────────────────────────────

def test_wait_until_dead_reports_filled_and_pending_orders_separately():
    from execution import alpaca

    seen = {"a": "canceled", "b": "filled", "c": "pending_cancel"}
    with patch("execution.alpaca.wait_for_fill",
               side_effect=lambda oid, timeout_s, poll_s: {"id": oid, "status": seen[oid]}):
        out = alpaca.wait_until_dead(["a", "b", "c"], timeout_s=10, poll_s=1)
    assert out["dead"] is False
    assert out["filled"] == ["b"]
    assert out["pending"] == [("c", "pending_cancel")]
    assert out["errors"] == []


def test_wait_until_dead_treats_replaced_as_released():
    from execution import alpaca

    with patch("execution.alpaca.wait_for_fill", return_value={"id": "a", "status": "replaced"}):
        out = alpaca.wait_until_dead(["a"])
    assert out == {"dead": True, "filled": [], "pending": [], "errors": []}


def test_wait_until_dead_passes_the_timeout_and_poll_through():
    from execution import alpaca

    with patch("execution.alpaca.wait_for_fill",
               return_value={"id": "a", "status": "canceled"}) as wait:
        alpaca.wait_until_dead(["a"], timeout_s=10, poll_s=1)
    wait.assert_called_once_with("a", timeout_s=10, poll_s=1)


# ── replace_order ────────────────────────────────────────────────────────────

def test_replace_order_patches_stop_price_and_qty():
    from execution.alpaca import replace_order

    with patch("execution.alpaca.requests.patch",
               return_value=_resp(json_body={"id": "new-id", "status": "new"})) as mock_patch:
        result = replace_order("old-id", ticker="AAPL", stop_price=145.678, qty=10)

    assert "/v2/orders/old-id" in mock_patch.call_args.args[0]
    payload = mock_patch.call_args.kwargs["json"]
    assert payload["stop_price"] == "145.68"
    assert payload["qty"] == "10"
    # Alpaca replaces are a NEW order — callers must use the returned id.
    assert result["id"] == "new-id"
    assert result["id"] != "old-id"


def test_replace_order_carries_the_bstop_tag_on_the_replacement():
    """A PATCH without client_order_id gets an Alpaca UUID on the new order.
    When that ladder-raised stop later fires, execution/ledger reads no tag
    and books the exit as UNKNOWN instead of BROKER_STOP — and
    db.recent_stop_exits, which only counts STOP_*/BROKER_STOP, then skips
    the re-entry cooldown for it. 179 of 182 sell fills in the live ledger
    were untagged on 2026-09-21."""
    from execution.alpaca import replace_order

    today = date.today().strftime("%Y%m%d")
    with patch("execution.alpaca.requests.patch",
               return_value=_resp(json_body={"id": "new-id"})) as mock_patch:
        replace_order("old-id", ticker="BRK-B", stop_price=145.678, qty=10)

    cid = mock_patch.call_args.kwargs["json"]["client_order_id"]
    assert cid.startswith(f"bstop-BRK.B-145.68-10-{today}T")
    assert cid.split("-", 1)[0] == "bstop"
    assert len(cid) <= 48
