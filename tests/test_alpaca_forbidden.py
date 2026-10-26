"""403s on every write went undiagnosed because the response body was discarded.

requests' HTTPError message is "403 Client Error: Forbidden for url: …" and
nothing else. Alpaca states the real reason in the body — typically
"insufficient qty available for order (requested: 5, available: 0)" because an
open OCO still reserves the position's shares. These tests pin the body into the
error message, the release of reserving orders before any sell, and the
fail-closed behaviour when the open-order list cannot be read.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_FORBIDDEN_BODY = '{"message":"insufficient qty available for order (requested: 5, available: 0)"}'


def _resp(json_data=None, status=200, text="", url="https://paper-api.alpaca.markets/v2/orders"):
    r = MagicMock()
    r.json.return_value = json_data if json_data is not None else {}
    r.content = b"x"
    r.text = text
    r.url = url
    r.status_code = status
    r.reason = "Forbidden" if status == 403 else "OK"
    if status >= 400:
        r.raise_for_status.side_effect = requests.HTTPError(response=r)
    else:
        r.raise_for_status.return_value = None
    return r


@pytest.fixture(autouse=True)
def _clear_events():
    from execution import alpaca
    alpaca.FORBIDDEN_EVENTS.clear()
    yield
    alpaca.FORBIDDEN_EVENTS.clear()


@patch("utils.retry.time.sleep")
def test_get_error_carries_response_body(_sleep):
    from execution import alpaca

    with patch("execution.alpaca.requests.get", return_value=_resp(status=403, text=_FORBIDDEN_BODY)):
        with pytest.raises(requests.HTTPError) as exc:
            alpaca._get("/v2/positions/LLY")
    assert "insufficient qty available" in str(exc.value)


@patch("utils.retry.time.sleep")
def test_post_error_carries_response_body(_sleep):
    from execution import alpaca

    with patch("execution.alpaca.requests.post", return_value=_resp(status=403, text=_FORBIDDEN_BODY)):
        with pytest.raises(requests.HTTPError) as exc:
            alpaca._post("/v2/orders", {"symbol": "LLY"})
    assert "insufficient qty available" in str(exc.value)


@patch("utils.retry.time.sleep")
def test_delete_error_carries_response_body(_sleep):
    from execution import alpaca

    with patch("execution.alpaca.requests.delete", return_value=_resp(status=403, text=_FORBIDDEN_BODY)):
        with pytest.raises(requests.HTTPError) as exc:
            alpaca._delete("/v2/positions/LLY")
    assert "insufficient qty available" in str(exc.value)


@patch("utils.retry.time.sleep")
def test_enriched_error_keeps_response_for_transient_check(_sleep):
    """_transient and submit_order's 422/403 branches read exc.response — it must survive."""
    from execution import alpaca

    with patch("execution.alpaca.requests.get", return_value=_resp(status=403, text=_FORBIDDEN_BODY)) as mock_get:
        with pytest.raises(requests.HTTPError) as exc:
            alpaca._get("/v2/account")
    assert exc.value.response is not None
    assert exc.value.response.status_code == 403
    assert mock_get.call_count == 1  # 403 is permanent: still no retry


@patch("utils.retry.time.sleep")
def test_403_is_recorded_for_the_health_alert(_sleep):
    from execution import alpaca

    with patch("execution.alpaca.requests.delete", return_value=_resp(status=403, text=_FORBIDDEN_BODY)):
        with pytest.raises(requests.HTTPError):
            alpaca._delete("/v2/positions/LLY", symbol="LLY")
    assert len(alpaca.FORBIDDEN_EVENTS) == 1
    event = alpaca.FORBIDDEN_EVENTS[0]
    assert event["symbol"] == "LLY"
    assert event["endpoint"] == "/v2/positions/LLY"
    assert event["method"] == "DELETE"
    assert "insufficient qty" in event["detail"]


@patch("utils.retry.time.sleep")
def test_non_403_errors_are_not_recorded(_sleep):
    from execution import alpaca

    with patch("execution.alpaca.requests.post", return_value=_resp(status=422, text="nope")):
        with pytest.raises(requests.HTTPError):
            alpaca._post("/v2/orders", {"symbol": "AAPL"})
    assert alpaca.FORBIDDEN_EVENTS == []


def test_release_cancels_only_this_symbols_sell_orders():
    from execution import alpaca

    open_orders = [
        {"id": "oco-1", "symbol": "LLY", "side": "sell"},
        {"id": "buy-1", "symbol": "LLY", "side": "buy"},
        {"id": "oco-2", "symbol": "NVDA", "side": "sell"},
    ]
    with patch("execution.alpaca.get_open_orders", return_value=open_orders), \
         patch("execution.alpaca.cancel_order") as mock_cancel:
        canceled = alpaca.release_reserved_shares("LLY")
    assert canceled == ["oco-1"]
    mock_cancel.assert_called_once_with("oco-1")


def test_release_normalizes_dashed_tickers():
    """BRK-B is BRK.B at the broker; the filter must match the broker's spelling."""
    from execution import alpaca

    open_orders = [{"id": "oco-9", "symbol": "BRK.B", "side": "sell"}]
    with patch("execution.alpaca.get_open_orders", return_value=open_orders), \
         patch("execution.alpaca.cancel_order") as mock_cancel:
        canceled = alpaca.release_reserved_shares("BRK-B")
    assert canceled == ["oco-9"]
    mock_cancel.assert_called_once_with("oco-9")


def test_release_raises_when_order_state_unknown():
    """get_open_orders fails closed with None — never sell on unknown state."""
    from execution import alpaca

    with patch("execution.alpaca.get_open_orders", return_value=None), \
         patch("execution.alpaca.cancel_order") as mock_cancel:
        with pytest.raises(alpaca.OrderStateUnknown):
            alpaca.release_reserved_shares("LLY")
    mock_cancel.assert_not_called()


def test_close_position_releases_before_deleting():
    from execution import alpaca

    calls = []
    with patch("execution.alpaca.release_reserved_shares",
               side_effect=lambda t: calls.append(("release", t))), \
         patch("execution.alpaca._delete",
               side_effect=lambda *a, **k: calls.append(("delete", a[0])) or {}):
        alpaca.close_position("LLY")
    assert calls == [("release", "LLY"), ("delete", "/v2/positions/LLY")]


def test_close_position_aborted_when_state_unknown():
    from execution import alpaca

    with patch("execution.alpaca.get_open_orders", return_value=None), \
         patch("execution.alpaca._delete") as mock_delete:
        with pytest.raises(alpaca.OrderStateUnknown):
            alpaca.close_position("LLY")
    mock_delete.assert_not_called()


def test_close_partial_position_uses_delete_helper():
    """It used to call requests.delete directly: no retry, no body in errors."""
    from execution import alpaca

    with patch("execution.alpaca.release_reserved_shares"), \
         patch("execution.alpaca._delete", return_value={"ok": True}) as mock_delete, \
         patch("execution.alpaca.requests.delete") as raw_delete:
        out = alpaca.close_partial_position("LLY", 5)
    assert out == {"ok": True}
    raw_delete.assert_not_called()
    assert mock_delete.call_args.args[0] == "/v2/positions/LLY"
    assert mock_delete.call_args.kwargs["params"] == {"qty": "5"}


@patch("utils.retry.time.sleep")
def test_submit_order_logs_403_body(_sleep, capsys):
    from execution import alpaca

    with patch("execution.alpaca.requests.post", return_value=_resp(status=403, text=_FORBIDDEN_BODY)):
        with pytest.raises(requests.HTTPError):
            alpaca.submit_order("LLY", "sell", 5, tag="drift")
    out = capsys.readouterr().out
    assert "403 FORBIDDEN" in out
    assert "LLY" in out and "sell" in out and "qty=5" in out
    assert "insufficient qty available" in out
