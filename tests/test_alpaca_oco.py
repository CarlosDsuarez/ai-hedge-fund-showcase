# tests/test_alpaca_oco.py — temporary test, delete after Task 1 passes
from datetime import date
from unittest.mock import patch, MagicMock
import os
import sys

import pytest
import requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_get_open_orders_returns_list():
    from execution.alpaca import get_open_orders
    with patch("execution.alpaca.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"symbol": "LLY", "order_class": "oco"}]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        result = get_open_orders()
    assert isinstance(result, list)
    assert result[0]["symbol"] == "LLY"

def test_submit_oco_order_sends_correct_payload():
    from execution.alpaca import submit_oco_order
    with patch("execution.alpaca.requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "abc123", "status": "pending_new"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        result = submit_oco_order("LLY", 28, 980.50, 1200.00)
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs["json"]
    assert payload["order_class"] == "oco"
    assert payload["side"] == "sell"
    assert payload["time_in_force"] == "gtc"
    assert "take_profit" in payload
    assert "stop_loss" in payload
    assert result["id"] == "abc123"


# ── OCO client_order_id: tag + duplicate reconciliation ─────────────────────
#
# The id is deterministic per (tag, symbol, side, day). A 422 duplicate used to
# raise, and the (since retired) OCO audit turned a raise into
# {"status": "ERROR"} — a position left unprotected behind one printed line.


def _resp(status_code=200, json_body=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_body if json_body is not None else {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def _duplicate_resp():
    return _resp(422, text='{"code":42210000,"message":"client_order_id must be unique"}')


def test_submit_oco_order_tag_changes_the_client_order_id():
    from execution.alpaca import submit_oco_order

    today = date.today().isoformat()
    with patch("execution.alpaca.requests.post") as post:
        post.return_value = _resp(json_body={"id": "o1", "status": "new"})
        submit_oco_order("LLY", 28, 980.5, 1200.0, tag="chatoco")
        payload = post.call_args.kwargs["json"]

    assert payload["client_order_id"] == f"chatoco-LLY-sell-{today}"


def test_duplicate_id_on_a_resting_order_reuses_it():
    """Already protected: return the resting OCO instead of raising."""
    from execution.alpaca import submit_oco_order

    resting = {"id": "oco-live", "status": "new", "order_class": "oco"}
    with patch("execution.alpaca.requests.post", return_value=_duplicate_resp()), \
         patch("execution.alpaca.get_order_by_client_id", return_value=resting), \
         patch("utils.retry.time.sleep"):
        result = submit_oco_order("LLY", 28, 980.5, 1200.0)

    assert result["id"] == "oco-live"


def test_duplicate_id_on_a_dead_order_retries_with_a_fresh_id():
    """The old OCO filled or was canceled — the position needs a NEW one."""
    from execution.alpaca import submit_oco_order

    today = date.today().isoformat()
    dead = {"id": "oco-old", "status": "canceled"}
    responses = [_duplicate_resp(), _resp(json_body={"id": "oco-new", "status": "new"})]
    with patch("execution.alpaca.requests.post", side_effect=responses) as post, \
         patch("execution.alpaca.get_order_by_client_id", return_value=dead), \
         patch("utils.retry.time.sleep"):
        result = submit_oco_order("LLY", 28, 980.5, 1200.0)

    assert result["id"] == "oco-new"
    ids = [c.kwargs["json"]["client_order_id"] for c in post.call_args_list]
    assert ids == [f"oco-LLY-sell-{today}", f"oco-LLY-sell-{today}-2"]


def test_non_duplicate_422_still_raises():
    """Only the duplicate-id case is reconciled; every other rejection propagates."""
    from execution.alpaca import submit_oco_order

    other = _resp(422, text='{"message":"stop price must be below current price"}')
    with patch("execution.alpaca.requests.post", return_value=other), \
         patch("utils.retry.time.sleep"):
        with pytest.raises(requests.HTTPError):
            submit_oco_order("LLY", 28, 980.5, 1200.0)
