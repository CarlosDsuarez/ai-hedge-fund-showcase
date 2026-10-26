# tests/test_alpaca_retry_fills.py — retry transient + fill reconciliation (auditoría Fase B, prompt 12)
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resp(json_data=None, status=200, text=""):
    r = MagicMock()
    r.json.return_value = json_data if json_data is not None else {}
    r.content = b"x"
    r.text = text
    r.status_code = status
    if status >= 400:
        err = requests.HTTPError(response=r)
        r.raise_for_status.side_effect = err
    else:
        r.raise_for_status.return_value = None
    return r


@patch("utils.retry.time.sleep")  # no real waits in tests
def test_get_retries_on_5xx_then_succeeds(mock_sleep):
    from execution import alpaca

    with patch("execution.alpaca.requests.get") as mock_get:
        mock_get.side_effect = [_resp(status=503), _resp({"ok": True})]
        out = alpaca._get("/v2/account")
    assert out == {"ok": True}
    assert mock_get.call_count == 2


@patch("utils.retry.time.sleep")
def test_get_does_not_retry_on_403(mock_sleep):
    from execution import alpaca

    with patch("execution.alpaca.requests.get") as mock_get:
        mock_get.side_effect = [_resp(status=403), _resp({"ok": True})]
        with pytest.raises(requests.HTTPError):
            alpaca._get("/v2/account")
    assert mock_get.call_count == 1  # permanent error: no retry


@patch("utils.retry.time.sleep")
def test_submit_order_reconciles_duplicate_client_order_id(mock_sleep):
    """422-duplicate means the order already landed — return it, don't fail."""
    from execution import alpaca

    existing = {"id": "ord-1", "status": "filled", "filled_qty": "10"}
    with (
        patch("execution.alpaca.requests.post",
              return_value=_resp(status=422, text='{"message": "client_order_id must be unique"}')),
        patch("execution.alpaca.get_order_by_client_id", return_value=existing) as mock_by_cid,
    ):
        out = alpaca.submit_order("AAPL", "buy", 10)
    assert out == existing
    mock_by_cid.assert_called_once()


@patch("execution.alpaca.time.sleep")
def test_wait_for_fill_polls_until_terminal(mock_sleep):
    from execution import alpaca

    seq = [
        {"id": "o1", "status": "new", "filled_qty": "0"},
        {"id": "o1", "status": "partially_filled", "filled_qty": "4"},
        {"id": "o1", "status": "filled", "filled_qty": "10", "filled_avg_price": "150.1"},
    ]
    with patch("execution.alpaca.get_order", side_effect=seq):
        out = alpaca.wait_for_fill("o1", timeout_s=30, poll_s=0)
    assert out["status"] == "filled"
    assert out["filled_qty"] == "10"


@patch("execution.alpaca.time.sleep")
def test_wait_for_fill_times_out_returns_last_seen(mock_sleep):
    from execution import alpaca

    pending = {"id": "o1", "status": "new", "filled_qty": "0"}
    with (
        patch("execution.alpaca.get_order", return_value=pending),
        patch("execution.alpaca.time.monotonic", side_effect=[0, 1, 31, 32, 33]),
    ):
        out = alpaca.wait_for_fill("o1", timeout_s=30, poll_s=0)
    assert out["status"] == "new"  # caller decides what to do with a pending order


def test_order_manager_persists_state_per_mutation():
    """Crash after a broker close must not resurrect the position from stale disk state."""
    import execution.order_manager as om

    positions = [
        {"symbol": "AAPL", "current_price": "90.0", "qty": "10", "market_value": "900.0",
         "avg_entry_price": "100.0"},
        {"symbol": "MSFT", "current_price": "500.0", "qty": "5", "market_value": "2500.0",
         "avg_entry_price": "500.0"},
    ]
    initial_state = {
        "AAPL": {"entry_price": 100.0, "atr_14": 2.0, "initial_stop": 95.0, "trail_stop": 95.0,
                 "highest_close": 100.0, "qty_total": 10, "qty_remaining": 10,
                 "entry_time": "2026-09-01T10:00:00"},
        "MSFT": {"entry_price": 500.0, "atr_14": 8.0, "initial_stop": 480.0, "trail_stop": 480.0,
                 "highest_close": 500.0, "qty_total": 5, "qty_remaining": 5,
                 "entry_time": "2026-09-01T10:00:00"},
    }
    saves = []

    with (
        patch.object(om, "load_position_state", return_value=dict(initial_state)),
        patch.object(om, "save_position_state", side_effect=lambda s: saves.append(dict(s))),
        patch("execution.trade_executor.sell_and_confirm", return_value=(10, 90.0)) as sell,
        patch("data.fetcher.get_atr", return_value=2.0),
    ):
        exited, closed = om.run_exit_engine(positions, score_map={}, pm_sells={}, metadata={})

    sell.assert_called_once_with("AAPL", 10, tag="stop_initial")   # AAPL hit its stop
    assert exited == {"AAPL"}
    # First save (right after the AAPL fill) must already lack AAPL
    assert "AAPL" not in saves[0]
    assert "MSFT" in saves[0]                # untouched position survives
