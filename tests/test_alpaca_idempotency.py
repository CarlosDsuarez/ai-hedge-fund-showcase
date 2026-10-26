# tests/test_alpaca_idempotency.py — client_order_id determinista (auditoría Fase A, prompt 1)
import os
import sys
from datetime import date
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TODAY = date.today().isoformat()


def _mock_post_response():
    resp = MagicMock()
    resp.json.return_value = {"id": "ord1", "status": "accepted"}
    resp.raise_for_status.return_value = None
    return resp


def _submit_and_capture(fn, *args, **kwargs):
    with patch("execution.alpaca.requests.post") as mock_post:
        mock_post.return_value = _mock_post_response()
        fn(*args, **kwargs)
        return mock_post.call_args.kwargs["json"]


def test_submit_order_includes_deterministic_client_order_id():
    from execution.alpaca import submit_order

    payload = _submit_and_capture(submit_order, "AAPL", "buy", 10)
    assert payload["client_order_id"] == f"main-AAPL-buy-{TODAY}"


def test_submit_order_same_intent_same_id():
    """Retry of the same logical order produces the same id → Alpaca rejects duplicate."""
    from execution.alpaca import submit_order

    p1 = _submit_and_capture(submit_order, "MSFT", "buy", 5)
    p2 = _submit_and_capture(submit_order, "MSFT", "buy", 5)
    assert p1["client_order_id"] == p2["client_order_id"]


def test_submit_order_tags_disambiguate_intents():
    """Drift trim and chat order for the same ticker/side/day must not collide."""
    from execution.alpaca import submit_order

    p_main = _submit_and_capture(submit_order, "NVDA", "sell", 3)
    p_drift = _submit_and_capture(submit_order, "NVDA", "sell", 3, tag="drift")
    p_chat = _submit_and_capture(submit_order, "NVDA", "sell", 3, tag="chat")
    ids = {p_main["client_order_id"], p_drift["client_order_id"], p_chat["client_order_id"]}
    assert len(ids) == 3


def test_submit_order_normalizes_ticker_in_id():
    from execution.alpaca import submit_order

    payload = _submit_and_capture(submit_order, "BRK-B", "buy", 2)
    assert payload["symbol"] == "BRK.B"
    assert payload["client_order_id"] == f"main-BRK.B-buy-{TODAY}"


def test_submit_oco_order_includes_client_order_id():
    from execution.alpaca import submit_oco_order

    payload = _submit_and_capture(submit_oco_order, "LLY", 28, 980.50, 1200.00)
    assert payload["client_order_id"] == f"oco-LLY-sell-{TODAY}"


def test_client_order_id_max_48_chars():
    from execution.alpaca import _client_order_id

    long_id = _client_order_id("a-very-long-tag-name-here", "SOMETICKER", "sell")
    assert len(long_id) <= 48


# --- A2: get_open_orders fail-closed ---------------------------------------


def test_get_open_orders_returns_none_on_error():
    """API failure must return None (state unknown), never [] (state known-empty)."""
    from execution.alpaca import get_open_orders

    with patch("execution.alpaca.requests.get", side_effect=ConnectionError("api down")):
        assert get_open_orders() is None


def test_stop_audit_fails_closed_when_order_state_unknown():
    """With order state unknown, no stop may be placed or replaced (double-sell risk)."""
    import agents.risk_sl_tp_agent as agent

    # qty_available=0: whatever rests (unknown!) may hold every share.
    positions = [{"symbol": "AAPL", "current_price": "200.0", "qty": "10",
                  "qty_available": "0", "market_value": "2000.0"}]
    state = {"AAPL": {"entry_price": 190.0, "trail_stop": 185.0, "atr_14": 2.0}}
    with patch.object(agent, "get_open_orders", return_value=None), \
         patch.object(agent, "submit_stop_order") as submit, \
         patch.object(agent, "replace_order") as replace:
        result = agent.run_stop_audit(positions, state, exited=set(), dry_run=False)

    submit.assert_not_called()
    replace.assert_not_called()
    assert result["placed"] == [] and result["replaced"] == []
    assert result["errors"][0]["ticker"] == "AAPL"
    assert result.get("open_orders_unavailable") is True
