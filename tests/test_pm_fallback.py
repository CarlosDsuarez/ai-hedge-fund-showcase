"""The PM agent must never return an empty decision set without saying so.

A forced tool call that hits max_tokens returns stop_reason="max_tokens" with a
truncated tool_use block whose .input is {} — no exception is raised, so the
old `except Exception` fallback never fired and run_pm returned {}. Downstream,
run_entries() builds its BUY candidates from pm_result["recommendations"], so an
empty dict silently blocks every entry regardless of market regime.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_RESEARCH = [
    {"ticker": "AAPL", "recommendation": "BUY", "conviction": 4, "thesis": "t1"},
    {"ticker": "MSFT", "recommendation": "HOLD", "conviction": 2, "thesis": "t2"},
    {"ticker": "NVDA", "recommendation": "BUY", "conviction": 5, "thesis": "t3"},
]
_RISK = {"var_95": 0.02, "max_drawdown": -0.05, "sharpe_ratio": 0.6,
         "avg_correlation": 0.3, "total_exposure": 0.5, "status": "OK", "violations": []}


def _mock_client(content, stop_reason="tool_use"):
    mock = MagicMock()
    message = MagicMock()
    message.content = content
    message.stop_reason = stop_reason
    mock.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = message
    return mock


def _tool_block(payload):
    block = MagicMock()
    block.type = "tool_use"
    block.input = payload
    return block


def _run(mock):
    from agents.pm_agent import run_pm
    with patch("agents.pm_agent.client", mock), \
         patch("agents.pm_agent.record_usage"):
        return run_pm(_RESEARCH, _RISK)


def _assert_is_fallback(result):
    assert result.get("degraded") is True, "fallback must be flagged for the health alert"
    assert len(result["recommendations"]) == len(_RESEARCH)
    assert {r["ticker"] for r in result["recommendations"]} == {"AAPL", "MSFT", "NVDA"}


def test_truncated_response_falls_back():
    """stop_reason=max_tokens: the tool_use block is present but incomplete."""
    mock = _mock_client([_tool_block({})], stop_reason="max_tokens")
    _assert_is_fallback(_run(mock))


def test_truncated_response_with_partial_payload_falls_back():
    """Truncation can leave a *partially* populated payload — still unusable."""
    partial = {"recommendations": [
        {"ticker": "AAPL", "action": "BUY", "size_pct": 0.0, "conviction": 4, "rationale": "t"},
    ]}
    mock = _mock_client([_tool_block(partial)], stop_reason="max_tokens")
    _assert_is_fallback(_run(mock))


def test_empty_tool_input_falls_back():
    """.input == {} without a truncation signal is still an empty decision set."""
    mock = _mock_client([_tool_block({})])
    _assert_is_fallback(_run(mock))


def test_empty_recommendations_list_falls_back():
    payload = {"recommendations": [], "portfolio_comment": "nothing to do"}
    mock = _mock_client([_tool_block(payload)])
    _assert_is_fallback(_run(mock))


def test_missing_tool_use_block_falls_back():
    """No tool_use block at all — next() raises StopIteration."""
    text_block = MagicMock()
    text_block.type = "text"
    mock = _mock_client([text_block])
    _assert_is_fallback(_run(mock))


def test_recommendation_count_mismatch_falls_back():
    """Fewer decisions than tickers means tickers were silently dropped."""
    payload = {
        "recommendations": [
            {"ticker": "AAPL", "action": "BUY", "size_pct": 0.0, "conviction": 4, "rationale": "t"},
            {"ticker": "MSFT", "action": "HOLD", "size_pct": 0.0, "conviction": 2, "rationale": "t"},
        ],
        "portfolio_comment": "partial",
    }
    mock = _mock_client([_tool_block(payload)])
    _assert_is_fallback(_run(mock))


def test_api_exception_falls_back():
    """Transport failures get three attempts total before the fallback fires."""
    mock = MagicMock()
    mock.messages.stream.side_effect = RuntimeError("Connection error.")
    with patch("time.sleep"):
        _assert_is_fallback(_run(mock))
    assert mock.messages.stream.call_count == 3


def test_valid_response_is_not_degraded():
    payload = {
        "recommendations": [
            {"ticker": t, "action": "BUY", "size_pct": 0.0, "conviction": 4, "rationale": "t"}
            for t in ("AAPL", "MSFT", "NVDA")
        ],
        "portfolio_comment": "all good",
    }
    result = _run(_mock_client([_tool_block(payload)]))
    assert "degraded" not in result
    assert len(result["recommendations"]) == 3
    assert all(r["size_pct"] == 0.0 for r in result["recommendations"])


def test_max_tokens_is_high_enough_for_full_universe():
    """2048 truncated 20 tickers with rationales; the ceiling costs nothing."""
    payload = {
        "recommendations": [
            {"ticker": t, "action": "HOLD", "size_pct": 0.0, "conviction": 3, "rationale": "t"}
            for t in ("AAPL", "MSFT", "NVDA")
        ],
        "portfolio_comment": "ok",
    }
    mock = _mock_client([_tool_block(payload)])
    _run(mock)
    assert mock.messages.stream.call_args.kwargs["max_tokens"] >= 8192
