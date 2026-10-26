from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def _mock_pm_client(action="BUY"):
    # run_pm now uses forced strict tool use: the response carries one
    # tool_use block whose .input is the already-validated dict.
    mock = MagicMock()
    block = MagicMock()
    block.type = "tool_use"
    block.input = {
        "recommendations": [{"ticker": "AAPL", "action": action,
                              "size_pct": 0.0, "conviction": 4, "rationale": "ok"}],
        "portfolio_comment": "test"
    }
    mock.messages.stream.return_value.__enter__.return_value.get_final_message.return_value.content = [block]
    return mock

_RESEARCH  = [{"ticker": "AAPL", "recommendation": "BUY", "conviction": 4, "thesis": "test"}]
_RISK      = {"var_95": 0.02, "max_drawdown": -0.05, "sharpe_ratio": 0.6,
              "avg_correlation": 0.3, "total_exposure": 0.5, "status": "OK", "violations": []}
_SENTIMENT = [{"ticker": "AAPL", "sentiment": "POSITIVE", "sentiment_score": 0.8,
               "summary": "Strong earnings beat"}]

def test_run_pm_accepts_sentiment_results():
    from agents.pm_agent import run_pm
    mock = _mock_pm_client()
    with patch("agents.pm_agent.client", mock):
        result = run_pm(_RESEARCH, _RISK, sentiment_results=_SENTIMENT)
    assert result is not None
    assert "recommendations" in result

def test_run_pm_prompt_includes_sentiment_section():
    from agents.pm_agent import run_pm
    mock = _mock_pm_client()
    with patch("agents.pm_agent.client", mock):
        run_pm(_RESEARCH, _RISK, sentiment_results=_SENTIMENT)
    call_kwargs = mock.messages.stream.call_args
    # Support both positional and keyword call signatures
    if call_kwargs.kwargs:
        messages = call_kwargs.kwargs.get("messages", [])
    else:
        messages = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else call_kwargs[1].get("messages", [])
    user_content = messages[0]["content"] if messages else ""
    assert "SENTIMENT" in user_content.upper(), "PM prompt should include SENTIMENT section"

def test_run_pm_works_without_sentiment():
    from agents.pm_agent import run_pm
    mock = _mock_pm_client()
    with patch("agents.pm_agent.client", mock):
        result = run_pm(_RESEARCH, _RISK)
    assert result is not None
