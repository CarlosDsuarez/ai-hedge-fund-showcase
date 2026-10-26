from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _mock_client(tool_input: dict):
    # Agents now use forced strict tool use: the response carries one
    # tool_use block whose .input is the already-validated dict.
    mock = MagicMock()
    block = MagicMock()
    block.type = "tool_use"
    block.input = tool_input
    mock.messages.stream.return_value.__enter__.return_value.get_final_message.return_value.content = [block]
    return mock


def _get_system_param(mock_client):
    call_kwargs = mock_client.messages.stream.call_args
    if call_kwargs.kwargs:
        return call_kwargs.kwargs.get("system", [])
    # positional fallback
    args, kwargs = call_kwargs
    return kwargs.get("system", [])


# Los system prompts están MUY por debajo del mínimo cacheable de Anthropic
# (2048 tok Sonnet / 4096 tok Haiku), así que cache_control nunca cacheaba nada.
# Auditoría Fase A item 9: se eliminaron los flags; estos tests garantizan que
# no vuelvan sin que crezca el prefijo estático.


def test_research_agent_system_has_no_dead_cache_control():
    import agents.research_agent as ra
    resp = {"analyses": [
        {"ticker": "AAPL", "recommendation": "BUY", "conviction": 4,
         "thesis": "test", "key_risks": []}
    ]}
    mock = _mock_client(resp)
    with patch("agents.research_agent.client", mock), \
         patch("agents.research_agent.get_stock_info", return_value={
             "ticker": "AAPL", "name": "Apple", "sector": "Tech",
             "pe_ratio": 30, "roe": 0.15, "revenue_growth": 0.1,
             "market_cap": 3e12, "free_cashflow": 1e11}), \
         patch("agents.research_agent.get_atr", return_value=2.5), \
         patch("agents.research_agent.get_stock_price", return_value=MagicMock(
             empty=False, __len__=lambda s: 1,
             __getitem__=MagicMock(return_value=MagicMock(
                 iloc=MagicMock(__getitem__=MagicMock(return_value=150.0)))))):
        ra.run_research(["AAPL"])
    system = _get_system_param(mock)
    assert isinstance(system, list) and len(system) > 0, "system param missing or empty"
    assert not any(item.get("cache_control") for item in system), (
        "dead cache_control found: prompt is below Anthropic's minimum cacheable "
        "prefix (2048/4096 tok) so the marker never caches — remove it or grow the prefix"
    )


def test_pm_agent_system_has_no_dead_cache_control():
    import agents.pm_agent as pm
    resp = {"recommendations": [
        {"ticker": "AAPL", "action": "BUY", "size_pct": 0.0, "conviction": 4, "rationale": "ok"}
    ], "portfolio_comment": "fine"}
    mock = _mock_client(resp)
    research = [{"ticker": "AAPL", "recommendation": "BUY", "conviction": 4, "thesis": "test"}]
    risk = {"var_95": 0.02, "max_drawdown": -0.05, "sharpe_ratio": 0.6,
            "avg_correlation": 0.3, "total_exposure": 0.5, "status": "OK", "violations": []}
    with patch("agents.pm_agent.client", mock):
        pm.run_pm(research, risk)
    system = _get_system_param(mock)
    assert isinstance(system, list) and len(system) > 0, "system param missing or empty"
    assert not any(item.get("cache_control") for item in system), (
        "dead cache_control found: prompt is below Anthropic's minimum cacheable "
        "prefix (2048/4096 tok) so the marker never caches — remove it or grow the prefix"
    )


def test_sentiment_agent_system_has_no_dead_cache_control():
    import agents.sentiment_agent as sa
    resp = {"results": [
        {"ticker": "AAPL", "sentiment_score": 0.5,
         "sentiment": "POSITIVE", "summary": "good"}
    ]}
    mock = _mock_client(resp)
    with patch("agents.sentiment_agent.client", mock), \
         patch("agents.sentiment_agent._fetch_headlines", return_value=["Good earnings"]):
        sa.run_sentiment(["AAPL"])
    system = _get_system_param(mock)
    assert isinstance(system, list) and len(system) > 0, "system param missing or empty"
    assert not any(item.get("cache_control") for item in system), (
        "dead cache_control found: prompt is below Anthropic's minimum cacheable "
        "prefix (2048/4096 tok) so the marker never caches — remove it or grow the prefix"
    )
