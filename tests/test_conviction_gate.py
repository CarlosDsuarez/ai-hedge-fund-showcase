"""The PM must be told the conviction bar the executor will actually apply.

Run 2026-08-03_08-04: regime SIDEWAYS (a state Sprint 1 task 4 retired), so
run_entries gated entries at a raised bar of 4 — but the PM prompt asked for
">= MIN_PM_CONVICTION" (3). The PM returned 4 BUY approvals, all at conviction
exactly 3, and every one was dropped before sizing. Zero orders, no stated
reason. The regime that now carries a raised bar is BEAR
(config.BEAR_MIN_CONVICTION); the drift risk pinned here is unchanged.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
import main as main_mod

_RESEARCH = [{"ticker": "AAPL", "recommendation": "BUY", "conviction": 4, "thesis": "t"}]
_RISK = {"var_95": 0.02, "max_drawdown": -0.05, "sharpe_ratio": 0.6,
         "avg_correlation": 0.3, "total_exposure": 0.5, "status": "OK", "violations": []}


def _mock_pm_client():
    mock = MagicMock()
    block = MagicMock()
    block.type = "tool_use"
    block.input = {"recommendations": [{"ticker": "AAPL", "action": "BUY",
                                        "size_pct": 0.0, "conviction": 4,
                                        "rationale": "ok"}],
                   "portfolio_comment": "test"}
    mock.messages.stream.return_value.__enter__.return_value.get_final_message.return_value.content = [block]
    return mock


def _prompt_from(mock):
    return mock.messages.stream.call_args.kwargs["messages"][0]["content"]


def test_effective_min_conviction_tracks_regime():
    assert config.effective_min_conviction("BEAR") == config.BEAR_MIN_CONVICTION
    assert config.effective_min_conviction("BULL") == config.MIN_PM_CONVICTION


def test_pm_prompt_states_the_regime_gate_not_the_base_gate():
    from agents.pm_agent import run_pm
    mock = _mock_pm_client()
    with patch("agents.pm_agent.client", mock):
        run_pm(_RESEARCH, dict(_RISK), min_conviction=4)
    prompt = _prompt_from(mock)
    assert ">=4" in prompt, "PM must be told the effective conviction bar"
    assert ">=3" not in prompt, "PM must not be told the base gate when the regime raises it"


def test_pm_prompt_defaults_to_base_gate():
    from agents.pm_agent import run_pm
    mock = _mock_pm_client()
    with patch("agents.pm_agent.client", mock):
        run_pm(_RESEARCH, dict(_RISK))
    assert f">={config.MIN_PM_CONVICTION}" in _prompt_from(mock)


def test_run_decisions_passes_the_bear_gate_to_pm():
    pm = {"recommendations": [], "portfolio_comment": "ok"}
    with patch("agents.pm_agent.run_pm", return_value=pm) as run_pm:
        main_mod.run_decisions(["AAPL"], list(_RESEARCH), [], dict(_RISK),
                               {"regime": "BEAR"}, {}, {})
    assert run_pm.call_args.kwargs["min_conviction"] == config.BEAR_MIN_CONVICTION
