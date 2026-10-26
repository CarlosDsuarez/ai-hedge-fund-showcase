# tests/test_cost_governor_wiring.py — the 5 Anthropic call sites obey the governor
#
# test_cost_governor.py covers the ladder itself. Here we only care that each
# call site (a) does not touch the API when denied, (b) lands on the fallback
# that already existed, (c) alerts on Telegram when the *main* pipeline is the
# one degrading, and (d) honours a degraded model/max_tokens when allowed.
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents import cost_governor as cg

SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"

RESEARCH_IN = [{
    "ticker": "AAPL", "recommendation": "BUY", "conviction": 4,
    "thesis": "Strong cash flow.", "key_risks": ["China"],
}]


def setup_function(_fn):
    cg.reset_cache()  # also clears the per-(agent, day) alert de-dup


def _blocked(model=SONNET, max_tokens=512):
    return cg.Decision(False, model, max_tokens, None, "BLOCKED", "budget_exhausted")


def _allowed(model=HAIKU, max_tokens=100):
    """Allowed but degraded — MINIMAL swaps in Haiku and trims the ceiling."""
    return cg.Decision(True, model, max_tokens, None, "MINIMAL", "budget_minimal")


def _full(model, max_tokens):
    return cg.Decision(True, model, max_tokens, None, "FULL", "within_budget")


def _tool_response(payload: dict):
    """Anthropic Message double carrying a *completed* forced tool call.

    Every call site now uses strict tool use, so the payload arrives as
    `.content[i].input`, not as JSON in a text block.
    """
    block = MagicMock()
    block.type = "tool_use"
    block.input = payload
    msg = MagicMock()
    msg.content = [block]
    msg.stop_reason = "tool_use"
    msg.usage.input_tokens = 10
    msg.usage.output_tokens = 5
    msg.usage.cache_read_input_tokens = 0
    return msg


_ONE_ANALYSIS = {"analyses": [
    {"ticker": "AAPL", "recommendation": "BUY", "conviction": 4,
     "thesis": "t", "key_risks": []},
]}


@pytest.fixture
def telegram():
    """Capture Telegram sends and keep spend lookups off the real DB."""
    with patch("alerts.telegram_alert.send_message", return_value=True) as send, \
         patch.object(cg, "spent_today", lambda day=None: 0.97):
        yield send


@pytest.fixture
def exit_levels_offline(monkeypatch):
    """_enrich_exit_levels fetches ATR14 and a 1d bar per ticker — on the HOLD
    fallback too, which is the path the blocked-governor tests take."""
    import pandas as pd

    import agents.research_agent as ra

    monkeypatch.setattr(ra, "get_atr", lambda *a, **k: 3.5)
    monkeypatch.setattr(ra, "get_stock_price",
                        lambda *a, **k: pd.DataFrame({"Close": [190.0]}))


# ── research ────────────────────────────────────────────────────────────

def test_research_blocked_skips_api_and_returns_hold_fallback(telegram, exit_levels_offline):
    import agents.research_agent as ra

    client = MagicMock()
    with patch.object(ra, "authorize", return_value=_blocked()), \
         patch.object(ra, "client", client), \
         patch.object(ra, "get_stock_info", return_value={"name": "Apple"}):
        result = ra.run_research(["AAPL"])[0]

    client.messages.stream.assert_not_called()
    assert result["recommendation"] == "HOLD"
    assert result["conviction"] == 1
    assert "resupuesto" in result["thesis"]
    assert telegram.call_count == 1
    body = telegram.call_args[0][0]
    assert "BLOCKED" in body and "0.97" in body  # tier + spend of the day


def test_research_estimate_covers_the_system_block_too(telegram, exit_levels_offline):
    import agents.research_agent as ra

    with patch.object(ra, "authorize", return_value=_blocked()) as auth, \
         patch.object(ra, "client", MagicMock()), \
         patch.object(ra, "get_stock_info", return_value={"name": "Apple"}):
        ra.run_research(["AAPL"])

    est = auth.call_args.kwargs["est_tokens_in"]
    system_only = len(ra._RESEARCH_SYSTEM[0]["text"]) // 4
    assert est > system_only, "system prompt missing from the token estimate"
    # The batched call answers for the whole universe in one response.
    assert auth.call_args.kwargs["est_max_tokens_out"] == 4096


def test_research_allowed_uses_the_degraded_model_and_ceiling(telegram):
    import agents.research_agent as ra

    client = MagicMock()
    client.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = _tool_response(_ONE_ANALYSIS)
    with patch.object(ra, "authorize", return_value=_allowed(HAIKU, 200)), \
         patch.object(ra, "client", client), \
         patch.object(ra, "get_stock_info", return_value={"name": "Apple"}), \
         patch.object(ra, "record_usage") as rec, \
         patch.object(ra, "get_atr", side_effect=RuntimeError("no atr")):
        result = ra.run_research(["AAPL"])[0]

    kwargs = client.messages.stream.call_args.kwargs
    assert kwargs["model"] == HAIKU          # degraded, not config.CLAUDE_RESEARCH_MODEL
    assert kwargs["max_tokens"] == 200
    assert rec.call_args[0][1] == HAIKU      # usage priced against the model actually used
    assert result["recommendation"] == "BUY"
    # MINIMAL still swapped Sonnet for Haiku behind the scenes — say so out loud
    assert telegram.call_count == 1
    assert "MINIMAL" in telegram.call_args[0][0]


def test_research_alerts_once_per_run_not_once_per_ticker(telegram, exit_levels_offline):
    import agents.research_agent as ra

    with patch.object(ra, "authorize", return_value=_blocked()) as auth, \
         patch.object(ra, "client", MagicMock()), \
         patch.object(ra, "get_stock_info", return_value={"name": "X"}):
        results = ra.run_research(["AAPL", "MSFT", "NVDA"])

    assert len(results) == 3
    assert all(r["recommendation"] == "HOLD" for r in results)
    assert auth.call_count == 1, "the batched call must be authorized exactly once"
    assert telegram.call_count == 1


# ── sentiment ───────────────────────────────────────────────────────────

def test_sentiment_blocked_skips_api_and_returns_neutral(telegram):
    import agents.sentiment_agent as sa

    client = MagicMock()
    with patch.object(sa, "authorize", return_value=_blocked(HAIKU, 256)), \
         patch.object(sa, "client", client), \
         patch.object(sa, "_fetch_headlines", return_value=["Apple beats estimates"]):
        result = sa.run_sentiment(["AAPL"])[0]

    client.messages.stream.assert_not_called()
    assert result == {
        "ticker": "AAPL",
        "sentiment_score": 0.0,
        "sentiment": "NEUTRAL",
        "summary": "Presupuesto LLM diario agotado; sentimiento no evaluado.",
    }
    assert telegram.call_count == 1
    assert "sentiment" in telegram.call_args[0][0]


def test_sentiment_estimate_covers_the_system_block_too(telegram):
    import agents.sentiment_agent as sa

    with patch.object(sa, "authorize", return_value=_blocked(HAIKU, 256)) as auth, \
         patch.object(sa, "client", MagicMock()), \
         patch.object(sa, "_fetch_headlines", return_value=["h"]):
        sa.run_sentiment(["AAPL"])

    est = auth.call_args.kwargs["est_tokens_in"]
    assert est > len(sa._SENTIMENT_SYSTEM[0]["text"]) // 4
    # The batched call scores the whole universe in one response.
    assert auth.call_args.kwargs["est_max_tokens_out"] == 2048


# ── pm ──────────────────────────────────────────────────────────────────

def test_pm_blocked_skips_api_and_falls_back_to_research(telegram):
    import agents.pm_agent as pm

    client = MagicMock()
    with patch.object(pm, "authorize", return_value=_blocked(SONNET, 2048)), \
         patch.object(pm, "client", client):
        result = pm.run_pm(RESEARCH_IN, {"status": "OK", "violations": []})

    client.messages.stream.assert_not_called()
    assert [r["ticker"] for r in result["recommendations"]] == ["AAPL"]
    assert result["recommendations"][0]["action"] == "BUY"       # mirrors research
    assert result["recommendations"][0]["size_pct"] == 0.0       # engine sizes, PM never does
    assert "resupuesto" in result["portfolio_comment"]
    assert telegram.call_count == 1
    assert "pm" in telegram.call_args[0][0]


def test_pm_estimate_covers_the_system_block_too(telegram):
    import agents.pm_agent as pm

    with patch.object(pm, "authorize", return_value=_blocked(SONNET, 2048)) as auth, \
         patch.object(pm, "client", MagicMock()):
        pm.run_pm(RESEARCH_IN, {"status": "OK", "violations": []})

    est = auth.call_args.kwargs["est_tokens_in"]
    assert est > len(pm._PM_SYSTEM[0]["text"]) // 4
    # 2048 truncated the forced tool call for a 20-ticker universe; the
    # governor has to price the ceiling the call actually uses.
    assert auth.call_args.kwargs["est_max_tokens_out"] == 8192


def test_pm_allowed_uses_the_degraded_model_and_ceiling(telegram):
    import agents.pm_agent as pm

    client = MagicMock()
    client.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = _tool_response(
        {"recommendations": [{"ticker": "AAPL", "action": "BUY", "size_pct": 5.0,
                              "conviction": 4, "rationale": "r"}],
         "portfolio_comment": "c"}
    )
    with patch.object(pm, "authorize", return_value=_allowed(HAIKU, 819)), \
         patch.object(pm, "client", client), \
         patch.object(pm, "record_usage") as rec:
        result = pm.run_pm(RESEARCH_IN, {"status": "OK", "violations": []})

    kwargs = client.messages.stream.call_args.kwargs
    assert kwargs["model"] == HAIKU
    assert kwargs["max_tokens"] == 819
    assert rec.call_args[0][1] == HAIKU
    assert result["recommendations"][0]["size_pct"] == 0.0
    assert telegram.call_count == 1
    assert "MINIMAL" in telegram.call_args[0][0]


# ── chat router (interactive: explicit budget message, no alert) ─────────

def test_router_blocked_returns_default_intent_with_budget_message(telegram):
    import chat.router as router

    client = MagicMock()
    with patch.object(cg, "authorize", return_value=_blocked(HAIKU, 512)), \
         patch.object(router, "_client", client):
        out = router.route_question("¿vale la pena comprar NVDA?",
                                    {"account": {}, "live_positions": []})

    client.messages.stream.assert_not_called()
    assert out["agents"] == ["portfolio_summary"]
    assert out["action"]["type"] is None
    assert "presupuesto diario agotado" in out["budget_notice"].lower()
    assert "presupuesto diario agotado" in out["reasoning"].lower()
    telegram.assert_not_called()  # the user is right there reading the answer


def test_router_estimate_covers_the_system_prompt_too(telegram):
    import chat.router as router

    with patch.object(cg, "authorize", return_value=_blocked(HAIKU, 512)) as auth, \
         patch.object(router, "_client", MagicMock()):
        router.route_question("hola", {"account": {}, "live_positions": []})

    est = auth.call_args.kwargs["est_tokens_in"]
    assert est > len(router._SYSTEM_PROMPT) // 4
    assert auth.call_args.kwargs["est_max_tokens_out"] == 512


def test_router_budget_notice_reaches_the_user_response(telegram):
    """A notice nobody renders is the same silent failure as no notice at all."""
    import chat.executor as ex
    import chat.router as router

    with patch.object(cg, "authorize", return_value=_blocked(HAIKU, 512)), \
         patch.object(router, "_client", MagicMock()):
        routing = router.route_question("hola", {"account": {}, "live_positions": []})

    ctx = {"account": {"equity": 1000.0, "cash": 500.0}, "live_positions": [], "open_orders": []}
    response, pending = ex.execute_routing(routing, ctx)

    assert "presupuesto diario agotado" in response.lower()
    assert pending is None


def test_router_allowed_uses_the_degraded_model_and_ceiling():
    import chat.router as router

    client = MagicMock()
    client.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = _tool_response(
        {"agents": ["research"], "tickers": ["NVDA"], "response_language": "es"}
    )
    with patch.object(cg, "authorize", return_value=_allowed(HAIKU, 204)), \
         patch.object(router, "_client", client), \
         patch("agents.llm_usage.record_usage") as rec:
        out = router.route_question("análisis de NVDA", {"account": {}, "live_positions": []})

    kwargs = client.messages.stream.call_args.kwargs
    assert kwargs["model"] == HAIKU
    assert kwargs["max_tokens"] == 204
    assert rec.call_args[0][1] == HAIKU
    assert out["agents"] == ["research"]
    assert out["action"]["type"] is None  # missing keys still get filled


# ── FULL tier must be a no-op: the governor cannot degrade a healthy run ──

def test_research_at_full_tier_keeps_configured_model_and_stays_quiet(telegram):
    import agents.research_agent as ra

    client = MagicMock()
    client.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = _tool_response(_ONE_ANALYSIS)
    with patch.object(ra, "authorize", return_value=_full(config.CLAUDE_RESEARCH_MODEL, 4096)), \
         patch.object(ra, "client", client), \
         patch.object(ra, "get_stock_info", return_value={"name": "Apple"}), \
         patch.object(ra, "get_atr", side_effect=RuntimeError("no atr")):
        ra.run_research(["AAPL"])

    kwargs = client.messages.stream.call_args.kwargs
    assert kwargs["model"] == config.CLAUDE_RESEARCH_MODEL
    assert kwargs["max_tokens"] == 4096
    telegram.assert_not_called()  # nothing to report — no alert fatigue


def test_sentiment_at_full_tier_keeps_configured_model_and_stays_quiet(telegram):
    import agents.sentiment_agent as sa

    client = MagicMock()
    client.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = _tool_response({"results": [
        {"ticker": "AAPL", "sentiment_score": 0.5, "sentiment": "POSITIVE", "summary": "s"},
    ]})
    with patch.object(sa, "authorize", return_value=_full(config.CLAUDE_SENTIMENT_MODEL, 2048)), \
         patch.object(sa, "client", client), \
         patch.object(sa, "_fetch_headlines", return_value=["Apple beats estimates"]):
        result = sa.run_sentiment(["AAPL"])[0]

    kwargs = client.messages.stream.call_args.kwargs
    assert kwargs["model"] == config.CLAUDE_SENTIMENT_MODEL
    assert kwargs["max_tokens"] == 2048
    assert result["sentiment"] == "POSITIVE"
    telegram.assert_not_called()


def test_pm_at_full_tier_keeps_configured_model_and_stays_quiet(telegram):
    import agents.pm_agent as pm

    client = MagicMock()
    client.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = _tool_response(
        {"recommendations": [{"ticker": "AAPL", "action": "BUY", "size_pct": 0.0,
                              "conviction": 4, "rationale": "r"}],
         "portfolio_comment": "c"}
    )
    with patch.object(pm, "authorize", return_value=_full(config.CLAUDE_PM_MODEL, 8192)), \
         patch.object(pm, "client", client):
        pm.run_pm(RESEARCH_IN, {"status": "OK", "violations": []})

    kwargs = client.messages.stream.call_args.kwargs
    assert kwargs["model"] == config.CLAUDE_PM_MODEL
    assert kwargs["max_tokens"] == 8192
    telegram.assert_not_called()


# ── alert plumbing ──────────────────────────────────────────────────────

def test_notify_degraded_reports_tier_and_daily_spend(telegram):
    cg.notify_degraded("research", _blocked())

    body = telegram.call_args[0][0]
    assert "research" in body
    assert "BLOCKED" in body
    assert "0.9700" in body                              # spend of the day
    assert f"{config.DAILY_LLM_BUDGET_USD:.2f}" in body   # against the cap


def test_notify_degraded_names_the_substituted_model_when_still_allowed(telegram):
    cg.notify_degraded("research", _allowed(HAIKU, 204))

    body = telegram.call_args[0][0]
    assert "MINIMAL" in body
    assert HAIKU in body        # which model actually ran
    assert "204" in body        # and under which ceiling


def test_notify_degraded_never_raises_when_telegram_is_down():
    with patch("alerts.telegram_alert.send_message", side_effect=RuntimeError("no net")), \
         patch.object(cg, "spent_today", lambda day=None: 0.5):
        cg.notify_degraded("pm", _blocked())  # must not propagate


def test_notify_degraded_logs_an_error_when_delivery_fails(caplog):
    """An undelivered alert still has to leave a trace an operator can find."""
    with patch("alerts.telegram_alert.send_message", return_value=False), \
         patch.object(cg, "spent_today", lambda day=None: 0.99), \
         caplog.at_level("ERROR"):
        cg.notify_degraded("sentiment", _blocked())

    assert "UNDELIVERED" in caplog.text
    assert "sentiment" in caplog.text


def test_notify_degraded_dedups_per_agent_but_not_across_agents(telegram):
    cg.notify_degraded("research", _blocked())
    cg.notify_degraded("research", _blocked())
    cg.notify_degraded("sentiment", _blocked())

    assert telegram.call_count == 2


def test_notify_degraded_alerts_again_when_the_tier_steps_down(telegram):
    """REDUCED → MINIMAL → BLOCKED is three distinct facts, not a repeat."""
    cg.notify_degraded("research", cg.Decision(True, SONNET, 307, None, "REDUCED", "budget_reduced"))
    cg.notify_degraded("research", _allowed(HAIKU, 204))
    cg.notify_degraded("research", _blocked())

    assert telegram.call_count == 3


def test_send_message_without_credentials_still_persists_the_alert(
        tmp_path, monkeypatch, unguarded_telegram):
    """An unconfigured bot must not be a silent hole in the alert path.

    Needs the real send_message (conftest stubs it suite-wide); safe because it
    clears the credentials first, so the function returns before any network call.
    """
    import alerts.telegram_alert as tg

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    monkeypatch.setattr(tg, "_LOGS_DIR", tmp_path)
    monkeypatch.setattr(tg, "_FALLBACK_LOG", tmp_path / "telegram_fallback.log")

    assert tg.send_message("Presupuesto LLM agotado — research bloqueado") is False

    persisted = (tmp_path / "telegram_fallback.log").read_text()
    assert "research bloqueado" in persisted
    assert "missing TELEGRAM_BOT_TOKEN" in persisted


# ── the 4-chars/token heuristic ─────────────────────────────────────────

def test_est_tokens_sums_every_block_and_ignores_empties():
    assert cg.est_tokens("a" * 400) == 100
    assert cg.est_tokens("a" * 400, "b" * 400) == 200
    assert cg.est_tokens("a" * 400, "", None) == 100
