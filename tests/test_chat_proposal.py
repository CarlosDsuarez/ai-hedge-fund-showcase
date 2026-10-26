"""chat/proposal.py — the decision panel, built from the cached run.

The headline regression: agents/strategy_agent.py z-scores its factors ACROSS
the list it is given. Ask it for one ticker and pandas' std() is NaN, _zscore's
divisor falls back to 1e-9, composite comes out 0.0, and 0.0 >= 0.3 is never
true — so every single-name call answers HOLD forever. chat/executor.py has
been doing exactly that. build_proposal must never repeat it.
"""
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db.database as db
from chat.proposal import build_proposal, render


def _tmp_db(tmp_path):
    return patch.object(db, "DB_PATH", str(tmp_path / "test.db"))


def _run():
    return {
        "timestamp": "2026-09-04T09:00:00",
        "strategy": {"NVDA": {"signal": "BUY", "composite_score": 0.82,
                              "macro_stressed": False, "credit_spread_z": -1.2}},
        "regime_signals": {"NVDA": {"signal": "BUY"}},
        "research": [{"ticker": "NVDA", "recommendation": "BUY", "conviction": 4,
                      "thesis": "Demanda de aceleradores.", "key_risks": ["ciclo"],
                      "atr_14": 4.0}],
        "sentiment": [{"ticker": "NVDA", "sentiment": "POSITIVE",
                       "sentiment_score": 0.4}],
        "pm": {"recommendations": [{"ticker": "NVDA", "action": "BUY",
                                    "conviction": 4, "rationale": "Momentum sólido."}],
               "portfolio_comment": ""},
        "funnel": {"stages": {"gates": 9}},
        "orders_submitted": [],
    }


def _ctx(positions=None):
    return {
        "account": {"equity": 100_000.0, "buying_power": 186_666.67, "cash": 50_000.0},
        "live_positions": positions or [],
        "open_orders": [],
        "portfolio_state": {"regime": "BULL"},
    }


def _build(tmp_path, run=None, ctx=None, **kw):
    """Everything that would touch the network is patched off."""
    with _tmp_db(tmp_path), \
         patch.dict(os.environ, {"PAPER_TRADING": "true"}), \
         patch("agents.regime_agent.detect_regime",
               return_value={"regime": "BULL", "vol_regime": "LOW",
                             "size_multiplier": 1.0}), \
         patch("execution.trade_executor.resolve_price", return_value=200.0), \
         patch("agents.sector_rotation_agent._get_ticker_sector",
               return_value="Technology"):
        db.init_db()
        if run is not None:
            db.save_run(run)
        return build_proposal("NVDA", ctx if ctx is not None else _ctx(), **kw)


# ── the H1 regression ───────────────────────────────────────────────────────

def test_the_composite_comes_from_the_run_never_from_a_one_ticker_call(tmp_path):
    with patch("agents.strategy_agent.generate_signals") as gen:
        p = _build(tmp_path, _run())
    gen.assert_not_called()
    assert p.composite == 0.82
    assert p.strategy_signal == "BUY"


def test_the_cached_path_makes_no_llm_calls(tmp_path):
    with patch("agents.research_agent.run_research") as research, \
         patch("agents.sentiment_agent.run_sentiment") as sentiment, \
         patch("agents.pm_agent.run_pm") as pm:
        p = _build(tmp_path, _run())
    research.assert_not_called()
    sentiment.assert_not_called()
    pm.assert_not_called()
    assert p.source == "cached_run"
    assert p.pm_action == "BUY"
    assert p.conviction == 4


def test_fresh_reruns_the_per_name_agents_but_keeps_the_cached_composite(tmp_path):
    fresh_research = [{"ticker": "NVDA", "recommendation": "BUY", "conviction": 5,
                       "thesis": "Nueva tesis.", "atr_14": 5.0}]
    with patch("agents.research_agent.run_research",
               return_value=fresh_research) as research, \
         patch("agents.sentiment_agent.run_sentiment", return_value=[]), \
         patch("agents.pm_agent.run_pm",
               return_value={"recommendations": [
                   {"ticker": "NVDA", "action": "BUY", "conviction": 5,
                    "rationale": "fresh"}]}) as pm, \
         patch("agents.strategy_agent.generate_signals") as gen:
        p = _build(tmp_path, _run(), force_fresh=True)

    research.assert_called_once_with(["NVDA"])
    gen.assert_not_called()
    assert p.source == "fresh"
    assert p.conviction == 5
    assert p.composite == 0.82, "the composite must still come from the batch run"
    # run_pm gets the two arguments the chat used to omit.
    kwargs = pm.call_args.kwargs
    assert "sentiment_results" in kwargs
    assert kwargs["min_conviction"] is not None


# ── levels and size ─────────────────────────────────────────────────────────

def test_levels_come_from_the_atr_of_the_run(tmp_path):
    """The stop shown is the exit engine's own (config.STOP_ATR_MULT × ATR14);
    the retired ATR_MULTIPLIER_SL env var must not move it, and there is no
    take-profit to quote any more."""
    import config
    with patch.dict(os.environ, {"ATR_MULTIPLIER_SL": "1.5"}):
        p = _build(tmp_path, _run())
    assert p.atr_14 == 4.0
    assert p.sl_price == pytest.approx(200.0 - config.STOP_ATR_MULT * 4.0)
    assert p.tp_price == 0.0


def test_missing_atr_is_named_rather_than_silently_unprotected(tmp_path):
    run = _run()
    run["research"][0].pop("atr_14")
    p = _build(tmp_path, run)
    assert p.sl_price == 0
    assert any("Sin ATR" in n for n in p.notes)




# ── gates ───────────────────────────────────────────────────────────────────

def test_a_clean_candidate_is_not_blocked(tmp_path):
    p = _build(tmp_path, _run())
    assert p.hard_failures == []
    assert p.blocked is False
    assert p.to_action()["type"] == "submit_buy"
    assert p.to_action()["qty"] == p.qty


def test_a_critical_position_blocks_the_buy_and_cannot_be_overridden(tmp_path):
    ctx = _ctx([{"symbol": "AAPL", "qty": 10, "current_price": 100.0,
                 "market_value": 1000.0, "unrealized_pl": -120.0,
                 "unrealized_plpc": -0.12}])
    p = _build(tmp_path, _run(), ctx=ctx)
    codes = {g.reason_code for g in p.hard_failures}
    assert "CRITICAL_BLOCK" in codes
    assert p.blocked is True


def test_entry_gate_failures_surface_as_overridable_warnings(tmp_path):
    run = _run()
    run["pm"]["recommendations"][0]["action"] = "HOLD"
    p = _build(tmp_path, run)
    codes = {g.reason_code for g in p.soft_failures}
    assert "PM_NOT_BUY" in codes
    assert p.blocked is False, "an opinion about a name is not a hard block"


def test_no_previous_run_says_so_instead_of_inventing_a_verdict(tmp_path):
    p = _build(tmp_path, None)
    assert p.source == "no_data"
    assert p.composite is None
    assert any("corrida previa" in n for n in p.notes)


# ── rendering ───────────────────────────────────────────────────────────────

def test_the_panel_shows_size_levels_and_every_cap(tmp_path):
    text = render(_build(tmp_path, _run()))
    assert "TAMAÑO SUGERIDO" in text
    assert "NIVELES" in text
    assert "máx. por posición" in text
    assert "máx. por sector" in text
    assert "n_floor" in text


def test_the_panel_separates_blocks_from_warnings(tmp_path):
    run = _run()
    run["pm"]["recommendations"][0]["action"] = "HOLD"
    text = render(_build(tmp_path, run))
    assert "AVISOS" in text
    assert "override" in text
