"""Unit tests for the six pipeline phases extracted from _run() (audit B11).

Each phase is exercised in isolation with its external dependencies mocked at
the source module, since main.py uses function-local imports.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main as main_mod


RESEARCH = [{"ticker": "AAPL", "recommendation": "BUY", "conviction": 4,
             "thesis": "Strong.", "key_risks": [], "atr_14": 2.0, "entry_ref": 198.0}]
SENTIMENT = [{"ticker": "AAPL", "sentiment": "POSITIVE", "sentiment_score": 0.5,
              "summary": "Good news."}]
RISK = {"var_95": -0.02, "max_drawdown": -0.1, "sharpe_ratio": 1.1,
        "avg_correlation": 0.3, "total_exposure": 0.5, "status": "OK", "violations": []}
STRATEGY = {"AAPL": {"signal": "BUY", "vwap_signal": "BUY", "mom_signal": "BUY",
                     "composite_score": 1.2}}


def test_run_signals_returns_pipeline_inputs(monkeypatch):
    monkeypatch.setattr(main_mod, "TICKERS", ["AAPL"])
    # prefetch_prices returns how many tickers it cached; run_signals now aborts
    # when too few came back, so the stub has to report the real count.
    with patch("data.fetcher.prefetch_prices", return_value=1), \
         patch("agents.research_agent.run_research", return_value=list(RESEARCH)), \
         patch("agents.sentiment_agent.run_sentiment", return_value=list(SENTIMENT)):
        tickers, research, sentiment, blended = main_mod.run_signals()
    assert tickers == ["AAPL"]
    assert research == RESEARCH
    assert sentiment == SENTIMENT
    assert blended == {}


def test_run_signals_builds_sentiment_scores_when_weight_enabled(monkeypatch):
    """SENTIMENT_WEIGHT > 0 (Sprint 2, once RSS IC is measured) must feed the
    strategy composite a real {ticker: score} dict straight from
    sentiment_results, not the {} run_signals returns while the weight is 0."""
    monkeypatch.setattr(main_mod.config, "SENTIMENT_WEIGHT", 0.25)
    monkeypatch.setattr(main_mod, "TICKERS", ["AAPL"])
    with patch("data.fetcher.prefetch_prices", return_value=1), \
         patch("agents.research_agent.run_research", return_value=list(RESEARCH)), \
         patch("agents.sentiment_agent.run_sentiment", return_value=list(SENTIMENT)):
        tickers, research, sentiment, blended = main_mod.run_signals()
    assert blended == {"AAPL": 0.5}


def test_run_gates_returns_the_score_map_and_leaves_research_alone():
    """The time-decay SELL override is retired: exits belong to run_exits'
    engine (rank decay, time cap), never to a rewritten research verdict."""
    with patch("agents.risk_agent.analyze_portfolio_risk", return_value=dict(RISK)), \
         patch("agents.regime_agent.detect_regime", return_value={"regime": "BULL"}), \
         patch("agents.regime_strategy_agent.generate_regime_signals", return_value={}), \
         patch("agents.strategy_agent.generate_signals", return_value=dict(STRATEGY)):
        (risk_report, regime_result, regime_signals, strategy_signals,
         score_map, research) = main_mod.run_gates(["AAPL"], list(RESEARCH), {})
    assert risk_report["status"] == "OK"
    assert score_map == {"AAPL": 1.2}
    assert research == RESEARCH
    assert not hasattr(main_mod, "check_time_decay_exits")


def test_run_decisions_injects_regime_context_and_returns_pm():
    pm = {"recommendations": [{"ticker": "AAPL", "action": "BUY", "size_pct": 5.0,
                               "conviction": 4, "rationale": "Buy it."}],
          "portfolio_comment": "ok"}
    risk_report = dict(RISK)
    regime_result = {"regime": "BULL"}
    with patch("agents.pm_agent.run_pm", return_value=pm) as run_pm:
        result = main_mod.run_decisions(
            ["AAPL"], list(RESEARCH), list(SENTIMENT), risk_report,
            regime_result, {}, dict(STRATEGY))
    assert result is pm
    assert "BULL" in risk_report["regime_context"]
    run_pm.assert_called_once()


_EMPTY_AUDIT = {"protected": [], "placed": [], "replaced": [], "unprotected": [], "errors": []}


def test_run_exits_dry_run_returns_account_state(monkeypatch):
    monkeypatch.setattr(main_mod, "DRY_RUN", True)
    with patch("execution.alpaca.get_account", return_value={"equity": "50000"}), \
         patch("execution.alpaca.get_positions", return_value=[]), \
         patch("execution.alpaca.get_activities", return_value=[]), \
         patch("agents.risk_sl_tp_agent.run_stop_audit", return_value=_EMPTY_AUDIT) as audit, \
         patch("execution.drift_rebalancer.compute_trim_orders", return_value=[]):
        (equity, positions, exited, closed,
         stop_audit, trimmed, failed_trims) = main_mod.run_exits()
    assert equity == 50000.0
    assert positions == []
    assert exited == set() and closed == []
    assert stop_audit is _EMPTY_AUDIT
    assert audit.call_args.kwargs["dry_run"] is True
    assert trimmed == set()
    assert failed_trims == []


def _positions():
    return [
        {"symbol": "JPM", "current_price": "356.3", "qty": "12", "qty_available": "12",
         "market_value": "4275.6", "unrealized_pl": "300", "avg_entry_price": "330.0"},
        {"symbol": "AVGO", "current_price": "300.0", "qty": "23", "qty_available": "23",
         "market_value": "6900.0", "unrealized_pl": "120", "avg_entry_price": "280.0"},
    ]


def _run_exits(monkeypatch, *, engine, pm_result=None, score_map=None, positions=None):
    """Drive run_exits with the broker, the engine and the audit stubbed."""
    monkeypatch.setattr(main_mod, "DRY_RUN", False)
    import execution.order_manager as om
    om.save_position_state({"JPM": {"qty_remaining": 12, "entry_price": 330.0},
                            "AVGO": {"qty_remaining": 23, "entry_price": 280.0}})
    with patch("execution.alpaca.get_account", return_value={"equity": "50000"}), \
         patch("execution.alpaca.get_positions", return_value=positions or _positions()), \
         patch("execution.alpaca.get_activities", return_value=[]), \
         patch("execution.order_manager.run_exit_engine", side_effect=engine) as eng, \
         patch("agents.risk_sl_tp_agent.run_stop_audit", return_value=dict(_EMPTY_AUDIT)) as audit, \
         patch("execution.drift_rebalancer.compute_trim_orders", return_value=[]):
        result = main_mod.run_exits(pm_result=pm_result, strategy_score_map=score_map)
    return result, eng, audit


def test_run_exits_hands_the_engines_exits_to_the_stop_audit(monkeypatch):
    """A ticker the engine sold is still listed by the broker until the sell
    settles; auditing it would place a stop over shares that are leaving."""
    def _engine(positions, **kw):
        return {"JPM"}, [{"ticker": "JPM", "reason": "STOP_TRAIL", "qty": 12,
                          "price": 356.3, "pnl_pct": 0.08}]

    result, eng, audit = _run_exits(monkeypatch, engine=_engine)
    _, _, exited, closed, stop_audit, _, _ = result
    assert exited == {"JPM"}
    assert closed[0]["reason"] == "STOP_TRAIL"
    assert audit.call_args.kwargs["exited"] == {"JPM"}
    assert audit.call_args.kwargs["dry_run"] is False


def test_run_exits_runs_the_stop_audit_even_when_the_engine_raises(monkeypatch):
    def _engine(positions, **kw):
        raise RuntimeError("engine exploded")

    result, eng, audit = _run_exits(monkeypatch, engine=_engine)
    _, _, exited, closed, stop_audit, _, _ = result
    assert exited == set() and closed == []
    audit.assert_called_once()
    assert audit.call_args.kwargs["exited"] == set()


def test_run_exits_passes_pm_sells_on_held_names_only_and_never_when_degraded(monkeypatch):
    def _engine(positions, **kw):
        return set(), []

    pm = {"recommendations": [
        {"ticker": "JPM", "action": "SELL", "conviction": 4},
        {"ticker": "NVDA", "action": "SELL", "conviction": 5},   # not held
        {"ticker": "AVGO", "action": "BUY", "conviction": 5},
    ]}
    _, eng, _ = _run_exits(monkeypatch, engine=_engine, pm_result=pm, score_map={"JPM": 0.4})
    assert eng.call_args.kwargs["pm_sells"] == {"JPM": 4}
    assert eng.call_args.kwargs["score_map"] == {"JPM": 0.4}

    degraded = {**pm, "degraded": True}
    _, eng, _ = _run_exits(monkeypatch, engine=_engine, pm_result=degraded)
    assert eng.call_args.kwargs["pm_sells"] == {}, "a degraded PM result carries research verdicts"


def test_run_entries_dry_run_deploys_budget(monkeypatch):
    monkeypatch.setattr(main_mod, "DRY_RUN", True)
    monkeypatch.setattr(main_mod, "_load_position_metadata", lambda: {})
    saved = {}
    monkeypatch.setattr(main_mod, "_save_position_metadata", saved.update)
    monkeypatch.setattr(main_mod, "load_pending_signals", lambda: [])
    # Slot-based sizing (design B) looks up realised vol and sector momentum;
    # a unit test must not reach the network for either.
    monkeypatch.setattr(main_mod.config, "SECTOR_ROTATION_ENABLED", False)
    monkeypatch.setattr("data.fetcher.get_closes", lambda *a, **k: pd.DataFrame())
    conviction_log = []
    monkeypatch.setattr(main_mod, "append_conviction_log",
                        lambda *a, **k: conviction_log.append(a))
    pm = {"recommendations": [{"ticker": "AAPL", "action": "BUY", "size_pct": 5.0,
                               "conviction": 4, "rationale": "Buy it."}]}
    regime = {"regime": "BULL", "new_entries_allowed": True, "size_multiplier": 1.0}
    orders, closed = main_mod.run_entries(
        equity=100_000.0, positions=[], exited_tickers=set(), trimmed_tickers=set(),
        closed_positions=[], orders_submitted=[],
        pm_result=pm, sentiment_results=list(SENTIMENT), research_results=list(RESEARCH),
        strategy_signals=dict(STRATEGY), regime_signals={}, regime_result=regime,
        strategy_score_map={"AAPL": 1.2})
    assert orders == []          # dry-run submits nothing
    assert closed == []
    assert conviction_log, "dry-run buy should log conviction for AAPL"
    assert conviction_log[0][1] == "AAPL"
    assert conviction_log[0][3] == 198.0, "the reference price is research's entry_ref"


# ── Post-entry stop audit: a fresh entry is protected the same run ───────────

def _run_entries_with_one_fill(monkeypatch, *, dry_run: bool, positions_after: list,
                               extra_state: dict | None = None):
    """Drive run_entries with the buy path stubbed to one FILLED AAPL order.

    The broker stop audit's own broker calls are patched at
    agents.risk_sl_tp_agent.<name> (module-level imports there).
    """
    from execution.trade_executor import BuyResult
    import execution.order_manager as om

    monkeypatch.setattr(main_mod, "DRY_RUN", dry_run)
    monkeypatch.setattr(main_mod, "_load_position_metadata", lambda: {})
    monkeypatch.setattr(main_mod, "_save_position_metadata", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "load_pending_signals", lambda: [])
    monkeypatch.setattr(main_mod, "expire_pending_signals", lambda *a, **k: 0)
    monkeypatch.setattr(main_mod, "append_conviction_log", lambda *a, **k: None)
    monkeypatch.setattr(main_mod.config, "SECTOR_ROTATION_ENABLED", False)
    monkeypatch.setattr("data.fetcher.get_closes", lambda *a, **k: pd.DataFrame())
    # What buy_and_register -> register_levels would have written.
    om.save_position_state({"AAPL": {"qty_remaining": 10, "entry_price": 198.0,
                                     "initial_stop": 193.0, "trail_stop": 193.004},
                            **(extra_state or {})})

    def _buy(ticker, spend, **kw):
        return BuyResult(status="FILLED", qty_requested=10, filled_qty=10,
                         fill_price=198.0, spent=1980.0, order={"id": "buy-1"})

    pm = {"recommendations": [{"ticker": "AAPL", "action": "BUY", "size_pct": 5.0,
                               "conviction": 4, "rationale": "Buy it."}]}
    regime = {"regime": "BULL", "new_entries_allowed": True, "size_multiplier": 1.0}
    stop_audit_result = {"protected": [{"ticker": "JPM"}], "placed": [], "replaced": [],
                         "unprotected": [], "errors": []}
    with patch("execution.trade_executor.buy_and_register", side_effect=_buy), \
         patch("execution.alpaca.get_positions", return_value=positions_after), \
         patch("agents.risk_sl_tp_agent.get_open_orders", return_value=[]), \
         patch("agents.risk_sl_tp_agent.submit_stop_order",
               return_value={"id": "stop-1", "status": "accepted"}) as submit_stop:
        orders, _ = main_mod.run_entries(
            equity=100_000.0, positions=[], exited_tickers=set(), trimmed_tickers=set(),
            closed_positions=[], orders_submitted=[],
            pm_result=pm, sentiment_results=list(SENTIMENT), research_results=list(RESEARCH),
            strategy_signals=dict(STRATEGY), regime_signals={}, regime_result=regime,
            strategy_score_map={"AAPL": 1.2}, stop_audit_result=stop_audit_result)
    return orders, submit_stop, stop_audit_result


def test_run_entries_places_a_broker_stop_on_the_fresh_entry(monkeypatch):
    """The morning audit ran at the end of run_exits, before this buy existed:
    without a post-entry pass the position had no broker stop until the next
    morning. The stop goes in at round(trail_stop, 2) on the refetched
    position, and its rows merge into the run's stop_audit_result."""
    fresh = [{"symbol": "AAPL", "qty": "10", "qty_available": "10",
              "market_value": "1980.0", "current_price": "198.0"}]
    orders, submit_stop, audit = _run_entries_with_one_fill(
        monkeypatch, dry_run=False, positions_after=fresh)
    assert [o["ticker"] for o in orders] == ["AAPL"]
    submit_stop.assert_called_once_with("AAPL", 10, 193.0)
    assert [r["ticker"] for r in audit["placed"]] == ["AAPL"]
    assert [r["ticker"] for r in audit["protected"]] == ["JPM"], "the morning rows survive"


def test_run_entries_post_entry_audit_only_covers_the_new_tickers(monkeypatch):
    """Positions the morning audit already handled are not audited twice —
    JPM has a record with a trail and no resting stop, so a pass over the
    whole book would have placed one for it."""
    fresh = [{"symbol": "AAPL", "qty": "10", "qty_available": "10",
              "market_value": "1980.0", "current_price": "198.0"},
             {"symbol": "JPM", "qty": "12", "qty_available": "12",
              "market_value": "4275.6", "current_price": "356.3"}]
    _, submit_stop, audit = _run_entries_with_one_fill(
        monkeypatch, dry_run=False, positions_after=fresh,
        extra_state={"JPM": {"qty_remaining": 12, "entry_price": 330.0, "trail_stop": 340.0}})
    assert submit_stop.call_count == 1
    assert submit_stop.call_args[0][0] == "AAPL"
    assert not any(r["ticker"] == "JPM" for r in audit["placed"] + audit["unprotected"])


def test_run_entries_reports_a_fill_the_broker_does_not_list_yet(monkeypatch):
    """An assume_fill_on_timeout order may not be a position yet — nothing to
    hold a stop against, so it is named as unprotected, not silently skipped."""
    _, submit_stop, audit = _run_entries_with_one_fill(
        monkeypatch, dry_run=False, positions_after=[])
    submit_stop.assert_not_called()
    assert [r["ticker"] for r in audit["unprotected"]] == ["AAPL"]


def test_run_entries_dry_run_places_no_stop(monkeypatch):
    fresh = [{"symbol": "AAPL", "qty": "10", "qty_available": "10",
              "market_value": "1980.0", "current_price": "198.0"}]
    orders, submit_stop, audit = _run_entries_with_one_fill(
        monkeypatch, dry_run=True, positions_after=fresh)
    assert orders == []
    submit_stop.assert_not_called()
    assert audit["placed"] == []


def test_run_entries_survives_a_post_entry_audit_failure(monkeypatch):
    """A broken post-entry pass must not take the run down with it."""
    fresh = [{"symbol": "AAPL", "qty": "10", "qty_available": "10",
              "market_value": "1980.0", "current_price": "198.0"}]
    with patch("agents.risk_sl_tp_agent.run_stop_audit", side_effect=RuntimeError("audit boom")):
        orders, submit_stop, audit = _run_entries_with_one_fill(
            monkeypatch, dry_run=False, positions_after=fresh)
    assert [o["ticker"] for o in orders] == ["AAPL"]
    assert audit["placed"] == []


def test_run_reports_persists_run_data(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "LOGS_DIR", str(tmp_path))
    save_run = MagicMock()
    monkeypatch.setattr(main_mod, "save_run", save_run)
    monkeypatch.setattr(main_mod, "save_signals", MagicMock())
    with patch("agents.llm_usage.run_summary",
               return_value={"calls": 0, "input": 0, "output": 0,
                             "cache_read": 0, "cost_usd": 0.0}), \
         patch("reports.conviction_backfill.backfill_conviction_returns", return_value=0):
        main_mod.run_reports(
            False, [], [], list(RESEARCH), list(SENTIMENT), dict(RISK),
            {"recommendations": []}, dict(STRATEGY), {"regime": "BULL"}, {}, {})
    assert os.path.exists(tmp_path / "last_run.json")
    assert os.listdir(tmp_path / "runs"), "run should be archived"
    run_data = save_run.call_args[0][0]
    assert run_data["research"] == RESEARCH
    assert run_data["paper_trading"] is False


def test_run_signals_aborts_when_price_data_is_mostly_missing(monkeypatch):
    """A half-empty universe must stop the run, not produce a confident book.

    Every downstream agent degrades politely on missing data — research falls
    back to conviction 1, get_stock_info returns "Unknown" — so a broken feed
    used to read as a genuinely quiet market.
    """
    monkeypatch.setattr(main_mod.config, "MAX_DATA_FAILURE_PCT", 0.20)
    monkeypatch.setattr(main_mod, "TICKERS", ["AAPL", "MSFT", "NVDA", "KO", "PG"])

    sent = []
    with patch("data.fetcher.prefetch_prices", return_value=1), \
         patch("alerts.telegram_alert.send_message", side_effect=lambda t: sent.append(t)):
        with pytest.raises(RuntimeError, match="Insufficient price data"):
            main_mod.run_signals()

    assert sent, "an aborted run must alert; a silent abort is the old failure mode"


def test_run_signals_tolerates_failures_under_the_limit(monkeypatch):
    monkeypatch.setattr(main_mod.config, "MAX_DATA_FAILURE_PCT", 0.20)
    monkeypatch.setattr(main_mod, "TICKERS", ["AAPL", "MSFT", "NVDA", "KO", "PG"])

    with patch("data.fetcher.prefetch_prices", return_value=5), \
         patch("agents.research_agent.run_research", return_value=[]), \
         patch("agents.sentiment_agent.run_sentiment", return_value=[]):
        tickers, _, _, _ = main_mod.run_signals()
    assert len(tickers) == 5
