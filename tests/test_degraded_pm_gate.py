"""Quant-only degrade when the PM veto gate did not run (P0.1).

_research_fallback copies the research recommendation through verbatim, so a
research BUY arrives at the executor looking exactly like a PM-approved BUY —
and its `conviction` is a research conviction, which must not satisfy a PM
conviction gate. On a degraded cycle the executor drops the PM conviction
check and instead demands the quant composite stand on its own: strategy_agent
only emits BUY when score >= SIGNAL_THRESHOLD and the credit-stress macro
filter passes.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import config
import main as main_mod

RESEARCH = [{"ticker": "AAPL", "recommendation": "BUY", "conviction": 5,
             "thesis": "Strong.", "key_risks": [], "atr_14": 2.0, "entry_ref": 198.0}]
SENTIMENT = [{"ticker": "AAPL", "sentiment": "POSITIVE", "sentiment_score": 0.5,
              "summary": "Good."}]


def _run(monkeypatch, *, degraded, strategy_signal, score, regime_signal=None):
    monkeypatch.setattr(main_mod, "DRY_RUN", True)
    monkeypatch.setattr(main_mod, "_load_position_metadata", lambda: {})
    monkeypatch.setattr(main_mod, "_save_position_metadata", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "load_pending_signals", lambda: [])
    monkeypatch.setattr(main_mod, "expire_pending_signals", lambda *a, **k: 0)
    # Slot-based sizing (design B) looks up realised vol and sector momentum;
    # a unit test must not reach the network for either.
    monkeypatch.setattr(main_mod.config, "SECTOR_ROTATION_ENABLED", False)
    monkeypatch.setattr("data.fetcher.get_closes", lambda *a, **k: pd.DataFrame())
    bought: list = []
    monkeypatch.setattr(main_mod, "append_conviction_log",
                        lambda *a, **k: bought.append(a[1]))

    pm = {"recommendations": [{"ticker": "AAPL", "action": "BUY", "size_pct": 0.0,
                               "conviction": 5, "rationale": "x"}]}
    if degraded:
        pm["degraded"] = True
    strat = {"AAPL": {"signal": strategy_signal, "composite_score": score}}
    main_mod.run_entries(
        equity=100_000.0, positions=[], exited_tickers=set(), trimmed_tickers=set(),
        closed_positions=[], orders_submitted=[],
        pm_result=pm, sentiment_results=list(SENTIMENT),
        research_results=list(RESEARCH), strategy_signals=strat,
        regime_signals=({"AAPL": {"signal": regime_signal}} if regime_signal else {}),
        regime_result={"regime": "BULL", "new_entries_allowed": True,
                       "size_multiplier": 1.0},
        strategy_score_map={"AAPL": score})
    return bought


def test_healthy_pm_buy_is_taken(monkeypatch):
    bought = _run(monkeypatch, degraded=False, strategy_signal="BUY",
                  score=config.SIGNAL_THRESHOLD + 0.5)
    assert "AAPL" in bought


def test_degraded_with_strong_composite_still_trades(monkeypatch):
    """Quant-only degrade trades on its own signal — it is not a HOLD-all."""
    bought = _run(monkeypatch, degraded=True, strategy_signal="BUY",
                  score=config.SIGNAL_THRESHOLD + 0.5)
    assert "AAPL" in bought


def test_degraded_below_threshold_is_rejected(monkeypatch):
    bought = _run(monkeypatch, degraded=True, strategy_signal="BUY",
                  score=config.SIGNAL_THRESHOLD - 0.1)
    assert bought == [], "degraded cycle must require composite >= SIGNAL_THRESHOLD"


def test_degraded_ignores_regime_only_signal(monkeypatch):
    """Healthy cycles accept strategy OR regime BUY; a degraded cycle needs the
    strategy composite itself, which carries the threshold and macro filter."""
    bought = _run(monkeypatch, degraded=True, strategy_signal="HOLD",
                  score=config.SIGNAL_THRESHOLD + 0.5, regime_signal="BUY")
    assert bought == [], "regime signal alone must not carry a degraded cycle"


def test_healthy_accepts_regime_only_signal(monkeypatch):
    bought = _run(monkeypatch, degraded=False, strategy_signal="HOLD",
                  score=config.SIGNAL_THRESHOLD + 0.5, regime_signal="BUY")
    assert "AAPL" in bought, "healthy behaviour is unchanged"


def test_degraded_ignores_research_conviction(monkeypatch):
    """Research conviction 5 must not substitute for a PM verdict."""
    monkeypatch.setattr(main_mod.config, "MIN_PM_CONVICTION", 5)
    bought = _run(monkeypatch, degraded=True, strategy_signal="HOLD",
                  score=config.SIGNAL_THRESHOLD + 0.5)
    assert bought == []
