"""Every discarded asset must carry a reason.

The gate this covers, main.run_entries' `_buy_candidates`, used to be one list
comprehension testing eight conditions; a name failing any of them vanished
with no print, no log line and no row. Rewriting it as an auditable loop is
only safe if it selects exactly the same survivors, so the first test here
compares production against an independent re-statement of the original
predicate rather than against hand-written expectations.
"""
import os
import sys
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
import main as main_mod
from utils import funnel as fn

EQUITY = 100_000.0


def _drive(monkeypatch, *, pm_recs, strategy_signals, scores, sentiment,
           held=(), exited=(), weight_blocked=(), regime="BULL",
           new_entries_allowed=True, degraded=False):
    """Run the entry phase in dry-run and hand back its funnel recorder."""
    monkeypatch.setattr(main_mod, "DRY_RUN", True)
    monkeypatch.setattr(main_mod, "_load_position_metadata", lambda: {})
    monkeypatch.setattr(main_mod, "_save_position_metadata", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "load_pending_signals", lambda: [])
    monkeypatch.setattr(main_mod, "expire_pending_signals", lambda *a, **k: 0)
    monkeypatch.setattr(main_mod, "append_conviction_log", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "ACCOUNT_SNAPSHOT", {"buying_power": 200_000.0})
    # The sector overlay downloads 11 ETFs and slot sizing looks up realised
    # vol; this file is about the gate, and a unit test must not reach the
    # network to find out who was discarded.
    monkeypatch.setattr(main_mod.config, "SECTOR_ROTATION_ENABLED", False)
    monkeypatch.setattr("data.fetcher.get_closes", lambda *a, **k: pd.DataFrame())

    # A held or over-cap position is expressed the way Alpaca reports it.
    positions = [{"symbol": t, "qty": 1, "market_value": EQUITY * 0.05} for t in held]
    positions += [{"symbol": t, "qty": 1,
                   "market_value": EQUITY * (config.MAX_POSITION_PCT + 0.05)}
                  for t in weight_blocked]

    pm = {"recommendations": pm_recs}
    if degraded:
        pm["degraded"] = True

    recorder = fn.FunnelRecorder("test-run", list(strategy_signals))
    main_mod.run_entries(
        equity=EQUITY, positions=positions,
        exited_tickers=set(exited), trimmed_tickers=set(),
        closed_positions=[], orders_submitted=[],
        pm_result=pm,
        sentiment_results=[{"ticker": t, "sentiment": s} for t, s in sentiment.items()],
        research_results=[], strategy_signals=strategy_signals,
        regime_signals={}, strategy_score_map=scores,
        regime_result={"regime": regime, "new_entries_allowed": new_entries_allowed,
                       "size_multiplier": 1.0},
        funnel=recorder)
    return recorder


def _original_predicate(pm_recs, strategy_signals, scores, sentiment,
                        held, exited, weight_blocked, gate):
    """The eight-condition comprehension as it read before the rewrite."""
    sent_map = dict(sentiment)
    return [
        r["ticker"] for r in pm_recs
        if r.get("action") == "BUY"
        and (r.get("conviction") or 0) >= gate
        and strategy_signals.get(r["ticker"], {}).get("signal") == "BUY"
        and sent_map.get(r["ticker"], "NEUTRAL") != "NEGATIVE"
        and r["ticker"] not in held
        and r["ticker"] not in weight_blocked
        and r["ticker"] not in exited
    ]


def _scenario():
    """One ticker per rejection reason, plus two that should survive."""
    good = config.SIGNAL_THRESHOLD + 0.5
    pm_recs = [
        {"ticker": "AAPL", "action": "BUY",  "conviction": 5, "rationale": ""},
        {"ticker": "MSFT", "action": "BUY",  "conviction": 5, "rationale": ""},
        {"ticker": "NVDA", "action": "HOLD", "conviction": 5, "rationale": ""},
        {"ticker": "KO",   "action": "BUY",  "conviction": 1, "rationale": ""},
        {"ticker": "PEP",  "action": "BUY",  "conviction": 5, "rationale": ""},
        {"ticker": "XOM",  "action": "BUY",  "conviction": 5, "rationale": ""},
        {"ticker": "CVX",  "action": "BUY",  "conviction": 5, "rationale": ""},
        {"ticker": "GS",   "action": "BUY",  "conviction": 5, "rationale": ""},
        {"ticker": "JPM",  "action": "SELL", "conviction": 5, "rationale": ""},
        {"ticker": "BAC",  "action": "BUY",  "conviction": 5, "rationale": ""},
    ]
    tickers = [r["ticker"] for r in pm_recs]
    strategy_signals = {t: {"signal": "BUY", "composite_score": good} for t in tickers}
    strategy_signals["PEP"] = {"signal": "HOLD", "composite_score": 0.0}   # NO_SIGNAL
    scores = {t: v["composite_score"] for t, v in strategy_signals.items()}
    sentiment = dict.fromkeys(tickers, "POSITIVE")
    sentiment["XOM"] = "NEGATIVE"                                          # veto
    # GS is over the per-position cap AND was exited this cycle. That pairing is
    # what makes WEIGHT_CAPPED reachable at all: held_tickers excludes exits, so
    # an over-cap name still held simply reports ALREADY_HELD first.
    return dict(pm_recs=pm_recs, strategy_signals=strategy_signals, scores=scores,
                sentiment=sentiment, held=("CVX",), exited=("BAC", "GS"),
                weight_blocked=("GS",))


def test_rewritten_gate_selects_exactly_the_original_survivors(monkeypatch):
    sc = _scenario()
    recorder = _drive(monkeypatch, **sc)
    expected = _original_predicate(
        sc["pm_recs"], sc["strategy_signals"], sc["scores"], sc["sentiment"],
        sc["held"], sc["exited"], sc["weight_blocked"],
        config.effective_min_conviction("BULL"))
    executed = {r["ticker"] for r in recorder.by_outcome(fn.EXECUTED)}
    queued = {r["ticker"] for r in recorder.by_outcome(fn.QUEUED)}
    assert executed | queued == set(expected)
    assert executed | queued == {"AAPL", "MSFT"}


def test_every_rejected_ticker_names_its_reason(monkeypatch):
    sc = _scenario()
    recorder = _drive(monkeypatch, **sc)
    reasons = {r["ticker"]: r["reason_code"] for r in recorder.by_outcome(fn.REJECTED)}
    assert reasons == {
        "NVDA": "PM_NOT_BUY",
        "KO":   "CONVICTION_BELOW_GATE",
        "PEP":  "NO_SIGNAL",
        "XOM":  "SENTIMENT_NEGATIVE",
        "CVX":  "ALREADY_HELD",
        "GS":   "WEIGHT_CAPPED",
        "JPM":  "PM_SELL",
        "BAC":  "EXITED_TODAY",
    }, "each of the eight conditions must report itself, not a generic verdict"
    assert all(fn.label(c) != c for c in reasons.values()), "every code needs a label"


def test_regime_block_names_the_candidates_it_silenced(monkeypatch):
    """Defense in depth, not a live path: Sprint 1 task 4 changed
    detect_regime() to always return new_entries_allowed=True, and the task 4
    fix round 1 review hardened every _load_cache() fallback so a stale cache
    cannot resurrect False either (agents.regime_agent._sanitize_cached). No
    real detect_regime() call can produce new_entries_allowed=False any more.

    run_entries still honors it if it is ever handed one — via a hand-built
    regime dict, as this test does, not through detect_regime — so this
    covers that the REJECTED/REGIME_BLOCKED bookkeeping still works should
    any future caller (or a bug) reintroduce a blocked regime.
    """
    sc = _scenario()
    recorder = _drive(monkeypatch, **sc, new_entries_allowed=False, regime="BULL")
    blocked = [r for r in recorder.by_outcome(fn.REJECTED)
               if r["reason_code"] == "REGIME_BLOCKED"]
    assert blocked, "a closed gate must still say which tickers it stopped"
    assert recorder.summary()["bottleneck"]["code"] == "REGIME_BLOCKED"


def test_cooldown_rejection_is_recorded(monkeypatch):
    """A name stopped out within REENTRY_COOLDOWN_DAYS trading days is refused
    with its own reason code — re-buying HON 5× within days of each stop is
    the churn design C exists to end."""
    from datetime import date, timedelta

    from db import database
    sc = _scenario()
    # AAPL was stopped out yesterday; MSFT never was.
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    database.record_exit("AAPL", f"{yesterday}T15:55:00", "STOP_TRAIL", 10, 100.0, "run")
    recorder = _drive(monkeypatch, **sc)
    reasons = {r["ticker"]: r["reason_code"] for r in recorder.by_outcome(fn.REJECTED)}
    assert reasons["AAPL"] == "COOLDOWN"
    assert fn.label("COOLDOWN") != "COOLDOWN"
    survivors = {r["ticker"] for r in recorder.by_outcome(fn.EXECUTED)} | \
                {r["ticker"] for r in recorder.by_outcome(fn.QUEUED)}
    assert survivors == {"MSFT"}


def test_gross_target_label_exists_for_the_sizing_task():
    assert fn.label("GROSS_TARGET_REACHED") != "GROSS_TARGET_REACHED"
    assert "CRITICAL_BLOCK" not in fn.REASON_LABELS


def test_summary_ranks_reasons_and_names_the_bottleneck():
    f = fn.FunnelRecorder("r", ["A", "B", "C", "D"])
    f.record_many(["A", "B", "C"], fn.REJECTED, "NO_SIGNAL")
    f.record("D", fn.REJECTED, "SENTIMENT_NEGATIVE")
    s = f.summary()
    assert s["bottleneck"]["code"] == "NO_SIGNAL"
    assert s["bottleneck"]["count"] == 3
    assert [r["code"] for r in s["reasons"]] == ["NO_SIGNAL", "SENTIMENT_NEGATIVE"]
    assert s["rejected_total"] == 4


def test_summary_exposes_queued_reasons_separately_from_rejected():
    """NO_BUDGET is recorded QUEUED, not REJECTED — a deferred name, not a
    turned-away one — so it must show up in its own bucket, not silently
    disappear from the summary and not count toward `reasons`/`bottleneck`.
    """
    f = fn.FunnelRecorder("r", ["A", "B", "C"])
    f.record("A", fn.QUEUED, "NO_BUDGET")
    f.record("B", fn.QUEUED, "NO_BUDGET")
    f.record("C", fn.REJECTED, "GROSS_TARGET_REACHED")
    s = f.summary()
    assert [r["code"] for r in s["queued_reasons"]] == ["NO_BUDGET"]
    assert s["queued_reasons"][0]["count"] == 2
    assert s["queued_reasons"][0]["tickers"] == ["A", "B"]
    # NO_BUDGET must not leak into the rejected-only view.
    assert [r["code"] for r in s["reasons"]] == ["GROSS_TARGET_REACHED"]
    assert s["bottleneck"]["code"] == "GROSS_TARGET_REACHED"


def test_later_stage_overwrites_the_earlier_verdict():
    """The funnel is sequential: what stopped a ticker is the LAST thing recorded."""
    f = fn.FunnelRecorder("r", ["A"])
    f.record("A", fn.REJECTED, "NO_SIGNAL")
    f.record("A", fn.EXECUTED, "EXECUTED")
    assert f.by_outcome(fn.EXECUTED) and not f.by_outcome(fn.REJECTED)


def test_funnel_rows_survive_a_database_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("HEDGE_FUND_DB", str(tmp_path / "t.db"))
    from db import database
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "t.db"), raising=False)
    database.init_db()

    f = fn.FunnelRecorder("2026-08-31T09:45", ["AAPL", "KO"])
    f.record("AAPL", fn.EXECUTED, "EXECUTED", score=0.8, conviction=5)
    f.record("KO", fn.REJECTED, "CONVICTION_BELOW_GATE", conviction=2)
    database.save_funnel_log(f.run_timestamp, f.rows())

    stored = {r["ticker"]: r["reason_code"]
              for r in database.load_funnel_log("2026-08-31T09:45")}
    assert stored == {"AAPL": "EXECUTED", "KO": "CONVICTION_BELOW_GATE"}

    # Re-running the same cycle must replace, not stack.
    database.save_funnel_log(f.run_timestamp, f.rows())
    assert len(database.load_funnel_log("2026-08-31T09:45")) == 2


def test_telegram_report_survives_a_universe_sized_funnel():
    """Telegram hard-rejects a message over 4096 chars; the report must fit."""
    import alerts.telegram_alert as telegram

    f = fn.FunnelRecorder("r", [f"TCK{i}" for i in range(150)])
    f.record_many([f"TCK{i}" for i in range(150)], fn.REJECTED, "NO_SIGNAL")
    sent = []
    with patch.object(telegram, "send_message", side_effect=lambda t: sent.append(t) or True), \
         patch.object(telegram, "BOT_TOKEN", "x"), patch.object(telegram, "CHAT_ID", "y"):
        assert telegram.send_funnel_report(f.summary(), regime="BULL")
    assert len(sent[0]) <= 4096
    assert "Cuello de botella" in sent[0]
