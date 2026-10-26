"""A run that completes without trading must not look like a healthy run.

Eleven consecutive runs printed "Run complete" while submitting zero orders,
producing no PM output and throwing 403s on every write. Nothing reached
Telegram. These tests pin each degradation to a named, stage-tagged alert.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main as main_mod
from alerts.telegram_alert import send_health_alert
from utils import funnel as fn

_PM_OK = {"recommendations": [
    {"ticker": "AAPL", "action": "BUY", "size_pct": 0.0, "conviction": 4, "rationale": "r"},
]}
_STRATEGY_BUY = {"AAPL": {"signal": "BUY"}}
_ORDER = [{"ticker": "AAPL", "qty": 10, "price": 190.0}]
_REGIME_OPEN = {"new_entries_allowed": True}
_REGIME_BLOCKED = {"new_entries_allowed": False}


def _collect(pm=None, orders=None, strategy=None, regime=None, failed_trims=(), forbidden=(),
             research_stats=None):
    from execution import alpaca
    with patch.object(alpaca, "FORBIDDEN_EVENTS", list(forbidden)):
        return main_mod._collect_health_issues(
            _PM_OK if pm is None else pm,
            _ORDER if orders is None else orders,
            dict(_STRATEGY_BUY) if strategy is None else strategy,
            dict(_REGIME_OPEN) if regime is None else regime,
            list(failed_trims),
            research_stats=research_stats,
        )


def test_healthy_run_produces_no_issues():
    assert _collect() == []


def test_empty_pm_recommendations_alerts():
    issues = _collect(pm={})
    assert len(issues) >= 1
    assert "STEP 4" in issues[0] and "ZERO recommendations" in issues[0]


def test_degraded_pm_fallback_alerts():
    pm = dict(_PM_OK, degraded=True)
    issues = _collect(pm=pm)
    assert any("fell back to raw research" in i and "STEP 4" in i for i in issues)


def test_research_chunk_fallback_alerts():
    """A research run where enough chunks fell back to HOLD must say so.

    research_stats is passed in explicitly (same pattern as `budget` below),
    not read from agents.research_agent.LAST_RESEARCH_STATS — see
    test_research_stats_ignores_stale_module_state for the regression this
    guards against.
    """
    research_stats = {
        "tickers": 120, "chunks": 6, "chunks_failed": 4,
        "fallback_tickers": 80, "degraded": True,
    }
    issues = _collect(research_stats=research_stats)
    assert any("research" in i.lower() and "80" in i and "120" in i for i in issues)


def test_research_stats_not_degraded_is_silent():
    research_stats = {
        "tickers": 120, "chunks": 6, "chunks_failed": 1,
        "fallback_tickers": 20, "degraded": False,
    }
    assert _collect(research_stats=research_stats) == []


def test_research_stats_ignores_stale_module_state():
    """Regression: _collect_health_issues must not fall back to reading
    agents.research_agent.LAST_RESEARCH_STATS when research_stats is omitted.

    Reproduces the bug directly: pytest tests/test_research_chunking.py
    tests/test_health_alert.py::test_healthy_run_produces_no_issues used to
    fail in that order because test_research_chunking.py's runs leave
    LAST_RESEARCH_STATS mutated, and the old lazy-import implementation read
    it as module state. Mirrors test_health_check_ignores_module_state for
    LAST_BUDGET below.
    """
    import agents.research_agent as ra

    before = dict(ra.LAST_RESEARCH_STATS)
    ra.LAST_RESEARCH_STATS.clear()
    ra.LAST_RESEARCH_STATS.update({
        "tickers": 120, "chunks": 6, "chunks_failed": 6,
        "fallback_tickers": 120, "degraded": True,
    })
    try:
        assert _collect() == []
    finally:
        ra.LAST_RESEARCH_STATS.clear()
        ra.LAST_RESEARCH_STATS.update(before)


def test_forbidden_events_alert_with_symbol_and_endpoint():
    events = [
        {"method": "DELETE", "endpoint": "/v2/positions/LLY", "symbol": "LLY",
         "detail": "insufficient qty available for order (requested: 5, available: 0)"},
        {"method": "POST", "endpoint": "/v2/orders", "symbol": "NVDA", "detail": "forbidden"},
    ]
    issues = _collect(forbidden=events)
    alert = next(i for i in issues if "403" in i)
    assert "2 request(s) refused 403" in alert
    assert "/v2/positions/LLY" in alert and "LLY" in alert
    assert "/v2/orders" in alert and "NVDA" in alert
    assert "insufficient qty available" in alert


def test_failed_trim_over_cap_alerts():
    failed = [{"ticker": "LLY", "weight": 0.323, "shares": 5,
               "error": "403 Forbidden — insufficient qty"}]
    issues = _collect(failed_trims=failed)
    alert = next(i for i in issues if "drift trim" in i)
    assert "LLY" in alert and "32.3%" in alert and "FAILED" in alert


def test_zero_orders_with_buy_signals_and_entries_allowed_alerts():
    issues = _collect(orders=[])
    alert = next(i for i in issues if "entries" in i)
    assert "zero orders submitted" in alert
    assert "should have traded and did not" in alert
    assert "AAPL" in alert


def test_zero_orders_is_not_an_issue_when_entries_are_blocked():
    """A regime block is a decision, not a fault — it must not cry wolf."""
    issues = _collect(orders=[], regime=dict(_REGIME_BLOCKED))
    assert not any("entries" in i for i in issues)


def test_zero_orders_is_not_an_issue_without_buy_signals():
    issues = _collect(orders=[], strategy={"AAPL": {"signal": "HOLD"}})
    assert issues == []


def test_zero_orders_names_the_conviction_gate_that_blocked_every_buy():
    """Run 2026-08-03: 4 PM BUYs at conviction 3, regime gate 4, 0 orders.

    "the pipeline should have traded and did not" was true but useless — the
    alert has to say WHICH gate ate the trades. The regime that carries a
    raised bar is now BEAR (config.BEAR_MIN_CONVICTION); the drift risk this
    test pins — the alert must name the gate, not just say "no orders" — is
    unchanged from the SIDEWAYS run that originally motivated it.
    """
    pm = {"recommendations": [
        {"ticker": "AAPL", "action": "BUY", "size_pct": 0.0, "conviction": 3, "rationale": "r"},
        {"ticker": "JNJ", "action": "BUY", "size_pct": 0.0, "conviction": 3, "rationale": "r"},
    ]}
    issues = _collect(pm=pm, orders=[], regime={"new_entries_allowed": True,
                                               "regime": "BEAR"})
    alert = next(i for i in issues if "entries" in i)
    assert "BEAR" in alert
    assert "conviction gate" in alert and ">= 4" in alert
    assert "AAPL(3)" in alert and "JNJ(3)" in alert
    assert "should have traded and did not" not in alert


def test_zero_orders_reports_when_pm_approved_no_buys():
    pm = {"recommendations": [
        {"ticker": "AAPL", "action": "HOLD", "size_pct": 0.0, "conviction": 2, "rationale": "r"},
    ]}
    issues = _collect(pm=pm, orders=[])
    alert = next(i for i in issues if "entries" in i)
    assert "PM approved no BUY" in alert


def test_zero_orders_still_flags_an_unexplained_miss():
    """A cleared candidate that still did not trade is the original fault."""
    issues = _collect(orders=[], regime={"new_entries_allowed": True, "regime": "BULL"})
    alert = next(i for i in issues if "entries" in i)
    assert "should have traded and did not" in alert


def test_stacked_failures_are_all_reported():
    """The defects were stacked; reporting only the first would hide the rest."""
    events = [{"method": "POST", "endpoint": "/v2/orders", "symbol": "LLY", "detail": "nope"}]
    failed = [{"ticker": "LLY", "weight": 0.323, "shares": 5, "error": "403"}]
    issues = _collect(pm={}, orders=[], failed_trims=failed, forbidden=events)
    assert len(issues) == 4


def test_send_health_alert_noop_on_empty_issues():
    with patch("alerts.telegram_alert.send_message") as mock_send:
        assert send_health_alert([]) is False
    mock_send.assert_not_called()


def test_send_health_alert_message_names_stage_and_fault():
    with patch("alerts.telegram_alert.BOT_TOKEN", "t"), \
         patch("alerts.telegram_alert.CHAT_ID", "c"), \
         patch("alerts.telegram_alert.send_message", return_value=True) as mock_send:
        assert send_health_alert(["STEP 4 (PM agent): returned ZERO recommendations"]) is True
    text = mock_send.call_args.args[0]
    assert "Degraded Run" in text
    assert "STEP 4 (PM agent): returned ZERO recommendations" in text


def test_send_health_alert_skips_when_telegram_unconfigured():
    with patch("alerts.telegram_alert.BOT_TOKEN", ""), \
         patch("alerts.telegram_alert.CHAT_ID", ""), \
         patch("alerts.telegram_alert.send_message") as mock_send:
        assert send_health_alert(["something broke"]) is False
    mock_send.assert_not_called()


# ── Budget ──────────────────────────────────────────────────────────────
# The budget is passed in rather than read from main.LAST_BUDGET. It used to be
# read from that module global, which made this function judge whichever run had
# touched it last — the three tests above started failing the moment any other
# test in the session drove run_entries.

def test_unusable_budget_base_is_reported():
    """A failed account read must not quietly halve the budget.

    Falling back to the equity basis roughly halves what the cycle may deploy,
    and from outside that is indistinguishable from a cautious day.
    """
    budget = {"degraded": True, "reason": "buying power unavailable or zero",
              "deployable": 52_000, "leftover": 0, "base": "buying_power"}
    issues = main_mod._collect_health_issues(
        _PM_OK, _ORDER, dict(_STRATEGY_BUY), dict(_REGIME_OPEN), [], budget=budget)
    assert any("BUDGET" in i and "buying power" in i for i in issues)


def test_largely_undeployed_gross_exposure_with_a_rejected_candidate_is_reported():
    """Slot-based sizing (design B): the alert fires on gross exposure vs its
    target, not on leftover cash — and only when the funnel says a candidate
    was actually turned away for lack of room or budget.

    NO_BUDGET is recorded QUEUED (main.py), not REJECTED — a real
    FunnelRecorder puts it in summary()["queued_reasons"], not ["reasons"] —
    so this drives the real recorder rather than hand-building a dict shaped
    like the wrong outcome bucket.
    """
    budget = {"degraded": False, "reason": "", "target_gross_dollars": 130_000,
              "gross_after": 20_000}
    recorder = fn.FunnelRecorder("r", ["AAPL"])
    recorder.record("AAPL", fn.QUEUED, "NO_BUDGET")
    funnel_summary = recorder.summary()
    assert funnel_summary["reasons"] == [], "NO_BUDGET must not land in reasons"
    issues = main_mod._collect_health_issues(
        _PM_OK, _ORDER, dict(_STRATEGY_BUY), dict(_REGIME_OPEN), [], budget=budget,
        funnel_summary=funnel_summary)
    assert any("gross exposure" in i and "target" in i for i in issues)


def test_largely_undeployed_gross_exposure_reported_for_gross_target_reached_too():
    budget = {"degraded": False, "reason": "", "target_gross_dollars": 130_000,
              "gross_after": 20_000}
    funnel_summary = {"reasons": [{"code": "GROSS_TARGET_REACHED", "count": 2}]}
    issues = main_mod._collect_health_issues(
        _PM_OK, _ORDER, dict(_STRATEGY_BUY), dict(_REGIME_OPEN), [], budget=budget,
        funnel_summary=funnel_summary)
    assert any("gross exposure" in i for i in issues)


def test_fully_deployed_gross_exposure_is_silent():
    budget = {"degraded": False, "reason": "", "target_gross_dollars": 130_000,
              "gross_after": 125_000}
    funnel_summary = {"reasons": [{"code": "NO_BUDGET", "count": 1}]}
    assert main_mod._collect_health_issues(
        _PM_OK, _ORDER, dict(_STRATEGY_BUY), dict(_REGIME_OPEN), [], budget=budget,
        funnel_summary=funnel_summary) == []


def test_low_gross_exposure_is_silent_without_a_rejected_candidate():
    """A quiet market (nothing worth buying) or a regime multiplier that
    deliberately shrinks every slot both leave gross exposure low without
    ever rejecting a candidate on budget grounds — not a fault."""
    budget = {"degraded": False, "reason": "", "target_gross_dollars": 130_000,
              "gross_after": 20_000}
    assert main_mod._collect_health_issues(
        _PM_OK, _ORDER, dict(_STRATEGY_BUY), dict(_REGIME_OPEN), [], budget=budget,
        funnel_summary={}) == []


def test_health_check_ignores_module_state():
    """Regression: no budget argument means no budget verdict, whatever the
    module global happens to hold from another cycle."""
    main_mod.LAST_BUDGET.clear()
    main_mod.LAST_BUDGET.update({"degraded": True, "deployable": 1, "leftover": 1})
    try:
        assert _collect() == []
    finally:
        main_mod.LAST_BUDGET.clear()


def test_undeployed_gross_exposure_is_silent_when_the_regime_shrank_sizes():
    """A deliberate de-risking is not a fault.

    A 0.25 regime multiplier (e.g. VIX > 35) shrinks every slot rather than
    rejecting a candidate for lack of room or budget, so it never produces a
    GROSS_TARGET_REACHED/NO_BUDGET funnel entry — the new rule is silent by
    construction, with no need to read size_multiplier directly.
    """
    budget = {"degraded": False, "reason": "", "target_gross_dollars": 130_000,
              "gross_after": 32_500}
    regime = {"new_entries_allowed": True, "size_multiplier": 0.25}
    assert main_mod._collect_health_issues(
        _PM_OK, _ORDER, dict(_STRATEGY_BUY), regime, [], budget=budget,
        funnel_summary={}) == []


def test_undeployed_gross_exposure_still_alerts_at_full_regime_size():
    """GROSS_TARGET_REACHED is recorded REJECTED, so it lands in
    summary()["reasons"] proper — the other half of the union this rule reads
    from a real FunnelRecorder."""
    budget = {"degraded": False, "reason": "", "target_gross_dollars": 130_000,
              "gross_after": 32_500}
    regime = {"new_entries_allowed": True, "size_multiplier": 1.0}
    recorder = fn.FunnelRecorder("r", ["AAPL", "MSFT", "NVDA"])
    recorder.record_many(["AAPL", "MSFT", "NVDA"], fn.REJECTED, "GROSS_TARGET_REACHED")
    funnel_summary = recorder.summary()
    issues = main_mod._collect_health_issues(
        _PM_OK, _ORDER, dict(_STRATEGY_BUY), regime, [], budget=budget,
        funnel_summary=funnel_summary)
    assert any("gross exposure" in i for i in issues)


def test_full_book_zero_orders_is_silent():
    """Reviewer probe (I1): once the book reaches its gross target, every run
    submits zero orders with a PM BUY that cleared the gate and a funnel that
    says GROSS_TARGET_REACHED — that is the design working, not "the pipeline
    should have traded and did not". It fired every single day."""
    pm = {"recommendations": [{"ticker": "AAPL", "action": "BUY", "conviction": 4}], "degraded": False}
    strategy = {"AAPL": {"signal": "BUY"}}
    regime = {"regime": "BULL", "new_entries_allowed": True}
    budget = {"degraded": False, "reason": "", "target_gross_dollars": 120_000, "gross_after": 121_000}
    funnel_summary = {"reasons": [{"code": "GROSS_TARGET_REACHED", "count": 1}]}
    issues = main_mod._collect_health_issues(pm, [], strategy, regime, [], budget=budget,
                                             funnel_summary=funnel_summary)
    assert not any("should have traded" in i for i in issues)


def test_full_book_by_gross_alone_is_silent_without_a_funnel_reason():
    """The second half of the rule: gross_after >= 95% of target is a full
    book even when the funnel summary did not carry the reason code (a
    candidate that failed an earlier gate, say)."""
    budget = {"degraded": False, "reason": "", "target_gross_dollars": 120_000, "gross_after": 114_000}
    issues = _collect(orders=[], regime={"new_entries_allowed": True, "regime": "BULL"})
    assert any("should have traded" in i for i in issues), "sanity: fires without the budget"
    issues = main_mod._collect_health_issues(
        _PM_OK, [], dict(_STRATEGY_BUY), {"new_entries_allowed": True, "regime": "BULL"}, [],
        budget=budget, funnel_summary={})
    assert not any("should have traded" in i for i in issues)


def test_queued_gross_target_reached_also_silences_the_rule():
    """The reason may sit in queued_reasons rather than reasons."""
    budget = {"degraded": False, "reason": "", "target_gross_dollars": 120_000, "gross_after": 60_000}
    funnel_summary = {"reasons": [], "queued_reasons": [{"code": "GROSS_TARGET_REACHED", "count": 1}]}
    issues = main_mod._collect_health_issues(
        _PM_OK, [], dict(_STRATEGY_BUY), {"new_entries_allowed": True, "regime": "BULL"}, [],
        budget=budget, funnel_summary=funnel_summary)
    assert not any("should have traded" in i for i in issues)


def test_short_book_zero_orders_still_flags_the_miss():
    """A book at 50% of target, no GROSS_TARGET_REACHED, a cleared PM BUY and
    zero orders is still the original unexplained miss."""
    budget = {"degraded": False, "reason": "", "target_gross_dollars": 120_000, "gross_after": 60_000}
    funnel_summary = {"reasons": [{"code": "NO_SIGNAL", "count": 1}]}
    issues = main_mod._collect_health_issues(
        _PM_OK, [], dict(_STRATEGY_BUY), {"new_entries_allowed": True, "regime": "BULL"}, [],
        budget=budget, funnel_summary=funnel_summary)
    assert any("should have traded" in i for i in issues)
