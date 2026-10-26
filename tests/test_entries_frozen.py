"""ENTRIES_FROZEN: the daily engine stops buying while the monthly strategy is built.

Spec docs/superpowers/specs/2026-10-10-monthly-core-satellite-design.md §5:
from now until the first monthly rebalance the pipeline places no new buys,
while exits and broker stops keep running. A frozen run has zero orders by
design, so the health check must not page about it every morning.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
import main as main_mod
from chat import pretrade

_PM_BUY = {"recommendations": [
    {"ticker": "AAPL", "action": "BUY", "size_pct": 0.0, "conviction": 4, "rationale": "r"},
]}
_STRATEGY_BUY = {"AAPL": {"signal": "BUY"}}


def _stub_phases(monkeypatch):
    """Stub every phase _run() drives; return the mocks to assert on."""
    monkeypatch.setattr(main_mod, "PAPER_TRADING", True)
    monkeypatch.setattr(main_mod, "run_signals",
                        MagicMock(return_value=(["AAPL"], [], [], {})))
    monkeypatch.setattr(main_mod, "run_gates",
                        MagicMock(return_value=({}, {"regime": "BULL"}, {}, dict(_STRATEGY_BUY),
                                                {"AAPL": 1.0}, [])))
    monkeypatch.setattr(main_mod, "run_decisions", MagicMock(return_value=dict(_PM_BUY)))
    positions = [{"symbol": "AAPL", "market_value": "1000"}]
    run_exits = MagicMock(return_value=(100_000.0, positions, set(), [], {}, set(), []))
    run_entries = MagicMock(return_value=([], []))
    run_reports = MagicMock()
    persist = MagicMock()
    monkeypatch.setattr(main_mod, "run_exits", run_exits)
    monkeypatch.setattr(main_mod, "run_entries", run_entries)
    monkeypatch.setattr(main_mod, "run_reports", run_reports)
    monkeypatch.setattr(main_mod, "_persist_holding_days", persist, raising=False)
    return run_exits, run_entries, run_reports, persist, positions


def test_frozen_run_skips_entries_but_still_runs_exits(monkeypatch):
    monkeypatch.setattr(config, "ENTRIES_FROZEN", True, raising=False)
    run_exits, run_entries, run_reports, persist, positions = _stub_phases(monkeypatch)
    with patch("alerts.telegram_alert.send_daily_summary", return_value=True):
        main_mod._run()
    run_exits.assert_called_once()
    run_entries.assert_not_called()
    run_reports.assert_called_once()
    assert run_reports.call_args.kwargs.get("entries_frozen") is True


def test_frozen_run_still_ages_tracked_positions(monkeypatch):
    """holding_days used to be bumped at the tail of run_entries; skipping
    that phase must not freeze the counter too."""
    monkeypatch.setattr(config, "ENTRIES_FROZEN", True, raising=False)
    _, _, _, persist, positions = _stub_phases(monkeypatch)
    with patch("alerts.telegram_alert.send_daily_summary", return_value=True):
        main_mod._run()
    persist.assert_called_once_with(positions, set(), [])


def test_unfrozen_run_still_enters(monkeypatch):
    monkeypatch.setattr(config, "ENTRIES_FROZEN", False, raising=False)
    _, run_entries, run_reports, persist, _ = _stub_phases(monkeypatch)
    with patch("alerts.telegram_alert.send_daily_summary", return_value=True):
        main_mod._run()
    run_entries.assert_called_once()
    persist.assert_not_called()   # run_entries does its own aging
    assert run_reports.call_args.kwargs.get("entries_frozen") is False


def test_persist_holding_days_bumps_held_and_drops_closed(monkeypatch):
    saved = {}
    monkeypatch.setattr(main_mod, "_load_position_metadata",
                        lambda: {"AAPL": {"holding_days": 3}, "OLD": {"holding_days": 9}})
    monkeypatch.setattr(main_mod, "_save_position_metadata", lambda m: saved.update(m))
    main_mod._persist_holding_days([{"symbol": "AAPL"}], set(), [])
    assert saved == {"AAPL": {"holding_days": 4}}


def test_persist_holding_days_keeps_todays_fills(monkeypatch):
    saved = {}
    monkeypatch.setattr(main_mod, "_load_position_metadata",
                        lambda: {"MSFT": {"holding_days": 0}})
    monkeypatch.setattr(main_mod, "_save_position_metadata", lambda m: saved.update(m))
    main_mod._persist_holding_days([], set(), [{"ticker": "MSFT"}])
    assert saved == {"MSFT": {"holding_days": 0}}


def _health(entries_frozen):
    from execution import alpaca
    with patch.object(alpaca, "FORBIDDEN_EVENTS", []):
        return main_mod._collect_health_issues(
            dict(_PM_BUY), [], dict(_STRATEGY_BUY), {"new_entries_allowed": True}, [],
            entries_frozen=entries_frozen)


def test_frozen_zero_orders_is_not_a_health_issue():
    assert not any("STEP C" in i for i in _health(entries_frozen=True))


def test_unfrozen_zero_orders_still_alerts():
    assert any("STEP C" in i for i in _health(entries_frozen=False))


def _chat_gates(kind, monkeypatch):
    monkeypatch.setattr(pretrade, "pipeline_is_running", lambda: False)
    monkeypatch.setattr(pretrade, "order_state_known", lambda: True)
    return pretrade.evaluate_chat_gates(kind=kind, ticker="AAPL",
                                        ctx={"live_positions": []})


def test_chat_buy_is_flagged_while_frozen(monkeypatch):
    """Soft, not hard: the freeze is a strategy call, and the chat's rule is
    that a person may overrule a strategy call in writing."""
    monkeypatch.setattr(config, "ENTRIES_FROZEN", True, raising=False)
    frozen = [r for r in _chat_gates("buy", monkeypatch) if r.reason_code == "ENTRIES_FROZEN"]
    assert len(frozen) == 1
    assert frozen[0].severity == "soft" and not frozen[0].passed


def test_chat_sell_is_not_affected_by_the_freeze(monkeypatch):
    monkeypatch.setattr(config, "ENTRIES_FROZEN", True, raising=False)
    assert not any(r.reason_code == "ENTRIES_FROZEN" for r in _chat_gates("close", monkeypatch))


def test_chat_buy_passes_the_freeze_gate_when_unfrozen(monkeypatch):
    monkeypatch.setattr(config, "ENTRIES_FROZEN", False, raising=False)
    assert not any(r.reason_code == "ENTRIES_FROZEN" for r in _chat_gates("buy", monkeypatch))
