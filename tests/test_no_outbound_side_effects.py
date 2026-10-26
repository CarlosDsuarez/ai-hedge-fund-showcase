"""The test suite must not reach the operator's phone or the live broker.

On 2026-08-03 15:24 a "Degraded Run" alert arrived on the operator's phone.
No run had happened — it was `pytest tests/`.
test_run_reports_persists_run_data calls run_reports(), whose health check ends
in send_health_alert(), and nothing in that test mocked the Telegram layer.

It had working credentials because the bare `load_dotenv(override=True)` in
execution/alpaca.py resolves its .env by walking UP from the source file. Run
from a git worktree under .claude/worktrees/, that walk escapes the worktree
and loads the parent checkout's production .env — live Telegram token AND live
Alpaca keys — into a test process.
"""
import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = Path(__file__).resolve().parent.parent


def _source_files():
    for path in REPO_ROOT.rglob("*.py"):
        parts = path.relative_to(REPO_ROOT).parts
        if parts[0] in {".git", ".claude", "tests", "logs"}:
            continue
        yield path


def test_no_module_resolves_its_env_by_walking_up_the_tree():
    """Every load_dotenv must name an explicit path anchored at the repo root.

    A bare load_dotenv() climbs out of a worktree into the parent checkout and
    hands production credentials to whatever is running.
    """
    offenders = []
    for path in _source_files():
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "load_dotenv":
                continue
            explicit = node.args or any(k.arg == "dotenv_path" for k in node.keywords)
            if not explicit:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        "load_dotenv() without an explicit path walks up out of a worktree and "
        f"loads the parent checkout's production .env: {offenders}"
    )


def test_project_env_path_is_the_repo_root():
    from utils.env import PROJECT_ENV, load_project_env
    assert PROJECT_ENV == REPO_ROOT / ".env"
    load_project_env()  # must not raise when the file is absent (worktrees)


def test_telegram_send_is_stubbed_for_the_whole_suite():
    """The conftest guard, not each test author, is what keeps messages in."""
    import alerts.telegram_alert as telegram
    assert getattr(telegram.send_message, "_test_guard", False), (
        "alerts.telegram_alert.send_message is not the conftest guard — a test "
        "that forgets to mock it will message the operator for real"
    )


def test_run_reports_health_alert_does_not_leave_the_process(tmp_path, monkeypatch):
    """Reproduction of the 15:24 alert: it must now be captured, not sent.

    Credentials are forced present on purpose. A worktree without its own .env
    already comes up unconfigured, which would make this pass for the wrong
    reason — what is under test here is the conftest stub, the layer that has
    to hold in the real checkout where the token IS configured.
    """
    from unittest.mock import MagicMock, patch

    import main as main_mod
    import alerts.telegram_alert as telegram

    monkeypatch.setattr(telegram, "BOT_TOKEN", "test-token")
    monkeypatch.setattr(telegram, "CHAT_ID", "test-chat")
    monkeypatch.setattr(main_mod, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(main_mod, "save_run", MagicMock())
    monkeypatch.setattr(main_mod, "save_signals", MagicMock())

    with patch("agents.llm_usage.run_summary",
               return_value={"calls": 0, "input": 0, "output": 0,
                             "cache_read": 0, "cost_usd": 0.0}), \
         patch("reports.conviction_backfill.backfill_conviction_returns", return_value=0):
        main_mod.run_reports(
            False, [], [], [{"ticker": "AAPL", "recommendation": "BUY",
                             "conviction": 4, "thesis": "t", "key_risks": []}],
            [], {"status": "OK", "violations": []},
            {"recommendations": []}, {"AAPL": {"signal": "BUY"}},
            {"regime": "BULL"}, {}, {})

    assert telegram.SENT_IN_TESTS, "the degraded-run alert should still be produced"
    assert "Degraded Run" in telegram.SENT_IN_TESTS[-1]


def test_forbidden_events_do_not_leak_between_tests():
    """A 403 asserted in one test showed up in another test's health alert."""
    from execution import alpaca
    assert alpaca.FORBIDDEN_EVENTS == []
    alpaca.FORBIDDEN_EVENTS.append({"method": "GET", "endpoint": "/v2/account",
                                    "symbol": None, "detail": ""})


def test_forbidden_events_are_empty_again_in_the_next_test():
    from execution import alpaca
    assert alpaca.FORBIDDEN_EVENTS == []
