"""Suite-wide guards against real side effects.

`pytest tests/` sent a real "Degraded Run" alert to the operator's phone on
2026-08-03 15:24: test_run_reports_persists_run_data drives run_reports(),
whose health check calls send_health_alert(), and that test mocks persistence
but not Telegram. Relying on every test author to remember the mock is what
failed; the guard belongs here, once, for everything.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True, scope="session")
def _no_real_telegram():
    """Capture outbound Telegram instead of delivering it.

    Returns True so callers that assert "the alert was sent" still pass — what
    changes is only that nothing leaves the process. Tests that patch
    send_message themselves stack on top of this and are unaffected.
    """
    import alerts.telegram_alert as telegram

    def _guard(text: str, **kwargs) -> bool:
        telegram.SENT_IN_TESTS.append(text)
        return True

    _guard._test_guard = True
    telegram.SENT_IN_TESTS = []
    telegram._REAL_SEND_MESSAGE = telegram.send_message
    telegram.send_message = _guard
    yield
    telegram.send_message = telegram._REAL_SEND_MESSAGE


@pytest.fixture
def unguarded_telegram():
    """Restore the real send_message for tests that exercise the transport itself.

    Only safe because those tests clear TELEGRAM_BOT_TOKEN/CHAT_ID first:
    send_message returns before any network call when it has no credentials.
    """
    import alerts.telegram_alert as telegram
    guard, telegram.send_message = telegram.send_message, telegram._REAL_SEND_MESSAGE
    yield
    telegram.send_message = guard


@pytest.fixture(autouse=True)
def _reset_forbidden_events():
    """FORBIDDEN_EVENTS is a module global that spans a whole run.

    A 403 recorded by test_alpaca_forbidden surfaced inside an unrelated test's
    health alert — the third bullet of the 15:24 message was exactly that.
    """
    from execution import alpaca
    alpaca.FORBIDDEN_EVENTS.clear()
    yield
    alpaca.FORBIDDEN_EVENTS.clear()


@pytest.fixture(autouse=True)
def _no_alpaca_market_data(monkeypatch):
    """Every test starts with Alpaca Market Data unavailable, whatever .env says.

    fetcher prefers Alpaca whenever ALPACA_API_KEY/SECRET are set (PR #12).
    Six fetcher tests mock yfinance and assert on its call count; in the main
    checkout, where .env is always present, they went to Alpaca for real —
    250 rows of live bars fetched from inside a unit test — and failed.
    Worktrees without .env passed, which is how it survived to Sprint 1. The
    network guard then found nine more tests in the same shape (fetcher
    errors, chat paper guard, cost governor), so the "no credentials" state
    every one of them was written against is now the default, not a per-test
    opt-in. Tests of data.alpaca_data itself take real_alpaca_availability.

    Availability is alpaca_data.credentials_present(), read from os.environ on
    every call, so nothing is cached at module level. delenv alone is still
    not enough: thirteen modules run load_project_env(override=True) at import,
    and a lazy import inside a test body would put the keys straight back.
    Patching the function makes the routing decision independent of the
    environment, in either direction. Yields the real function for the opt-in.
    """
    from data import alpaca_data
    real = alpaca_data.credentials_present
    monkeypatch.setattr(alpaca_data, "credentials_present", lambda: False)
    yield real


@pytest.fixture
def real_alpaca_availability(_no_alpaca_market_data, monkeypatch):
    """Opt back in to the env-reading credentials_present().

    For data.alpaca_data's own tests, which set fake keys and mock the
    transport, and need the availability check to actually read them.
    """
    from data import alpaca_data
    monkeypatch.setattr(alpaca_data, "credentials_present", _no_alpaca_market_data)


@pytest.fixture(autouse=True)
def _isolated_price_cache(tmp_path, monkeypatch):
    """Every test gets its own data/cache/.

    The price cache is gitignored, per-checkout and 1-hour TTL, and every test
    that called get_stock_price without pinning _CACHE_DIR read and wrote the
    checkout's real one: a worktree accumulated 131 entries from a run that
    still reached Yahoo, and the next hour of runs passed on them. Tests that
    pin _CACHE_DIR themselves simply re-pin.
    """
    import data.fetcher as fetcher
    monkeypatch.setattr(fetcher, "_CACHE_DIR", tmp_path / "fetcher_cache")


@pytest.fixture(autouse=True)
def _entries_unfrozen(monkeypatch):
    """config.ENTRIES_FROZEN is a transition switch (monthly strategy spec §5),
    not the behaviour the entry and chat tests describe. Every test sees the
    unfrozen engine; tests/test_entries_frozen.py re-pins it to cover the freeze."""
    import config
    monkeypatch.setattr(config, "ENTRIES_FROZEN", False)


@pytest.fixture(autouse=True)
def network_guard(request, monkeypatch):
    """Refuse, and record, every real HTTP request for the duration of a test.

    The six fetcher tests that went to Alpaca for real (see no_alpaca_market_data)
    passed or failed depending on which checkout ran them, and no assertion
    ever said "a request left the process". This makes that a failure in its
    own right, at the transport, for every test at once.

    Three transports because the repo uses three: requests (Alpaca broker and
    data, Telegram, health_check), curl_cffi's requests-alike (what yfinance
    switched to), and urllib.request (sentiment_agent). Patched on the class,
    so sessions built inside library code are covered too.

    The attempt is raised as an outage AND appended to the yielded list. The
    raise alone is not enough: data.fetcher catches everything and falls back,
    so a test with a hidden network dependency would still pass, slowly.

    `@pytest.mark.allow_network` opts a test out. Nothing in the suite uses it
    today; it exists so the next author patches the fetch rather than the guard.
    """
    if request.node.get_closest_marker("allow_network"):
        yield []
        return

    import urllib.request

    import requests
    from curl_cffi import requests as curl_requests

    attempts: list[str] = []

    def _refuse(method: str, url: str) -> None:
        attempts.append(f"{method} {url}")
        raise RuntimeError(f"network call in test: {method} {url}")

    def _refuse_session(self, method, url, *args, **kwargs):
        _refuse(str(method).upper(), str(url))

    def _refuse_urlopen(url, *args, **kwargs):
        target = getattr(url, "full_url", url)
        method = getattr(url, "get_method", lambda: "GET")()
        _refuse(method, str(target))

    monkeypatch.setattr(requests.Session, "request", _refuse_session)
    monkeypatch.setattr(curl_requests.Session, "request", _refuse_session)
    monkeypatch.setattr(urllib.request, "urlopen", _refuse_urlopen)
    yield attempts
    assert not attempts, (
        f"network call in test {request.node.name}: {attempts} — the code under "
        "test swallowed the refusal and fell back; patch the fetch it makes"
    )


@pytest.fixture
def synthetic_regime_market(monkeypatch):
    """A calm BULL market for detect_regime(), served from memory.

    Five smoke tests called detect_regime() bare and read whatever Yahoo
    returned — or, when it returned nothing, the _default_result() fallback,
    which satisfies "vix is a float" just as well. The network guard surfaced
    them. This gives them 260 sessions of SPY/^VIX and a 40-name universe
    (70% above SMA200), and keeps logs/regime_state.json out of the test.
    """
    import numpy as np
    import pandas as pd

    from agents import regime_agent

    n = 260
    idx = pd.bdate_range("2024-01-01", periods=n)
    spy = np.linspace(300, 500, n)
    cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Open"], ["SPY", "^VIX"]])
    raw = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for field in ("Close", "High", "Low", "Open"):
        raw[(field, "SPY")] = spy + (1.0 if field == "High" else -1.0 if field == "Low" else 0.0)
        raw[(field, "^VIX")] = 18.0

    up = {f"UP{i}": np.linspace(100, 200, n) for i in range(28)}
    down = {f"DN{i}": np.linspace(200, 100, n) for i in range(12)}
    universe = pd.DataFrame({**up, **down}, index=idx)

    monkeypatch.setattr(regime_agent.yf, "download", lambda *a, **k: raw)
    monkeypatch.setattr(regime_agent, "_fetch_universe_closes", lambda: universe)
    monkeypatch.setattr(regime_agent, "_load_cache", lambda: {})
    monkeypatch.setattr(regime_agent, "_save_cache", lambda *a, **k: None)
    yield raw


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_db: opt out of the per-test DB/state isolation and read the real logs/ files",
    )
    config.addinivalue_line(
        "markers",
        "allow_network: opt out of the network guard; prefer patching the fetch instead",
    )


@pytest.fixture(autouse=True)
def _isolated_db_and_state(request, tmp_path):
    """Every test gets its own SQLite file and position-state file.

    Ruling from the Task 2 review: with position_state living in the DB,
    a test that forgets to pin db.database.DB_PATH reads — and worse, writes —
    the developer's real logs/hedge_fund.db. Per-test pins in older files keep
    working (they simply re-pin), and `@pytest.mark.real_db` opts a test back
    out when it intentionally reads the real files.
    """
    if request.node.get_closest_marker("real_db"):
        yield
        return
    from db import database
    import execution.order_manager as order_manager

    old_db, old_state = database.DB_PATH, order_manager.STATE_FILE
    database.DB_PATH = str(tmp_path / "isolated_hedge_fund.db")
    order_manager.STATE_FILE = str(tmp_path / "isolated_position_state.json")
    database.init_db()
    try:
        yield
    finally:
        database.DB_PATH = old_db
        order_manager.STATE_FILE = old_state


@pytest.fixture(autouse=True)
def _paper_account(monkeypatch):
    """Broker calls fail closed without PAPER_TRADING=true (execution/alpaca.py).

    The suite exercises the paper broker client against mocked HTTP, so default
    the flag on; tests about the guard itself override it explicitly.
    """
    monkeypatch.setenv("PAPER_TRADING", "true")
