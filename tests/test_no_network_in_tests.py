"""No test may reach the network. The conftest guard, not each author, enforces it.

Six fetcher tests (test_fetcher_cache, test_fetcher_errors, test_prefetch_prices)
mocked yf.download and yf.Ticker and were green for months in worktrees. In the
main checkout, where .env holds Alpaca keys, data.fetcher took its preferred
Alpaca leg instead: a unit test asserting `result is None` got back
`DataFrame 250 rows x 5 columns` of live IEX bars. Nothing failed loudly at the
point of the request — it just quietly worked, which is the worst outcome for a
test.

The guard in conftest refuses every real HTTP transport this repo uses:
`requests.Session.request` (Alpaca, Telegram, health_check), the curl_cffi
session yfinance moved to, and `urllib.request.urlopen` (sentiment_agent). It
raises so production code sees an outage, and it records the attempt so that
a caller which swallows the exception and falls back still fails the test.
"""
import contextlib
import os
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

import pytest
import requests
from curl_cffi import requests as curl_requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = Path(__file__).resolve().parent.parent

# Port 9 (discard) on loopback: refused instantly, and never leaves the host.
# If the guard is missing, the attempt fails with a connection error rather
# than the guard's RuntimeError — the wrong exception, and a visible one.
LOCAL_DEAD_END = "http://127.0.0.1:9/"

# Captured at collection, before any fixture has run: the unpatched transports.
_REAL_REQUESTS_REQUEST = requests.Session.request
_REAL_CURL_REQUEST = curl_requests.Session.request
_REAL_URLOPEN = urllib.request.urlopen


def test_requests_transport_is_refused(network_guard):
    with pytest.raises(RuntimeError, match="network call in test"):
        requests.get(LOCAL_DEAD_END, timeout=1)
    assert network_guard == [f"GET {LOCAL_DEAD_END}"]
    network_guard.clear()  # the refusal was the point of this test


def test_curl_cffi_transport_is_refused(network_guard):
    """yfinance ≥0.2.5x speaks curl_cffi, not requests; the guard must follow it."""
    with pytest.raises(RuntimeError, match="network call in test"):
        curl_requests.get(LOCAL_DEAD_END, timeout=1)
    assert network_guard == [f"GET {LOCAL_DEAD_END}"]
    network_guard.clear()


def test_urllib_transport_is_refused(network_guard):
    """agents/sentiment_agent.py fetches with urllib.request, not requests."""
    with pytest.raises(RuntimeError, match="network call in test"):
        urllib.request.urlopen(urllib.request.Request(LOCAL_DEAD_END), timeout=1)
    assert network_guard == [f"GET {LOCAL_DEAD_END}"]
    network_guard.clear()


def test_yfinance_uses_the_guarded_transport():
    """Pins the dependency fact the curl_cffi patch relies on.

    Passes on day one by design: it exists to fail the day yfinance changes
    transport again, so the guard is re-pointed instead of silently bypassed.
    """
    from yfinance import data as yf_data
    assert yf_data.requests is curl_requests


def test_attempt_is_recorded_even_when_the_caller_swallows_it(network_guard):
    """data.fetcher wraps every fetch in try/except and falls back.

    A refusal that only raises would let such a test pass — on the fallback,
    after seven seconds of retry sleeps — and the network dependency would
    stay invisible. The record is what surfaces it.
    """
    with contextlib.suppress(RuntimeError):
        requests.post(LOCAL_DEAD_END, timeout=1)
    assert network_guard == [f"POST {LOCAL_DEAD_END}"]
    network_guard.clear()


def test_a_swallowed_attempt_fails_the_test_at_teardown(tmp_path):
    """A throwaway test, in its own pytest, catches the refusal and asserts nothing.

    That is the exact shape the six fetcher tests had: production code caught
    the error, fell back, and the test stayed green. The guard's teardown has
    to turn that run red and name the request — which can only be shown from
    outside the test being judged, hence the subprocess.
    """
    probe = tmp_path / "test_probe.py"
    probe.write_text(textwrap.dedent(f"""
        import contextlib
        import requests
        from tests.conftest import network_guard  # noqa: F401  (autouse rides along)

        def test_swallows_the_refusal():
            with contextlib.suppress(RuntimeError):
                requests.get("{LOCAL_DEAD_END}", timeout=1)
    """))
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(probe)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert run.returncode != 0, f"the probe must not be green:\n{run.stdout}"
    assert "network call in test" in run.stdout, run.stdout
    assert f"GET {LOCAL_DEAD_END}" in run.stdout, run.stdout


@pytest.mark.allow_network
def test_allow_network_marker_leaves_the_real_transports_in_place():
    """The opt-out for a test that must exercise a transport itself.

    Asserted by identity, not by making a request: an opted-out test is the
    one place a real call could happen, and this file is not going to be it.
    """
    assert requests.Session.request is _REAL_REQUESTS_REQUEST
    assert curl_requests.Session.request is _REAL_CURL_REQUEST
    assert urllib.request.urlopen is _REAL_URLOPEN
