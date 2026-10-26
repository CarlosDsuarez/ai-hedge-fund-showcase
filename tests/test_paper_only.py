"""The broker client has no live-trading path: without PAPER_TRADING=true it refuses."""
import pytest

from execution import alpaca


@pytest.mark.parametrize("value", [None, "false", "", "live"])
def test_base_url_fails_closed_without_paper_flag(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PAPER_TRADING", raising=False)
    else:
        monkeypatch.setenv("PAPER_TRADING", value)
    with pytest.raises(alpaca.LiveTradingDisabled):
        alpaca._get_base_url()


def test_base_url_is_paper_host(monkeypatch):
    monkeypatch.setenv("PAPER_TRADING", "true")
    assert alpaca._get_base_url() == "https://paper-api.alpaca.markets"
