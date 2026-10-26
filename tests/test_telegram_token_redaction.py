"""Transport errors must never write the bot token to logs or the fallback file."""
import logging

import requests

from alerts import telegram_alert

TOKEN = "123456789:FAKE-TOKEN-FOR-TESTS-ONLY-abcdefghijk"


def test_transport_error_redacts_token(monkeypatch, caplog, tmp_path):
    def boom(url, **_):
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}")

    written = {}
    monkeypatch.setattr(telegram_alert.requests, "post", boom)
    monkeypatch.setattr(telegram_alert.time, "sleep", lambda *_: None)
    monkeypatch.setattr(telegram_alert, "_write_fallback_log",
                        lambda text, reason: written.setdefault("reason", str(reason)))
    with caplog.at_level(logging.DEBUG):
        assert telegram_alert._post_chunk(TOKEN, "1", "hello") is False
    assert TOKEN not in caplog.text
    assert TOKEN not in written["reason"]
    assert "<TELEGRAM_BOT_TOKEN>" in written["reason"]


def test_console_mode_prints_instead_of_sending(unguarded_telegram, monkeypatch, capsys):
    monkeypatch.setenv("ALERTS_MODE", "console")
    monkeypatch.setattr(telegram_alert.requests, "post",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    assert telegram_alert.send_message("demo alert") is True
    assert "demo alert" in capsys.readouterr().out
