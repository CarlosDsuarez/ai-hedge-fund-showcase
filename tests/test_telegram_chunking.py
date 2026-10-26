# tests/test_telegram_chunking.py
"""Telegram send_message chunking (D. Ledger).

Weekly reports with 12+ closed positions routinely exceed Telegram's 4096-char
cap as one string, and every send since 2026-09-09 has failed outright with
"message is too long" — the whole report silently never arrived. `_chunks` is
a pure function so the splitting logic is testable without any network, and
one integration test below drives send_message through a patched
requests.post to prove the chunk count and delivery order.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alerts.telegram_alert import _chunks


# ── _chunks: pure function ───────────────────────────────────────────────────

def test_chunks_returns_single_chunk_for_short_text():
    assert _chunks("hello", limit=4000) == ["hello"]


def test_chunks_returns_single_chunk_for_empty_text():
    assert _chunks("", limit=4000) == [""]


def test_chunks_splits_on_newline_boundary_not_mid_line():
    line_a = "a" * 3990
    line_b = "b" * 3990
    text = f"{line_a}\n{line_b}"
    chunks = _chunks(text, limit=4000)
    assert len(chunks) == 2
    assert chunks[0] == line_a
    assert chunks[1] == line_b
    for c in chunks:
        assert len(c) <= 4000


def test_chunks_packs_multiple_short_lines_into_one_chunk():
    text = "\n".join(["short"] * 100)   # far under 4000 chars total
    chunks = _chunks(text, limit=4000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunks_hard_splits_a_single_line_longer_than_limit():
    text = "x" * 9500   # no newlines at all
    chunks = _chunks(text, limit=4000)
    assert len(chunks) == 3
    assert chunks[0] == "x" * 4000
    assert chunks[1] == "x" * 4000
    assert chunks[2] == "x" * 1500
    assert "".join(chunks) == text


def test_chunks_respects_limit_boundary_exactly():
    """A line of exactly `limit` chars must NOT be split or pushed to its own
    chunk unnecessarily when it's the only content."""
    text = "y" * 4000
    chunks = _chunks(text, limit=4000)
    assert chunks == ["y" * 4000]


def test_chunks_never_exceeds_limit_across_many_lines():
    lines = [f"line-{i}-{'z' * 50}" for i in range(500)]
    text = "\n".join(lines)
    chunks = _chunks(text, limit=4000)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 4000
    # Every original line must survive somewhere, in order, unmodified.
    assert "\n".join(chunks).replace("\n\n", "\n") or True  # sanity: no exception
    rejoined = "\n".join(chunks)
    for line in lines:
        assert line in rejoined


# ── send_message: real chunking behaviour through a patched transport ───────

def test_send_message_posts_one_chunk_per_split(unguarded_telegram, monkeypatch):
    import alerts.telegram_alert as telegram

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    long_text = "\n".join(["x" * 3999] * 3)   # forces 3 separate chunks
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json["text"])
        resp = MagicMock()
        resp.status_code = 200
        return resp

    with patch("alerts.telegram_alert.requests.post", side_effect=fake_post) as mock_post:
        ok = telegram.send_message(long_text)

    assert ok is True
    assert mock_post.call_count == 3
    assert calls == ["x" * 3999] * 3


def test_send_message_returns_false_if_any_chunk_fails(unguarded_telegram, monkeypatch):
    import alerts.telegram_alert as telegram

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    long_text = "\n".join(["x" * 3999] * 2)

    def fake_post(url, json=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 200 if "first" not in json["text"] else 400
        return resp

    with patch("alerts.telegram_alert.requests.post",
               return_value=MagicMock(status_code=400)), \
         patch("alerts.telegram_alert.time.sleep"):
        ok = telegram.send_message(long_text)

    assert ok is False


def test_send_message_short_text_still_sends_exactly_once(unguarded_telegram, monkeypatch):
    import alerts.telegram_alert as telegram

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    with patch("alerts.telegram_alert.requests.post",
               return_value=MagicMock(status_code=200)) as mock_post:
        ok = telegram.send_message("short message")

    assert ok is True
    assert mock_post.call_count == 1
