"""send_daily_summary must fit in one Telegram message and carry no dead numbers.

On 2026-09-21 the PM approved 48 BUYs at conviction >= 4. The summary printed
one line per name with `size_pct` — a field agents/pm_agent.py forces to 0.0
on every rec (the PM is veto-only) — plus a ~250-char rationale: ~13k chars,
four Telegram chunks, every line reading "0.0%". The operator's reading was
"almost everything says 0.0%", which is exactly what a dead field looks like.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import alerts.telegram_alert as telegram

_RATIONALE = "Compelling combination of revenue growth, ROE and a reasonable P/E."


def _rec(ticker, action="BUY", conviction=4, rationale=_RATIONALE):
    return {"ticker": ticker, "action": action, "size_pct": 0.0,
            "conviction": conviction, "rationale": rationale}


def _render(research=(), sentiment=(), recs=(), comment="", held=()):
    sent: list = []
    pm = {"recommendations": list(recs), "portfolio_comment": comment}
    with patch.object(telegram, "BOT_TOKEN", "t"), patch.object(telegram, "CHAT_ID", "c"), \
         patch.object(telegram, "send_message", side_effect=lambda text, **kw: sent.append(text) or True):
        assert telegram.send_daily_summary(list(research), list(sentiment), pm,
                                           held_tickers=set(held)) is True
    return sent[0]


def test_high_conviction_buys_are_grouped_by_conviction_without_size_or_rationale():
    conv5 = [_rec(f"A{i}", conviction=5) for i in range(6)]
    conv4 = [_rec(f"B{i:02d}", conviction=4) for i in range(42)]
    text = _render(recs=conv5 + conv4)

    assert "0.0%" not in text
    assert _RATIONALE not in text
    assert "High Conviction BUYs (48)" in text
    assert text.index("conv 5 (6)") < text.index("conv 4 (42)")
    assert "A0, A1, A2, A3, A4, A5" in text
    assert "+30 more" in text
    assert "más" not in text
    assert len(text.splitlines()) <= 12
    assert len(text) < 1500


def test_conviction_below_four_is_not_listed():
    text = _render(recs=[_rec("LOW", conviction=3)])
    assert "High Conviction" not in text
    assert "LOW" not in text


def test_sell_on_held_name_keeps_rationale_others_collapse_to_one_line():
    recs = [_rec("INTC", "SELL", 4, "Guidance cut; margins compressing."),
            _rec("APD", "SELL", 4, "APD rationale that must not appear."),
            _rec("D", "SELL", 4, "D rationale that must not appear.")]
    text = _render(recs=recs, held={"INTC"})

    assert "SELL Signals (3)" in text
    assert "<b>INTC</b> (held) — Guidance cut; margins compressing." in text
    assert "APD rationale" not in text and "D rationale" not in text
    assert "Not held (2): APD, D" in text


def test_divergences_are_grouped_per_direction():
    research = [{"ticker": "X", "recommendation": "BUY"},
                {"ticker": "Y", "recommendation": "BUY"},
                {"ticker": "Z", "recommendation": "SELL"}]
    sentiment = [{"ticker": "X", "sentiment": "NEGATIVE"},
                 {"ticker": "Y", "sentiment": "NEGATIVE"},
                 {"ticker": "Z", "sentiment": "POSITIVE"}]
    text = _render(research=research, sentiment=sentiment)

    assert "Research BUY / Sentiment NEGATIVE (2): X, Y" in text
    assert "Research SELL / Sentiment POSITIVE (1): Z" in text


def test_pm_signal_counts_and_comment_are_kept():
    recs = [_rec("A", "BUY", 5), _rec("B", "SELL", 4), _rec("C", "HOLD", 2)]
    text = _render(recs=recs, comment="Risk-on; staying long quality.")

    assert "1 BUY | 1 SELL | 1 HOLD" in text
    assert "PM Comment:</b> Risk-on; staying long quality." in text


def test_held_tickers_defaults_to_empty():
    """main.py passes held_tickers; other callers (tests, chat) may not."""
    sent: list = []
    pm = {"recommendations": [_rec("APD", "SELL")], "portfolio_comment": ""}
    with patch.object(telegram, "BOT_TOKEN", "t"), patch.object(telegram, "CHAT_ID", "c"), \
         patch.object(telegram, "send_message", side_effect=lambda text, **kw: sent.append(text) or True):
        assert telegram.send_daily_summary([], [], pm) is True
    assert "Not held (1): APD" in sent[0]
