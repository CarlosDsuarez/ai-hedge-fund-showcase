"""send_execution_alerts renders the realized return of each closed position.

The close line read `unrealized_plpc`, a key no closed row ever carried: the
exit engine's rows carry `pnl_pct` (execution/order_manager.run_exit_engine)
and the ledger's broker_exits rows carry `pnl_pct` or None — so every closed
position rendered as "+0.0%". A missing figure must say "n/a", not fake a
flat trade.
"""
import os
import sys
from datetime import date, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import alerts.telegram_alert as telegram


def _render(closed: list) -> str:
    sent: list = []
    with patch.object(telegram, "BOT_TOKEN", "t"), patch.object(telegram, "CHAT_ID", "c"), \
         patch.object(telegram, "send_message", side_effect=lambda text, **kw: sent.append(text) or True):
        assert telegram.send_execution_alerts([], closed) is True
    return sent[0]


def test_engine_row_pnl_pct_is_rendered():
    text = _render([{"ticker": "JPM", "reason": "STOP_TRAIL", "qty": 12,
                     "price": 356.3, "pnl_pct": 0.08}])
    assert "JPM" in text and "STOP_TRAIL" in text
    assert "+8.00%" in text
    assert "🟢" in text


def test_negative_pnl_pct_is_red():
    text = _render([{"ticker": "AAPL", "reason": "BROKER_STOP", "qty": 10,
                     "price": 90.0, "pnl_pct": -0.10}])
    assert "-10.00%" in text
    assert "🔴" in text


def test_missing_pnl_pct_renders_na_not_flat():
    """A ledger UNKNOWN exit (vanished position, no sell fill in the window)
    carries pnl_pct=None; a row with no key at all must behave the same."""
    text = _render([
        {"ticker": "GOOGL", "reason": "UNKNOWN", "qty": 4, "price": 0.0, "pnl_pct": None},
        {"ticker": "HON", "reason": "LEGACY_OCO", "qty": 10, "price": 200.0},
    ])
    assert text.count("P&L: n/a") == 2
    assert "+0.00%" not in text


def test_legacy_unrealized_plpc_key_still_honoured():
    text = _render([{"ticker": "NVDA", "reason": "PM_SELL", "unrealized_plpc": 0.05}])
    assert "+5.00%" in text


# ── precision ────────────────────────────────────────────────────────────────

def test_small_pnl_keeps_two_decimals_instead_of_rounding_to_flat():
    """PGR closed at +0.19% on 2026-09-18; one decimal printed it as +0.2%,
    and LLY's -0.07% as -0.1%. Two decimals say what happened."""
    text = _render([{"ticker": "PGR", "reason": "BROKER_STOP", "pnl_pct": 0.0019},
                    {"ticker": "LLY", "reason": "BROKER_STOP", "pnl_pct": -0.0007}])
    assert "+0.19%" in text
    assert "-0.07%" in text


# ── UNKNOWN is a missing record, and the report has to say so ────────────────

def test_unknown_reason_is_explained_not_printed_bare():
    """The ledger reads the exit reason from the sell order's client_order_id
    tag. Legacy OCO legs (and any untagged sell) carry an Alpaca UUID, so the
    reason was never recorded — the P&L is real, only the "why" is missing."""
    text = _render([{"ticker": "AMZN", "reason": "UNKNOWN", "pnl_pct": -0.0796}])
    assert "reason not recorded (broker-side close)" in text
    assert "— UNKNOWN |" not in text
    assert "-7.96%" in text


def test_engine_reason_codes_are_left_as_is():
    text = _render([{"ticker": "JPM", "reason": "STOP_TRAIL", "pnl_pct": 0.08}])
    assert "— STOP_TRAIL |" in text


# ── historical exits surfaced by the ledger are not today's closes ───────────

def test_exit_closed_on_an_earlier_day_is_reported_as_reconciled_with_its_date():
    """First Sprint-1 reconciliation (2026-09-21) booked eight positions the
    broker had closed months earlier (AMZN 06-03, META 06-05, UNH 07-16...).
    Listing them under "Positions Closed" read as eight sells today."""
    text = _render([{"ticker": "AMZN", "reason": "UNKNOWN", "pnl_pct": -0.0796,
                     "exit_time": "2026-06-03T15:06:02.365213Z"}])
    assert "Reconciled from broker history" in text
    assert "| closed 2026-06-03 |" in text
    assert "No positions closed." in text
    assert text.index("No positions closed.") < text.index("AMZN")


def test_engine_row_without_exit_time_counts_as_today():
    text = _render([{"ticker": "JPM", "reason": "STOP_TRAIL", "pnl_pct": 0.08}])
    assert "Positions Closed:" in text
    assert "Reconciled from broker history" not in text
    assert "| closed " not in text


def test_exit_time_today_counts_as_today():
    today = date.today().isoformat()
    text = _render([{"ticker": "PGR", "reason": "BROKER_STOP", "pnl_pct": 0.0019,
                     "exit_time": f"{today}T13:34:03.452403Z"}])
    assert "Positions Closed:" in text
    assert "Reconciled from broker history" not in text


def test_today_and_historical_rows_are_split_into_their_own_sections():
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    text = _render([
        {"ticker": "PGR", "reason": "BROKER_STOP", "pnl_pct": 0.0019},
        {"ticker": "AMZN", "reason": "UNKNOWN", "pnl_pct": -0.0796,
         "exit_time": f"{yesterday}T15:06:02Z"},
    ])
    closed_at = text.index("Positions Closed:")
    recon_at = text.index("Reconciled from broker history")
    assert closed_at < text.index("PGR") < recon_at < text.index("AMZN")


def test_unparsable_exit_time_is_treated_as_today():
    text = _render([{"ticker": "X", "reason": "PM_SELL", "pnl_pct": 0.01, "exit_time": "n/a"}])
    assert "Positions Closed:" in text
    assert "Reconciled from broker history" not in text


# ── orders: notional and the engine's real size ──────────────────────────────

def _render_orders(orders: list) -> str:
    sent: list = []
    with patch.object(telegram, "BOT_TOKEN", "t"), patch.object(telegram, "CHAT_ID", "c"), \
         patch.object(telegram, "send_message", side_effect=lambda text, **kw: sent.append(text) or True):
        assert telegram.send_execution_alerts(orders, []) is True
    return sent[0]


def test_order_line_shows_notional_and_engine_size():
    """size_pct here is the engine's real allocation (run_entries sets it),
    unlike the PM's always-zero size_pct in the daily summary."""
    text = _render_orders([{"ticker": "PSX", "qty": 47, "price": 271.51,
                            "size_pct": 12.532679613654748}])
    assert "<b>PSX</b> — BUY 47 @ ~$271.51 ≈ $12,761 (12.5% eq)" in text


def test_order_line_without_size_or_numeric_price_degrades_gracefully():
    text = _render_orders([{"ticker": "NUE", "qty": 23, "price": 247.69},
                           {"ticker": "MS", "qty": 5}])
    assert "<b>NUE</b> — BUY 23 @ ~$247.69 ≈ $5,697" in text
    assert "eq)" not in text
    assert "<b>MS</b> — BUY 5 @ ~$market" in text
    assert "≈ $" not in text.split("<b>MS</b>")[1]
