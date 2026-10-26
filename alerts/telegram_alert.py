import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env relative to this file so it works regardless of cwd
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

log = logging.getLogger(__name__)
log.debug("Telegram config — token: %s… chat_id: %s", BOT_TOKEN[:10] if BOT_TOKEN else "(empty)", CHAT_ID or "(empty)")

_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_FALLBACK_LOG = _LOGS_DIR / "telegram_fallback.log"
_MAX_RETRIES = 3
_RETRY_DELAYS = (2, 4)  # seconds between attempt 1→2 and 2→3


def _write_fallback_log(text: str, reason: object) -> None:
    """Persist undelivered message locally so alerts are never silently dropped."""
    try:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with _FALLBACK_LOG.open("a") as fh:
            fh.write(
                f"\n[{datetime.now().isoformat()}] TELEGRAM FAILED ({reason}):\n"
                f"{text}\n{'─' * 60}\n"
            )
    except Exception:
        pass


def _chunks(text: str, limit: int = 4000) -> list[str]:
    """Split `text` into pieces of at most `limit` chars, breaking only on
    '\\n' boundaries so HTML tags (<b>, <i>) never get cut mid-line.

    Telegram's real cap is 4096; 4000 leaves headroom. A single line longer
    than `limit` (no newlines at all) is hard-split rather than left intact —
    a message must still ship even when it has nowhere natural to break.

    Pure and network-free on purpose: this is the piece
    "message is too long" needed tested without mocking requests.post.
    """
    lines = text.split("\n")
    chunks: list[str] = []
    current = ""
    for line in lines:
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    chunks.append(current)
    return chunks


def _post_chunk(token: str, chat_id: str, text: str) -> bool:
    """One chunk's full attempt/retry/fallback-log cycle — the body
    send_message used to run once per whole message, now run per chunk."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    last_err: object = "unknown"
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url,
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.status_code == 200:
                return True
            last_err = f"HTTP {resp.status_code}"
            log.error("Telegram API error %s (attempt %d/%d): %s",
                      resp.status_code, attempt, _MAX_RETRIES, resp.text)
        except Exception as exc:
            # requests puts the full URL — token included — in the exception text.
            last_err = str(exc).replace(token, "<TELEGRAM_BOT_TOKEN>")
            log.warning("Telegram attempt %d/%d failed: %s", attempt, _MAX_RETRIES, last_err)
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAYS[attempt - 1])
    log.error("Telegram: all %d attempts failed — writing to fallback log", _MAX_RETRIES)
    _write_fallback_log(text, last_err)
    return False


def _console_mode() -> bool:
    """ALERTS_MODE=console: print alerts instead of sending them (offline demo)."""
    return os.getenv("ALERTS_MODE", "").strip().lower() == "console"


def send_message(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
    chat_id = os.getenv("TELEGRAM_CHAT_ID", CHAT_ID)
    if _console_mode():
        # Offline demo: print instead of calling the Telegram API.
        print("\n[ALERT]\n" + text)
        return True
    if not token or not chat_id:
        log.warning("Telegram send_message skipped: BOT_TOKEN=%s CHAT_ID=%s", bool(token), bool(chat_id))
        # Missing credentials must not swallow the alert either — an unconfigured
        # bot was dropping every message without even reaching the fallback log.
        _write_fallback_log(text, "missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID")
        return False
    # Telegram rejects anything over 4096 chars outright — "message is too
    # long" has been silently swallowing every weekly report since 2026-09-09.
    # Every chunk is sent regardless of an earlier one failing (a list, not a
    # short-circuiting generator) — one failed chunk must not stop the rest
    # of the message from going out.
    results = [_post_chunk(token, chat_id, chunk) for chunk in _chunks(text)]
    return all(results)


# Telegram rejects a message over 4096 characters outright, so the funnel report
# budgets for it rather than discovering it at send time.
_TELEGRAM_MAX_CHARS = 4096
_MAX_TICKERS_PER_REASON = 12


def _ticker_list(tickers: list, limit: int = _MAX_TICKERS_PER_REASON, more: str = "más") -> str:
    shown = ", ".join(tickers[:limit])
    extra = len(tickers) - limit
    return f"{shown} +{extra} {more}" if extra > 0 else shown


def _group_tickers(pairs: list, more: str = "more") -> list:
    """[(label, [tickers])] → ["label (n): A, B, C +k more", ...], empty groups dropped."""
    return [f"{label} ({len(tickers)}): {_ticker_list(tickers, more=more)}"
            for label, tickers in pairs if tickers]


def send_daily_summary(research: list, sentiment: list, pm_recommendations: dict,
                       held_tickers=()) -> bool:
    """One message: what the PM decided, compressed to what the operator can act on.

    `held_tickers` are the names currently in the book: a PM SELL on one of
    them is a veto that run_exits will act on, so it keeps its rationale;
    a SELL on a name we do not own is just a list entry.

    No per-ticker size here on purpose: the PM is veto-only and every rec
    carries size_pct=0.0 (agents/pm_agent.py) — printing it read as
    "0.0%" on 48 lines. The engine's real allocation is in the Execution
    Report, next to the order that got it.
    """
    if (not BOT_TOKEN or not CHAT_ID) and not _console_mode():
        return False

    held = set(held_tickers or ())
    sent_map = {s["ticker"]: s for s in sentiment}
    recs = pm_recommendations.get("recommendations", [])

    lines = ["<b>🤖 AI Hedge Fund — Daily Summary</b>\n"]

    buys = [r for r in recs if r["action"] == "BUY"]
    sells = [r for r in recs if r["action"] == "SELL"]
    holds = [r for r in recs if r["action"] == "HOLD"]
    lines.append(f"📊 <b>PM Signals:</b> {len(buys)} BUY | {len(sells)} SELL | {len(holds)} HOLD")

    high_conviction_buys = [r for r in buys if r.get("conviction", 0) >= 4]
    if high_conviction_buys:
        lines.append(f"\n🚀 <b>High Conviction BUYs ({len(high_conviction_buys)}):</b>")
        tiers = sorted({r["conviction"] for r in high_conviction_buys}, reverse=True)
        lines.extend(
            f"  {line}" for line in _group_tickers(
                [(f"conv {c}", [r["ticker"] for r in high_conviction_buys if r["conviction"] == c])
                 for c in tiers])
        )

    if sells:
        lines.append(f"\n🔴 <b>SELL Signals ({len(sells)}):</b>")
        for rec in sells:
            if rec["ticker"] in held:
                lines.append(f"  ❌ <b>{rec['ticker']}</b> (held) — {rec['rationale']}")
        not_held = [r["ticker"] for r in sells if r["ticker"] not in held]
        lines.extend(f"  {line}" for line in _group_tickers([("Not held", not_held)]))

    buy_vs_negative, sell_vs_positive = [], []
    for r in research:
        s = sent_map.get(r["ticker"], {})
        if r["recommendation"] == "BUY" and s.get("sentiment") == "NEGATIVE":
            buy_vs_negative.append(r["ticker"])
        elif r["recommendation"] == "SELL" and s.get("sentiment") == "POSITIVE":
            sell_vs_positive.append(r["ticker"])
    divergences = _group_tickers([("Research BUY / Sentiment NEGATIVE", buy_vs_negative),
                                  ("Research SELL / Sentiment POSITIVE", sell_vs_positive)])
    if divergences:
        lines.append("\n⚠️ <b>Signal Divergences:</b>")
        lines.extend(f"  ⚡ {d}" for d in divergences)

    comment = pm_recommendations.get("portfolio_comment", "")
    if comment:
        lines.append(f"\n💼 <b>PM Comment:</b> {comment}")

    return send_message("\n".join(lines))


# The ledger reads an exit's reason from the sell order's client_order_id tag
# (execution/ledger.EXIT_REASON_BY_TAG). A sell that carried no tag — a legacy
# OCO leg, a manual close, an untagged replacement — has no reason on record
# anywhere; the fill, price and P&L are real. "UNKNOWN" on its own read as
# "the system does not know what it did".
_EXIT_REASON_LABELS = {"UNKNOWN": "reason not recorded (broker-side close)"}


def _exit_day(row: dict) -> "date | None":
    """Calendar day of `exit_time` (ISO 8601), None when absent or unparsable.

    Engine rows (run_exit_engine) carry no exit_time — they closed this run.
    Ledger rows carry the broker fill's timestamp, which can be months old
    when the first reconciliation catches up on positions the broker closed
    long ago.
    """
    raw = row.get("exit_time")
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _closed_line(pos: dict, *, show_date: bool) -> str:
    # Engine rows (run_exit_engine) and ledger broker_exits rows both
    # carry `pnl_pct` — a fraction, None when the ledger could not
    # match a trade. `unrealized_plpc` is the older key.
    plpc = pos.get("pnl_pct", pos.get("unrealized_plpc"))
    if plpc is None:
        pl_emoji, pl_text = "⚪", "n/a"
    else:
        pl_emoji, pl_text = ("🟢" if plpc >= 0 else "🔴"), f"{plpc:+.2%}"
    reason = _EXIT_REASON_LABELS.get(pos["reason"], pos["reason"])
    when = f" | closed {_exit_day(pos).isoformat()}" if show_date else ""
    return f"  {pl_emoji} <b>{pos['ticker']}</b> — {reason}{when} | P&L: {pl_text}"


def _order_line(order: dict) -> str:
    price = order.get("price", "market")
    text = f"  📥 <b>{order['ticker']}</b> — BUY {order['qty']} @ ~${price}"
    if isinstance(price, (int, float)):
        text += f" ≈ ${float(order['qty']) * float(price):,.0f}"
    # The engine's real allocation (run_entries sets it) — not the PM's
    # always-zero size_pct.
    if order.get("size_pct") is not None:
        text += f" ({float(order['size_pct']):.1f}% eq)"
    return text


def send_execution_alerts(orders_submitted: list, positions_closed: list) -> bool:
    if (not BOT_TOKEN or not CHAT_ID) and not _console_mode():
        return False

    lines = ["<b>⚡ Execution Report</b>\n"]

    if orders_submitted:
        lines.append("✅ <b>New Positions Opened:</b>")
        lines.extend(_order_line(order) for order in orders_submitted)
    else:
        lines.append("✅ No new positions opened.")

    # A ledger row whose fill is from an earlier day is not a sell this run
    # made: it is the ledger catching up with the broker. Eight of those
    # (closed between June and September) showed up as "Positions Closed" on
    # the first Sprint-1 reconciliation and read as eight exits that morning.
    today = date.today()
    closed_today = [p for p in positions_closed if (_exit_day(p) or today) >= today]
    reconciled = [p for p in positions_closed if (_exit_day(p) or today) < today]

    if closed_today:
        lines.append("\n🔒 <b>Positions Closed:</b>")
        lines.extend(_closed_line(pos, show_date=False) for pos in closed_today)
    else:
        lines.append("\n🔒 No positions closed.")

    if reconciled:
        lines.append("\n🧾 <b>Reconciled from broker history</b> (closed earlier, booked today):")
        lines.extend(_closed_line(pos, show_date=True) for pos in reconciled)

    return send_message("\n".join(lines))


def send_funnel_report(summary: dict, regime: str = "", budget: dict = None) -> bool:
    """Report where every analysed asset died, and what the main blocker was.

    This is the message that answers "why has nothing traded in a week". Before
    it existed the operator saw only "No new positions opened" and had no way to
    tell a quiet market from a gate quietly eating every candidate.

    `summary` is utils.funnel.FunnelRecorder.summary().
    """
    if not summary:
        return False
    if (not BOT_TOKEN or not CHAT_ID) and not _console_mode():
        log.warning("Funnel report not sent (Telegram unconfigured)")
        _write_fallback_log(str(summary), "missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID")
        return False

    stages = summary.get("stages", {})
    lines = [
        "<b>🔎 Embudo de decisión</b>",
        f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        + (f" · régimen {regime}" if regime else "") + "</i>",
        "",
    ]

    cascade = " → ".join(
        f"{name} {stages[key]}"
        for key, name in (
            ("universe",  "Universo"),
            ("pm_buy",    "PM BUY"),
            ("gates",     "Pasan compuertas"),
            ("funded",    "Con presupuesto"),
            ("executed",  "Ejecutadas"),
        )
        if key in stages
    )
    if cascade:
        lines.append(f"📊 {cascade}")

    if budget:
        lines.append(
            f"💵 Presupuesto ${budget.get('deployable', 0):,.0f} "
            f"(base {budget.get('base', '?')}) · "
            f"desplegado ${budget.get('deployed', 0):,.0f} · "
            f"sin usar ${budget.get('leftover', 0):,.0f}"
        )

    executed = summary.get("executed") or []
    if executed:
        lines.append(f"\n✅ <b>Compradas ({len(executed)}):</b> {_ticker_list(executed)}")
    queued = summary.get("queued") or []
    if queued:
        lines.append(f"⏳ <b>En cola ({len(queued)}):</b> {_ticker_list(queued)}")

    reasons = summary.get("reasons") or []
    if reasons:
        lines.append(f"\n❌ <b>Descartadas ({summary.get('rejected_total', 0)}):</b>")
        for r in reasons:
            lines.append(f"  • {r['label']} ({r['count']}): {_ticker_list(r['tickers'])}")

    bottleneck = summary.get("bottleneck")
    if bottleneck:
        lines.append(
            f"\n🚧 <b>Cuello de botella:</b> {bottleneck['label']} "
            f"— {bottleneck['count']} activo(s)"
        )

    text = "\n".join(lines)
    if len(text) > _TELEGRAM_MAX_CHARS:
        # Trim whole lines from the tail of the reason list, never mid-tag:
        # HTML parse_mode rejects an unbalanced <b>.
        while len(text) > _TELEGRAM_MAX_CHARS - 40 and len(lines) > 4:
            lines.pop(-2)
            text = "\n".join(lines)
        text += "\n<i>(recortado)</i>"
    return send_message(text)


def send_health_alert(issues: list[str]) -> bool:
    """Report a run that completed but did not do its job.

    Every failure this covers is silent by construction: the pipeline prints
    "Run complete", writes its usual artifacts and exits 0 while the PM produced
    nothing, the broker refused every write, or the entry stage found no
    candidates it should have found. Nothing else surfaces it.

    Each issue is a pre-formatted line naming the stage and what broke.
    """
    if not issues:
        return False
    if (not BOT_TOKEN or not CHAT_ID) and not _console_mode():
        log.warning("Health alert not sent (Telegram unconfigured): %s", "; ".join(issues))
        return False

    lines = [
        "<b>🚨 AI Hedge Fund — Degraded Run</b>",
        f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>",
        "\nThe run completed, but the pipeline did not work as intended:",
    ]
    lines.extend(f"  • {issue}" for issue in issues)
    return send_message("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    ok = send_message("Test from hedge fund system")
    print("send_message result:", ok)
