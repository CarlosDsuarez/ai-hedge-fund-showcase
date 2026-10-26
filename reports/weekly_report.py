import json
import os
import re
from datetime import datetime, timedelta
from glob import glob

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alerts.telegram_alert import send_message
from config import TICKERS as _TICKERS


def _scan_week_runs(logs_dir: str) -> list[dict]:
    today = datetime.now()
    monday = (today - timedelta(days=today.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    result = []
    for path in sorted(glob(os.path.join(logs_dir, "runs", "*.json"))):
        fname = os.path.basename(path)
        try:
            run_date = datetime.strptime(fname[:10], "%Y-%m-%d")
        except ValueError:
            continue
        if monday <= run_date <= today:
            try:
                with open(path) as f:
                    result.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
    return result


def _aggregate_week_trades(week_runs: list[dict]) -> list[dict]:
    """Positions opened this week, deduped by broker order id.

    Closed-this-week used to be scanned out of the same run JSONs (fragile:
    it re-derived P&L from a weighted-average entry price computed here, and
    a run this scan never saw was a close never counted). That half is gone —
    _closed_this_week() below reads db.load_trades() instead, the ledger's own
    FIFO-reconstructed record of what actually closed and for how much.
    """
    seen_order_ids: set[str] = set()
    opened: list[dict] = []
    for run in week_runs:
        for order in run.get("orders_submitted", []):
            order_id = order.get("order", {}).get("id")
            if order_id and order_id in seen_order_ids:
                continue
            if order_id:
                seen_order_ids.add(order_id)
            opened.append(order)
    return opened


def _closed_this_week(monday_iso: str) -> list[dict]:
    """Positions closed since Monday, from the ledger's FIFO trades table —
    not a run-JSON scan. A trade with no matched buy lot (pre-ledger, entry
    unknown) carries pnl_usd=None and is still shown, just excluded from the
    dollar total by the caller's `or 0.0`.
    """
    try:
        from db.database import load_trades
        trades = load_trades(since=monday_iso)
    except Exception as e:
        print(f"  [Report] load_trades failed: {e}")
        return []
    closed = []
    for t in trades:
        pnl_pct = t.get("pnl_pct")
        closed.append({
            "ticker": t.get("symbol"),
            "unrealized_plpc": (pnl_pct * 100) if pnl_pct is not None else 0.0,
            "dollar_pnl": t.get("pnl_usd"),
            "reason": t.get("exit_reason") or "N/A",
        })
    return closed


def _realized_ledger_stats(limit: int = 30) -> tuple[int, float, float]:
    """(n, win_rate, payoff) over the trailing `limit` scored trades.

    payoff = avg win $ / |avg loss $|. Scored = pnl_usd IS NOT NULL — a trade
    whose entry predates LEDGER_BACKFILL_FROM has no basis to score and is
    excluded rather than silently counted as a wash.
    """
    try:
        from db.database import load_trades
        trades = load_trades()
    except Exception as e:
        print(f"  [Report] load_trades failed: {e}")
        return 0, 0.0, 0.0
    scored = [t for t in trades if t.get("pnl_usd") is not None][-limit:]
    if not scored:
        return 0, 0.0, 0.0
    wins = [t["pnl_usd"] for t in scored if t["pnl_usd"] > 0]
    losses = [t["pnl_usd"] for t in scored if t["pnl_usd"] < 0]
    win_rate = len(wins) / len(scored)
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    payoff = (avg_win / abs(avg_loss)) if avg_loss else 0.0
    return len(scored), win_rate, payoff


def _get_open_positions(position_state: dict) -> list[dict]:
    """Open positions from the broker's own book (execution.alpaca.get_positions),
    not the last run's SL/TP-audit snapshot — that snapshot is only as fresh
    as the last successful run_exits() call and silently drops any position
    the audit errored on. position_state (logs/position_state.json) supplies
    the stop levels Alpaca's position object doesn't carry.
    """
    try:
        from execution.alpaca import get_positions
        raw = get_positions()
    except Exception as e:
        print(f"  [Report] get_positions failed: {e}")
        return []

    positions = []
    for p in raw:
        ticker = (p.get("symbol") or "").replace(".", "-")
        if not ticker:
            continue
        state = position_state.get(ticker, {})
        positions.append({
            "ticker": ticker,
            "entry_price": float(p.get("avg_entry_price") or 0),
            "current_price": float(p.get("current_price") or 0),
            "unrealized_pnl_pct": float(p.get("unrealized_plpc") or 0) * 100,
            "dollar_pnl": round(float(p.get("unrealized_pl") or 0), 2),
            "qty": float(p.get("qty") or 0),
            "sl_price": state.get("sl_price"),
            "trail_stop": state.get("trail_stop"),
        })
    return positions


def _get_account_snapshot() -> tuple[float, float, float]:
    """(equity, cash, week_pnl_usd) straight from the broker.

    Replaces _get_portfolio_equity, which read data/portfolio_state.json — a
    file nothing has written since May (ground truth check: weekly_report.py
    used to derive both numbers from that stale snapshot). week_pnl_usd is the
    equity curve's own swing since the most recent Monday on/before today, via
    get_portfolio_history — not a re-derivation from closed-trade dollars, so
    it also reflects the week's unrealized moves in still-open positions.

    Every broker call is try/excepted to its own fallback: one Alpaca outage
    degrades a number to 0.0 rather than crashing the whole report.
    """
    equity = cash = 0.0
    try:
        from execution.alpaca import get_account
        account = get_account()
        equity = float(account.get("equity", 0) or 0)
        cash = float(account.get("cash", 0) or 0)
    except Exception as e:
        print(f"  [Report] get_account failed: {e}")

    week_pnl_usd = 0.0
    try:
        from execution.alpaca import get_portfolio_history
        history = get_portfolio_history(period="1M", timeframe="1D")
        timestamps = history.get("timestamp") or []
        equities = history.get("equity") or []
        if timestamps and equities:
            today = datetime.now().date()
            monday = today - timedelta(days=today.weekday())
            monday_equity = equities[0]
            # strict=True: Alpaca's timestamp[]/equity[] are parallel arrays by
            # contract — a length mismatch means a malformed response, and the
            # surrounding try/except already falls back to week_pnl_usd=0.0.
            for ts, eq in zip(timestamps, equities, strict=True):
                if datetime.fromtimestamp(ts).date() <= monday:
                    monday_equity = eq
                else:
                    break
            week_pnl_usd = float(equities[-1]) - float(monday_equity)
    except Exception as e:
        print(f"  [Report] get_portfolio_history failed: {e}")

    return equity, cash, week_pnl_usd


def _format_msg1(
    week_label: str,
    equity: float,
    cash: float,
    unrealized_total: float,
    realized_total: float,
    week_pnl_usd: float,
    opened: list[dict],
    closed: list[dict],
) -> str:
    today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"<b>📊 AI Hedge Fund — Weekly Report ({today_str})</b>",
        f"<i>{week_label}</i>\n",
        "<b>━━━━━━ PORTFOLIO ━━━━━━</b>",
        f"💰 Equity: ${equity:,.0f}  |  💵 Cash: ${cash:,.0f}",
        f"{'📈' if unrealized_total >= 0 else '📉'} Unrealized P&L (open): ${unrealized_total:+,.0f}",
        f"{'💚' if realized_total >= 0 else '🔴'} Realized P&L this week: ${realized_total:+,.0f}",
        f"📅 Equity change this week: ${week_pnl_usd:+,.0f} (broker equity curve)",
    ]
    n_scored, win_rate, payoff = _realized_ledger_stats()
    if n_scored:
        lines.append(
            f"📒 Realized (ledger, last {n_scored} trades): "
            f"win rate {win_rate:.0%} | payoff {payoff:.2f}"
        )
    lines.append("\n<b>━━━━━━ OPENED THIS WEEK ━━━━━━</b>")
    if opened:
        for o in opened:
            ts = o.get("order", {}).get("created_at", "")[:10]
            try:
                day = datetime.strptime(ts, "%Y-%m-%d").strftime("%a %-m/%-d")
            except ValueError:
                day = ts
            lines.append(
                f"📥 <b>{o['ticker']}</b> — {o['qty']} shares @ ${o['price']:,.2f}"
                f" | Conv {o.get('conviction', '?')}/5 | {day}"
            )
    else:
        lines.append("No positions opened this week.")

    lines.append("\n<b>━━━━━━ CLOSED THIS WEEK ━━━━━━</b>")
    if closed:
        for c in closed:
            plpc = c.get("unrealized_plpc", 0.0)
            emoji = "🟢" if plpc >= 0 else "🔴"
            dollar_str = f" (${c['dollar_pnl']:+,.0f})" if c.get("dollar_pnl") is not None else ""
            lines.append(
                f"{emoji} <b>{c['ticker']}</b> — {plpc:+.1f}%{dollar_str}"
                f" | {c.get('reason', 'N/A')}"
            )
    else:
        lines.append("No positions closed this week.")

    return "\n".join(lines)


def _format_msg2(open_positions: list[dict], week_runs: list[dict]) -> str:
    lines = ["<b>━━━━━━ CURRENT POSITIONS ━━━━━━</b>"]
    if open_positions:
        for p in open_positions:
            emoji = "🟢" if p["unrealized_pnl_pct"] >= 0 else "🔴"
            # trail_stop wins: it's the live chandelier level the exit engine
            # (task 3) actually protects the position at once populated; a
            # position still on the legacy ladder only has sl_price.
            stop = p.get("trail_stop") or p.get("sl_price")
            stop_str = f"Stop ${stop:,.2f}" if stop else "Stop N/A"
            lines.append(
                f"{emoji} <b>{p['ticker']}</b>  "
                f"${p['entry_price']:,.2f}→${p['current_price']:,.2f}  "
                f"{p['unrealized_pnl_pct']:+.1f}%  (${p['dollar_pnl']:+,.0f})   "
                f"{stop_str}"
            )
    else:
        lines.append("No open positions.")

    buy_count = sell_count = hold_count = 0
    last_risk: dict = {}
    last_regime: dict = {}
    for run in week_runs:
        for rec in run.get("pm", {}).get("recommendations", []):
            action = rec.get("action", "HOLD")
            if action == "BUY":
                buy_count += 1
            elif action == "SELL":
                sell_count += 1
            else:
                hold_count += 1
        last_risk = run.get("risk") or last_risk
        last_regime = run.get("regime") or last_regime

    top_tickers: list[str] = []
    if week_runs:
        buys = [r for r in week_runs[-1].get("pm", {}).get("recommendations", [])
                if r.get("action") == "BUY"]
        buys.sort(key=lambda x: x.get("conviction", 0), reverse=True)
        top_tickers = [b["ticker"] for b in buys[:3]]

    regime_str = last_regime.get("regime", "N/A") if isinstance(last_regime, dict) else "N/A"
    lines += [
        "\n<b>━━━━━━ WEEKLY SIGNALS ━━━━━━</b>",
        f"Analyzed: {len(_TICKERS)} tickers across {len(week_runs)} session(s)",
        f"📈 BUY: {buy_count} | 📉 SELL: {sell_count} | ➖ HOLD: {hold_count}",
        f"Regime: {regime_str}  |  Sharpe: {last_risk.get('sharpe_ratio', 'N/A')}  |  VaR 95%: {last_risk.get('var_95', 'N/A')}",
        f"Risk: {last_risk.get('status', 'N/A')}",
    ]
    if top_tickers:
        lines.append(f"Top picks: {', '.join(top_tickers)}")

    return "\n".join(lines)


def _get_report_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_weekly_report() -> str:
    root = _get_report_root()
    logs_dir = os.path.join(root, "logs")

    week_runs = _scan_week_runs(logs_dir)

    position_state: dict = {}
    ps_path = os.path.join(logs_dir, "position_state.json")
    if os.path.exists(ps_path):
        try:
            with open(ps_path) as f:
                position_state = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    equity, cash, week_pnl_usd = _get_account_snapshot()
    opened = _aggregate_week_trades(week_runs)

    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    monday_iso = monday.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    closed = _closed_this_week(monday_iso)

    open_positions = _get_open_positions(position_state)
    unrealized_total = sum(p["dollar_pnl"] for p in open_positions)
    realized_total = sum(c.get("dollar_pnl") or 0.0 for c in closed)

    week_label = f"Week of: {monday.strftime('%b %-d')} – {today.strftime('%b %-d')}"

    msg1 = _format_msg1(week_label, equity, cash, unrealized_total, realized_total,
                        week_pnl_usd, opened, closed)
    msg2 = _format_msg2(open_positions, week_runs)

    send_message(msg1)
    send_message(msg2)

    os.makedirs(logs_dir, exist_ok=True)
    md_text = re.sub(r"<[^>]+>", "", msg1 + "\n\n" + msg2)
    report_path = os.path.join(logs_dir, f"weekly_report_{today.strftime('%Y%m%d')}.md")
    with open(report_path, "w") as f:
        f.write(md_text)
    print(f"[Report] Saved to {report_path}")

    return md_text


if __name__ == "__main__":
    generate_weekly_report()
