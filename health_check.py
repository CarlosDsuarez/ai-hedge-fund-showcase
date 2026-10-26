#!/usr/bin/env python3
"""System health check for the AI Hedge Fund scheduler."""

import json
import os
import subprocess
from datetime import datetime, UTC

import requests
from utils.env import load_project_env

from execution.alpaca import PAPER_BASE_URL

load_project_env()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAST_RUN_FILE = os.path.join(BASE_DIR, "logs", "last_run.json")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
_PAPER = os.getenv("PAPER_TRADING", "false").lower() == "true"
BASE_URL = PAPER_BASE_URL  # paper only; live trading is not supported
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

warnings = []


def _check_anthropic_key(warnings_list: list) -> None:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        warnings_list.append("ANTHROPIC_API_KEY is missing or empty")


def _alpaca_get(endpoint: str) -> dict | list:
    resp = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def check_launchagent() -> tuple[bool, str]:
    result = subprocess.run(
        ["launchctl", "list"],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "aihedgefund" in line:
            parts = line.split()
            pid = parts[0] if parts else "-"
            running = pid != "-" and pid.isdigit()
            return running, pid
    return False, "-"


def check_last_run() -> dict:
    if not os.path.exists(LAST_RUN_FILE):
        return {}
    with open(LAST_RUN_FILE) as f:
        return json.load(f)


def format_age(ts_str: str) -> str:
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        delta = now - ts
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        return f"{hours}h {minutes}m ago"
    except Exception:
        return ts_str


def main():
    print("=" * 60)
    print("  AI HEDGE FUND — HEALTH CHECK")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # --- Config checks ---
    _check_anthropic_key(warnings)
    anthropic_status = "PRESENT" if os.getenv("ANTHROPIC_API_KEY") else "MISSING ⚠️"
    paper_mode = "PAPER" if _PAPER else "LIVE"
    print(f"\n[Config]  Anthropic API Key: {anthropic_status}  |  Trading Mode: {paper_mode}")

    # --- LaunchAgent status ---
    running, pid = check_launchagent()
    status_str = f"RUNNING (PID {pid})" if running else "NOT RUNNING"
    print(f"\n[LaunchAgent]  {status_str}")
    if not running:
        warnings.append("LaunchAgent is not running")

    # --- Last run info ---
    print("\n[Last Run]")
    data = check_last_run()
    if not data:
        print("  No last_run.json found")
        warnings.append("No last_run.json found")
    else:
        ts = data.get("timestamp", "unknown")
        print(f"  Timestamp : {ts}  ({format_age(ts)})")

        regime_info = data.get("regime", {})
        if isinstance(regime_info, dict):
            regime_name = regime_info.get("regime", "unknown")
            confirmed = regime_info.get("confirmed", False)
            days = regime_info.get("days_in_regime", "?")
            switched = regime_info.get("switched_today", False)
            print(f"  Regime    : {regime_name}  (confirmed={confirmed}, days={days}, switched_today={switched})")
        else:
            print(f"  Regime    : {regime_info}")

        orders = data.get("orders_submitted", [])
        closed = data.get("closed_positions", [])
        print(f"  Orders    : {len(orders)} submitted")
        for o in orders:
            ticker = o.get("symbol") or o.get("ticker", "?")
            side = o.get("side", "?")
            qty = o.get("qty", "?")
            print(f"              {side.upper()} {qty} {ticker}")
        print(f"  Closed    : {len(closed)} positions closed")

    # --- Alpaca account ---
    print("\n[Alpaca Account]")
    try:
        account = _alpaca_get("/v2/account")
        equity = float(account.get("equity", 0))
        cash = float(account.get("cash", 0))
        buying_power = float(account.get("buying_power", 0))
        day_pnl = float(account.get("equity", 0)) - float(account.get("last_equity", account.get("equity", 0)))
        print(f"  Equity      : ${equity:>12,.2f}")
        print(f"  Cash        : ${cash:>12,.2f}")
        print(f"  Buying Power: ${buying_power:>12,.2f}")
        print(f"  Day P&L     : ${day_pnl:>+12,.2f}")
        if account.get("account_blocked") or account.get("trading_blocked"):
            warnings.append("Alpaca account is blocked")
    except Exception as e:
        print(f"  ERROR: {e}")
        warnings.append(f"Alpaca account fetch failed: {e}")

    # --- Open positions ---
    print("\n[Open Positions]")
    try:
        positions = _alpaca_get("/v2/positions")
        if not positions:
            print("  No open positions")
        else:
            header = f"  {'Symbol':<8} {'Qty':>6} {'Avg Cost':>10} {'Mkt Value':>12} {'Unrealized P&L':>16}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            total_unrealized = 0.0
            for p in positions:
                sym = p.get("symbol", "?")
                qty = float(p.get("qty", 0))
                avg_entry = float(p.get("avg_entry_price", 0))
                market_val = float(p.get("market_value", 0))
                unrealized = float(p.get("unrealized_pl", 0))
                total_unrealized += unrealized
                pnl_str = f"${unrealized:>+,.2f}"
                print(f"  {sym:<8} {qty:>6.0f} ${avg_entry:>9,.2f} ${market_val:>11,.2f} {pnl_str:>16}")
            print(f"  {'':8} {'':6} {'Total unrl P&L:':>22} ${total_unrealized:>+11,.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        warnings.append(f"Alpaca positions fetch failed: {e}")

    # --- Final verdict ---
    print("\n" + "=" * 60)
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
        print(f"\n  WARNING: {warnings[0]} ⚠️")
    else:
        print("  System healthy ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
