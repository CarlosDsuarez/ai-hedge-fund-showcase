"""
backtest/reference_backtest.py — daily event-driven backtest of the reference strategy.

SHOWCASE: this replaces the private research layer (factor/IC studies,
walk-forward and parameter sweeps), which is not published. It reuses the SAME
building blocks the live pipeline uses, so the mechanics shown here are the
production mechanics, driven by the placeholder parameters in config.py:

  signal  agents.strategy_agent._compute_momentum / _zscore   (12-1 momentum z)
  regime  agents.regime_agent.direction_series / breadth_series / size_multiplier_for
  sizing  agents.position_sizer.slot_size_pct                 (slot × regime × vol)
  stops   data.fetcher.atr_from_ohlc + execution.order_manager.update_trail
          (initial ATR stop, chandelier trail that only rises)

Timing (no look-ahead): signals use closes up to t-1; entries fill at Open[t]
plus a per-side cost; a stop fills at min(Open[t], stop) when Low[t] touches it.
Rebalance: every REBALANCE_DAYS sessions, names that fall out of the top set
are sold at the open.

Run (offline, synthetic data):
    DATA_MODE=synthetic python -m backtest.reference_backtest --start 2021-01-01 --end 2025-12-31

On synthetic data the resulting numbers are meaningless as performance claims;
they demonstrate that the engine runs end to end.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import config
from agents.position_sizer import slot_size_pct
from agents.regime_agent import breadth_series, direction_series, size_multiplier_for
from agents.strategy_agent import _compute_momentum, _zscore
from data import synthetic
from data.fetcher import atr_from_ohlc
from execution.order_manager import update_trail

REBALANCE_DAYS = 21          # SHOWCASE: placeholder
COST_BPS_PER_SIDE = 5.0      # SHOWCASE: placeholder
INITIAL_CAPITAL = 100_000.0


def load_panel(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """OHLC wide frames for tickers + SPY + ^VIX (VIX column renamed "VIX")."""
    symbols = list(dict.fromkeys(list(tickers) + ["SPY", "^VIX"]))
    if synthetic.enabled():
        return synthetic.ohlc_panel(symbols, start, end)
    import yfinance as yf  # live mode only
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    return {f: raw[f].rename(columns={"^VIX": "VIX"}) for f in ("Open", "High", "Low", "Close")}


def run(tickers: list[str], start: str, end: str, warmup_days: int = 400) -> dict:
    data_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    p = load_panel(tickers, data_start, end)
    o, h, lo, c = (p[k][tickers] for k in ("Open", "High", "Low", "Close"))
    atr = atr_from_ohlc(h, lo, c)
    sigma = c.pct_change().rolling(63).std() * math.sqrt(252)
    direction = direction_series(p["Close"]["SPY"])
    breadth = breadth_series(c)
    vix = p["Close"]["VIX"]

    days = c.index[c.index >= pd.Timestamp(start)]
    cash, book, equity_curve, trades = INITIAL_CAPITAL, {}, [], []
    cost = COST_BPS_PER_SIDE / 1e4
    targets: set[str] = set()

    for i, day in enumerate(days):
        prev = c.index[c.index.get_loc(day) - 1]

        # 1. Rebalance decision from data up to t-1.
        if i % REBALANCE_DAYS == 0:
            mom = _compute_momentum(c.loc[:prev])
            z = _zscore(mom) if len(mom) > 1 else pd.Series(dtype=float)
            ranked = z[z >= config.SIGNAL_THRESHOLD].sort_values(ascending=False)
            targets = set(ranked.index[: config.TARGET_POSITIONS])
            for t in [t for t in book if t not in targets]:
                px = o.at[day, t] * (1 - cost)
                cash += book[t]["qty"] * px
                trades.append({"ticker": t, "exit": day, "reason": "REBALANCE",
                               "pnl": book[t]["qty"] * (px - book[t]["entry_px"])})
                del book[t]

        # 2. Entries at the open for targets not yet held.
        equity_open = cash + sum(r["qty"] * o.at[day, t] for t, r in book.items())
        regime_mult = size_multiplier_for(direction.get(prev, "BULL"), breadth.get(prev), vix.get(prev))
        for t in sorted(targets - set(book)):
            pct = slot_size_pct(regime_mult, None, sigma.at[prev, t])
            px = o.at[day, t] * (1 + cost)
            qty = math.floor(min(pct * equity_open, cash) / px) if px > 0 else 0
            a = atr.at[prev, t]
            if qty <= 0 or not np.isfinite(a):
                continue
            cash -= qty * px
            book[t] = {"qty": qty, "entry_px": px, "atr_14": a, "highest_close": px,
                       "trail_stop": px - config.STOP_ATR_MULT * a}

        # 3. Broker-style stops on today's bar, then ratchet the trail on the close.
        for t in list(book):
            rec = book[t]
            if lo.at[day, t] <= rec["trail_stop"]:
                px = min(o.at[day, t], rec["trail_stop"]) * (1 - cost)
                cash += rec["qty"] * px
                trades.append({"ticker": t, "exit": day, "reason": "STOP",
                               "pnl": rec["qty"] * (px - rec["entry_px"])})
                del book[t]
            else:
                book[t] = update_trail(rec, c.at[day, t]) | {"qty": rec["qty"], "entry_px": rec["entry_px"]}

        equity_curve.append((day, cash + sum(r["qty"] * c.at[day, t] for t, r in book.items())))

    eq = pd.Series(dict(equity_curve), name="equity")
    bench = c.loc[days].mean(axis=1)
    return {"equity": eq, "benchmark": bench / bench.iloc[0] * INITIAL_CAPITAL,
            "trades": pd.DataFrame(trades), "metrics": metrics(eq)}


def metrics(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    years = max(len(equity) / 252, 1e-9)
    vol = float(rets.std() * math.sqrt(252)) if len(rets) > 1 else 0.0
    excess = rets.mean() * 252 - config.RISK_FREE_ANNUAL
    return {
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1),
        "ann_vol": vol,
        "sharpe": float(excess / vol) if vol > 0 else 0.0,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args(argv)
    out = run(config.TICKERS, args.start, args.end)
    m = out["metrics"]
    label = "SYNTHETIC DATA — illustrative only" if synthetic.enabled() else "live data"
    print(f"\nReference backtest ({label}) {args.start} → {args.end}, {len(config.TICKERS)} tickers")
    print(f"  final equity  {out['equity'].iloc[-1]:>14,.2f}   (equal-weight universe: {out['benchmark'].iloc[-1]:,.2f})")
    for k in ("cagr", "ann_vol", "max_drawdown"):
        print(f"  {k:<13} {m[k]:>14.2%}")
    print(f"  {'sharpe':<13} {m['sharpe']:>14.2f}")
    t = out["trades"]
    print(f"  closed trades {len(t):>14}" + (f"   ({(t['reason'] == 'STOP').sum()} stops)" if len(t) else ""))
    return out


if __name__ == "__main__":
    main()
