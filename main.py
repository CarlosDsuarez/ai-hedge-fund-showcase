import argparse
import fcntl
import json
import os
import shutil
import sys
from datetime import datetime


import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from utils.env import load_project_env

load_project_env()

from db.database import (
    init_db,
    load_position_metadata, save_position_metadata,
    save_run, save_signals, append_conviction_log,
    save_pending_signals, load_pending_signals,
    mark_pending_executed, expire_pending_signals,
)

try:
    init_db()   # creates tables if not exist; no-op if already created
except Exception as _db_e:
    print(f"  [DB] init skipped, using JSON only: {_db_e}")


parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                    help="Print decisions without submitting orders")
_args, _ = parser.parse_known_args()
DRY_RUN = _args.dry_run

TICKERS = config.TICKERS   # single source of truth — edit config.py to change universe

PAPER_TRADING = os.getenv("PAPER_TRADING", "false").lower() == "true"


# Account fields the entry phase needs but does not fetch itself. run_exits
# already calls get_account(); a second call would double the broker round-trips
# and could disagree with itself mid-run.
ACCOUNT_SNAPSHOT: dict = {}
# What the cycle was allowed to spend and what it actually spent — read by the
# health check and the funnel report. It used to be a local that died with the
# function, so "the budget went unused" was unobservable.
LAST_BUDGET: dict = {}


def _load_current_positions() -> dict:
    if not PAPER_TRADING:
        return {}
    try:
        from execution.alpaca import get_account, get_positions
        account = get_account()
        equity = float(account.get("equity", 0))
        if equity <= 0:
            return {}
        positions = get_positions()
        return {
            pos["symbol"].replace(".", "-"): float(pos["market_value"]) / equity
            for pos in positions
            if float(pos.get("market_value", 0)) != 0
        }
    except Exception as e:
        print(f"  [Warning] Could not fetch Alpaca positions: {e}. Using empty portfolio.")
        return {}


CURRENT_POSITIONS = _load_current_positions()
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

_POSITION_METADATA_FILE = os.path.join(LOGS_DIR, "position_metadata.json")
_PENDING_SIGNALS_FILE = os.path.join(LOGS_DIR, "pending_signals.json")


def _dualwrite_pending_json():
    """Mirror the live PENDING queue to JSON (backup parity with position_metadata)."""
    try:
        with open(_PENDING_SIGNALS_FILE, "w") as f:
            json.dump(load_pending_signals(), f, indent=2, default=str)
    except Exception:
        pass


def _load_position_metadata() -> dict:
    # DB primary; fall back to JSON if DB empty (first run after migration)
    meta = load_position_metadata()
    if meta:
        return meta
    try:
        if os.path.exists(_POSITION_METADATA_FILE):
            with open(_POSITION_METADATA_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_position_metadata(metadata: dict):
    save_position_metadata(metadata)   # DB primary
    # Dual-write JSON backup
    with open(_POSITION_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# RUN LOCK — prevents duplicate analysis when multiple scheduler instances
# are running in parallel (LaunchAgent KeepAlive restarts, manual starts, etc.)
# Uses fcntl.flock so the lock is released automatically on crash — no stale files.
# ---------------------------------------------------------------------------
_LOCK_FILE = os.path.join(LOGS_DIR, "main.lock")
_lock_fh = None


def _acquire_run_lock() -> bool:
    """Try to acquire an exclusive run lock. Returns True if acquired."""
    global _lock_fh
    _lock_fh = open(_LOCK_FILE, "w")
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fh.write(f"{os.getpid()}\n")
        _lock_fh.flush()
        return True
    except OSError:
        _lock_fh.close()
        _lock_fh = None
        return False


def _release_run_lock():
    """Release the run lock and remove the lock file."""
    global _lock_fh
    if _lock_fh:
        try:
            fcntl.flock(_lock_fh, fcntl.LOCK_UN)
            _lock_fh.close()
            os.remove(_LOCK_FILE)
        except OSError:
            pass
        _lock_fh = None


def _sep(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)



def main():
    # Guard: exit immediately if another run is already in progress.
    if not _acquire_run_lock():
        print(
            "[LOCK] main.py is already running (another scheduler instance or manual run). "
            "Exiting to prevent duplicate analysis and token waste."
        )
        sys.exit(0)

    try:
        _run()
    except Exception as e:
        # Pipeline runs unattended (scheduler 09:00 ET) — a silent failure means
        # no signals, no exits, and nobody knows. Alert before re-raising.
        try:
            from alerts.telegram_alert import send_message
            send_message(f"❌ RUN FAILED: {type(e).__name__}: {e}")
        except Exception as alert_err:
            print(f"  [Alert] Could not send failure alert: {alert_err}")
        raise
    finally:
        _release_run_lock()


def run_signals():
    """Phase 1 — price prefetch, research and sentiment over config.TICKERS."""
    tickers = list(TICKERS)

    # Warm the price cache with one batched request (vs 1 request per ticker)
    from data import fetcher as _fetcher
    _fetcher.reset_fetch_stats()
    cached = 0
    try:
        cached = _fetcher.prefetch_prices(tickers)
    except Exception as _pf_err:
        print(f"  [Prefetch] batch price download failed ({_pf_err}) — per-ticker fallback")

    # Fail loudly on a half-empty universe. Downstream every agent degrades
    # politely — get_stock_info returns "Unknown", get_stock_price returns None,
    # research falls back to conviction 1 — and the run ends up reporting a
    # confident "nothing worth buying" built on missing data. A run that cannot
    # see the market should say so, not trade on the fog.
    missing = max(len(tickers) - cached, 0)
    fail_pct = missing / len(tickers) if tickers else 0.0
    if fail_pct > config.MAX_DATA_FAILURE_PCT:
        msg = (f"{missing}/{len(tickers)} tickers ({fail_pct:.0%}) returned no price "
               f"history — above the {config.MAX_DATA_FAILURE_PCT:.0%} limit. "
               f"Aborting rather than trading on partial data.")
        print(f"  [Prefetch] ABORT: {msg}")
        try:
            from alerts.telegram_alert import send_message
            send_message(f"🛑 <b>Corrida abortada</b> — datos insuficientes\n{msg}")
        except Exception as _ae:
            print(f"  [Alert] Could not send data-failure alert: {_ae}")
        raise RuntimeError(f"Insufficient price data: {msg}")
    if missing:
        print(f"  [Prefetch] {cached}/{len(tickers)} tickers with price history "
              f"({fail_pct:.0%} missing, under the {config.MAX_DATA_FAILURE_PCT:.0%} limit)")

    _sep("STEP 1: RESEARCH AGENT")
    import agents.research_agent as _research_agent
    print("  Analyzing all tickers in parallel chunks...")
    research_results = _research_agent.run_research(tickers)
    for r in research_results:
        print(f"  {r['ticker']:<8} {r['recommendation']:<5} conv={r['conviction']}  {r['thesis'][:80]}")

    _research_stats = getattr(_research_agent, "LAST_RESEARCH_STATS", {}) or {}
    if _research_stats.get("degraded"):
        print(f"  [Research] DEGRADED: {_research_stats.get('fallback_tickers', 0)}/"
              f"{_research_stats.get('tickers', 0)} tickers fell back to HOLD across "
              f"{_research_stats.get('chunks_failed', 0)}/{_research_stats.get('chunks', 0)} failed chunk(s).")

    _sep("STEP 1b: SENTIMENT AGENT")
    from agents.sentiment_agent import run_sentiment
    print("  Fetching headlines and scoring sentiment in parallel...")
    sentiment_results = run_sentiment(tickers)
    for s in sentiment_results:
        score_str = f"{s['sentiment_score']:+.2f}"
        print(f"  {s['ticker']:<8} {s['sentiment']:<10} score={score_str}  {s['summary'][:70]}")

    # STEP 1c: sentiment factor for the strategy composite.
    # SENTIMENT_WEIGHT is 0.0 by default (Sprint 1 Task 7 — RSS sentiment's IC
    # has never been measured; that's Sprint 2), so the composite stays
    # momentum/value only and this returns {}. Sentiment is not dropped from
    # the system either way: it still feeds the PM prompt (agents/pm_agent.py)
    # and the SENTIMENT_NEGATIVE entry gate (agents/entry_gates.py).
    blended_sentiment: dict[str, float] = {}
    if config.SENTIMENT_WEIGHT > 0:
        blended_sentiment = {
            s["ticker"]: float(s.get("sentiment_score", 0.0)) for s in sentiment_results
        }
    else:
        print("  [Sentiment] SENTIMENT_WEIGHT=0 — composite is momentum/value "
              "60/40; headlines still feed the PM prompt and the "
              "SENTIMENT_NEGATIVE gate")

    return tickers, research_results, sentiment_results, blended_sentiment


def run_gates(tickers, research_results, blended_sentiment):
    """Phase 2 — risk report, regime detection and strategy signals."""
    _sep("STEP 2: RISK AGENT")
    from agents.risk_agent import analyze_portfolio_risk
    print(f"  Evaluating current positions: {CURRENT_POSITIONS}")
    risk_report = analyze_portfolio_risk(CURRENT_POSITIONS, universe=tickers)
    print(f"  VaR 95%:       {risk_report['var_95']}")
    print(f"  Max Drawdown:  {risk_report['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio:  {risk_report['sharpe_ratio']}")
    print(f"  Avg Corr:      {risk_report['avg_correlation']}")
    print(f"  Total Exposure:{risk_report['total_exposure']:.2%}")
    print(f"  Status:        {risk_report['status']}")
    if risk_report["violations"]:
        for v in risk_report["violations"]:
            print(f"  VIOLATION: {v}")

    # Detect regime before PM so PM can incorporate it
    from agents.regime_agent import detect_regime, _default_result
    regime_result = _default_result()
    regime_signals: dict = {}
    try:
        from agents.regime_strategy_agent import generate_regime_signals
        regime_result = detect_regime()
        regime_signals = generate_regime_signals(regime_result['regime'])
    except Exception as _re:
        print(f"  [Regime] Detection failed: {_re}")

    _sep("STEP 3: STRATEGY AGENT")
    # The credit filter fails CLOSED: a broken HYG/LQD download and a genuine
    # crisis both suppress every BUY in the universe, and until now they looked
    # identical from outside. run_gates cannot reach the recorder, so the flag
    # travels in the signals dict and run_entries records it.
    from agents.strategy_agent import generate_signals
    print("  Generating VWAP+Momentum signals...")
    strategy_signals = generate_signals(tickers, sentiment_scores=blended_sentiment)
    for ticker, sig in strategy_signals.items():
        regime_sig = regime_signals.get(ticker, {})
        confirmed = sig['signal'] == 'BUY' and regime_sig.get('signal') == 'BUY'
        conf_str = ' *** CONFIRMED ***' if confirmed else ''
        print(f"  {ticker:<8} signal={sig['signal']:<5} vwap={sig['vwap_signal']:<5} mom={sig['mom_signal']:<5}{conf_str}")

    # The composite per ticker feeds the exit engine's rank decay (run_exits)
    # and the entry gates. The time-decay agent that used to turn these into
    # SELL overrides here is retired: it never fired, because its counter was
    # rewritten to 0 by _save_position_metadata every run.
    strategy_score_map = {
        t: sig.get("composite_score", 0.0)
        for t, sig in strategy_signals.items()
    }

    return (risk_report, regime_result, regime_signals, strategy_signals,
            strategy_score_map, research_results)


def run_decisions(tickers, research_results, sentiment_results, risk_report,
                  regime_result, regime_signals, strategy_signals):
    """Phase 3 — PM synthesis plus the regime and final-summary displays."""
    _sep("STEP 4: PM AGENT")
    from agents.pm_agent import run_pm
    print("  Synthesizing research + risk into final recommendations...")
    top_regime_buys = [t for t, v in regime_signals.items() if v.get('signal') == 'BUY']
    min_conviction = config.effective_min_conviction(regime_result.get('regime', 'BULL'))
    risk_report['regime_context'] = (
        f"Current market regime: {regime_result['regime']}. "
        f"Strategic layer recommends: {', '.join(top_regime_buys) or 'none'}. "
        f"Entry gate this cycle: conviction >= {min_conviction}"
    )
    pm_result = run_pm(research_results, risk_report, sentiment_results=sentiment_results,
                       min_conviction=min_conviction)
    print(f"\n  PM Comment: {pm_result.get('portfolio_comment', '')}\n")
    for rec in pm_result.get("recommendations", []):
        size = rec.get("size_pct", 0)
        print(f"  {rec['ticker']:<8} {rec['action']:<5} size={size:.1f}%  conv={rec['conviction']}  {rec['rationale'][:60]}")

    _sep("STEP 4b: REGIME AGENT")
    rr = regime_result
    conf_label = 'confirmed' if rr.get('confirmed') else 'unconfirmed'
    print(f"  [REGIME] {rr.get('regime')} | SPY={rr.get('spy_price')} | "
          f"EMA200={rr.get('spy_ema200')} | Buffer={rr.get('buffer_pct', 0):+.1%} | "
          f"Day {rr.get('days_in_regime')} {conf_label}")
    print(f"  [ADX]    ADX={rr.get('adx', 25.0):.1f} (observability only)")
    print(f"  [SIZE]   Multiplier={rr.get('size_multiplier', 1.0):.2f}x = "
          f"Direction={rr.get('direction_mult', 1.0):.2f} x "
          f"Breadth={rr.get('breadth_mult', 1.0):.2f} ({rr.get('breadth_pct', 1.0):.0%}) x "
          f"VIX={rr.get('vix_mult', 1.0):.2f}")
    print(f"  [VOL]    VIX={rr.get('vix', 20.0):.1f} (SMA10={rr.get('vix_sma10')}) | "
          f"VolRegime={rr.get('vol_regime', 'MED')} | Credit={rr.get('credit_signal', 'NORMAL')}")
    if rr.get('switched_today'):
        print(f"  *** REGIME SWITCH TODAY to {rr.get('regime')} ***")
    if rr.get('regime') == 'BEAR':
        print(f"  *** BEAR MARKET — defensive mode, bear universe active, "
              f"conviction >= {config.effective_min_conviction('BEAR')} ***")
    regime_buys = {t: v for t, v in regime_signals.items() if v.get('signal') == 'BUY'}
    print(f"  Strategic BUYs ({rr.get('regime')} regime, {len(regime_buys)} stocks):")
    for _t, _v in list(regime_buys.items())[:10]:
        print(f"    {_t:<8} score={_v.get('score', 0):.4f}  rank={_v.get('rank')}  [{_v.get('strategy')}]")

    _sep("FINAL SUMMARY")
    print("\n  [Research Snapshot]")
    research_map = {r["ticker"]: r for r in research_results}
    sent_map = {s["ticker"]: s for s in sentiment_results}
    pm_map = {r["ticker"]: r for r in pm_result.get("recommendations", [])}
    strat_map = strategy_signals

    print(f"  {'Ticker':<8} {'Rec':<6} {'Sent':<10} {'PM':<6} {'Strat':<6} {'Score':>6} {'Reg':<5} {'Conf':<5} {'Conv':>4}  Thesis")
    print("  " + "-" * 90)
    for ticker in tickers:
        r = research_map.get(ticker, {})
        s = sent_map.get(ticker, {})
        p = pm_map.get(ticker, {})
        st = strat_map.get(ticker, {})
        reg_sig = regime_signals.get(ticker, {}).get('signal', 'N/A')
        confirmed = st.get('signal') == 'BUY' and reg_sig == 'BUY'
        thesis_short = r.get("thesis", "")[:45]
        score_str = f"{st.get('composite_score', 0.0):+.2f}"
        print(
            f"  {ticker:<8} {r.get('recommendation','N/A'):<6} "
            f"{s.get('sentiment','N/A'):<10} {p.get('action','N/A'):<6} "
            f"{st.get('signal','N/A'):<6} {score_str:>6} {reg_sig:<5} {'YES' if confirmed else 'no':<5} "
            f"{r.get('conviction','-'):>4}  {thesis_short}"
        )

    print("\n  [Risk Snapshot]")
    print(f"  VaR(95%)={risk_report['var_95']}  Sharpe={risk_report['sharpe_ratio']}  "
          f"MaxDD={risk_report['max_drawdown']:.2%}  Status={risk_report['status']}")

    print("\n  [Trade Recommendations with Strategy Signals]")
    buy_recs = [r for r in pm_result.get("recommendations", []) if r["action"] == "BUY"]
    sell_recs = [r for r in pm_result.get("recommendations", []) if r["action"] == "SELL"]
    for rec in buy_recs:
        st = strat_map.get(rec["ticker"], {})
        print(f"  BUY  {rec['ticker']:<8} size={rec['size_pct']:.1f}%  strat={st.get('signal','N/A')}  {rec['rationale'][:55]}")
    for rec in sell_recs:
        print(f"  SELL {rec['ticker']:<8} {rec['rationale'][:65]}")

    return pm_result


def _sell_and_confirm(ticker: str, qty: int, *, tag: str = "main") -> tuple[int, float]:
    """Submit a sell and report what actually executed.

    The implementation moved to execution/trade_executor.py so the terminal
    chat can reuse it: importing main.py from the chat would fire the live
    get_account()/get_positions() call this module makes at import time.
    Kept as a wrapper rather than a bare re-export because main.py's own call
    sites and the suite both address it as main._sell_and_confirm.
    """
    from execution.trade_executor import sell_and_confirm

    return sell_and_confirm(ticker, qty, tag=tag)


def run_exits(*, pm_result=None, strategy_score_map=None):
    """Phase 4 — account snapshot, ledger reconciliation, the exit engine,
    drift rebalancing and the broker stop audit.

    pm_result / strategy_score_map come from the decision phase: a PM SELL on
    a held name (conviction >= PM_SELL_MIN_CONVICTION) and a composite rank
    below RANK_DECAY_PCTL are two of the engine's exit reasons. Neither used to
    reach this phase at all — pm_sell_tickers only ever blocked re-buys.
    """
    stop_audit_result: dict = {}   # written to run_data and read by the health check

    _sep("PAPER TRADING: EXECUTION")
    from execution.alpaca import get_account, get_positions
    from execution.drift_rebalancer import compute_trim_orders

    try:
        account = get_account()
        equity = float(account.get("equity", 0))
        ACCOUNT_SNAPSHOT.clear()
        ACCOUNT_SNAPSHOT.update({
            "equity": equity,
            "buying_power": float(account.get("buying_power", 0) or 0),
            "cash": float(account.get("cash", 0) or 0),
            "long_market_value": float(account.get("long_market_value", 0) or 0),
        })
        print(f"  Account equity: ${equity:,.2f} | "
              f"buying power: ${ACCOUNT_SNAPSHOT['buying_power']:,.2f} | "
              f"cash: ${ACCOUNT_SNAPSHOT['cash']:,.2f}")
    except Exception as e:
        print(f"  [Alpaca] Could not fetch account: {e}")
        equity = 100_000.0
        ACCOUNT_SNAPSHOT.clear()

    # ---- LEDGER RECONCILIATION ----
    # Before anything reads get_positions(): a broker-side stop or a manual
    # DELETE /v2/positions changes what "open positions" even means, and every
    # check below (exit engine, drift, stop audit) would otherwise reason
    # about a position the broker already closed.
    broker_exits: list = []
    if config.LEDGER_ENABLED:
        try:
            from execution.ledger import reconcile_fills
            _ledger_result = reconcile_fills(dry_run=DRY_RUN)
            broker_exits = _ledger_result.get("broker_exits", [])
            if _ledger_result.get("error"):
                print(f"  [Ledger] reconcile_fills error: {_ledger_result['error']}")
            else:
                print(f"  [Ledger] fills={_ledger_result.get('new_fills', 0)} "
                      f"trades={_ledger_result.get('trades_rebuilt', 0)} "
                      f"broker_exits={len(broker_exits)}")
        except Exception as _ledger_e:
            print(f"  [Ledger] reconcile_fills call failed: {_ledger_e}")
    # ---- end LEDGER RECONCILIATION ----

    try:
        positions = get_positions()
    except Exception as e:
        print(f"  [Alpaca] Could not fetch positions: {e}")
        positions = []

    # PM SELL verdicts the engine may act on: held names only, and only when
    # the PM actually ran. A degraded pm_result carries research verdicts
    # wearing a PM label, and research is not allowed to sell the book.
    held_now = {
        (p.get("symbol") or "").replace(".", "-") for p in positions
        if float(p.get("market_value") or 0) > 0
    }
    pm_sells: dict = {}
    if isinstance(pm_result, dict) and not pm_result.get("degraded"):
        pm_sells = {
            r["ticker"]: int(r.get("conviction") or 0)
            for r in pm_result.get("recommendations", [])
            if r.get("action") == "SELL" and r.get("ticker") in held_now
        }
    if pm_sells:
        print("  [PM] SELL verdicts on held names: "
              + ", ".join(f"{t}(conv {c})" for t, c in pm_sells.items()))

    print(f"\n  A) Exit engine on {len(positions)} open positions"
          f"{' [dry run]' if DRY_RUN else ''}...")
    exited_tickers: set = set()
    closed_positions: list = []
    try:
        from execution.order_manager import run_exit_engine
        exited_tickers, closed_positions = run_exit_engine(
            positions, score_map=strategy_score_map or {}, pm_sells=pm_sells,
            metadata=_load_position_metadata(), dry_run=DRY_RUN)
    except Exception as _engine_err:
        # The stop audit below still has to run: a broken engine pass must not
        # also leave the book without broker protection.
        print(f"  [Exit] Engine failed: {_engine_err} — continuing to the stop audit")
        exited_tickers, closed_positions = set(), []
    # Broker exits ledger reconciliation caught (a stop that fired, or a
    # position that simply vanished from the broker) — folded in here so
    # send_execution_alerts and save_run see them same as any other close.
    closed_positions = list(closed_positions) + broker_exits

    try:
        positions = get_positions()
    except Exception:
        positions = []

    # ---- DRIFT REBALANCING ----
    # Runs BEFORE the stop audit on purpose. The broker stop reserves the
    # whole position, so trimming afterwards asked Alpaca to sell shares it
    # had just locked — 51 of the run 403s were exactly that. All quantity
    # changes now happen first; protection is placed last, on the final
    # quantities.
    _sep("EXECUTION STEP A: DRIFT REBALANCING")
    trimmed_tickers: set = set()
    failed_trims: list = []
    try:
        _trim_orders = compute_trim_orders(positions, equity)
        if _trim_orders:
            for _trim in _trim_orders:
                _tticker = _trim["ticker"]
                _shares  = _trim["shares_to_sell"]
                _weight  = _trim["weight"]
                if _shares >= 1:
                    print(f"  [TRIM] {_tticker}: {_weight:.1%} > "
                          f"{config.MAX_POSITION_PCT * config.DRIFT_TRIM_HYSTERESIS:.1%} trigger "
                          f"→ selling {_shares} shares back to {config.MAX_POSITION_PCT:.0%}")
                    if DRY_RUN:
                        print(f"  [DryRun] Would SELL {_tticker} qty={_shares}")
                    else:
                        # A trim only counts once the shares actually leave the
                        # book: trimmed_tickers clears the position's weight
                        # block, so crediting an unfilled sell would leave a
                        # position above MAX_POSITION_PCT and unblocked.
                        _filled, _fill_px = _sell_and_confirm(_tticker, _shares, tag="drift")
                        if _filled > 0:
                            # Keep the engine record's quantity honest: the
                            # stop audit sizes off the broker, but the ledger
                            # reports an untagged exit at qty_remaining.
                            try:
                                from execution.order_manager import reduce_position_qty
                                reduce_position_qty(_tticker, _filled)
                            except Exception as _rq:
                                print(f"  [TRIM] {_tticker}: state qty not reduced — {_rq}")
                        if _filled >= _shares:
                            trimmed_tickers.add(_tticker)
                        else:
                            _why = ("no fill" if _filled == 0
                                    else f"partial fill {_filled}/{_shares}, still over cap")
                            print(f"  [TRIM] {_tticker}: {_why} — weight block stays on")
                            failed_trims.append({"ticker": _tticker, "weight": _weight,
                                                 "shares": _shares, "error": _why})
                else:
                    print(f"  [TRIM] {_tticker}: {_weight:.1%} over the trim trigger → "
                          f"shares_to_sell={_shares} < 1, no action")
        else:
            print(f"  [TRIM] No position past "
                  f"{config.MAX_POSITION_PCT * config.DRIFT_TRIM_HYSTERESIS:.0%} "
                  f"(MAX_POSITION_PCT x DRIFT_TRIM_HYSTERESIS) — no rebalancing needed")
    except Exception as _trim_err:
        print(f"  [TRIM] Drift rebalancing failed: {_trim_err}")
    if trimmed_tickers:
        print(f"  [TRIM] Successfully trimmed: {', '.join(sorted(trimmed_tickers))}")
        try:
            positions = get_positions()   # the stop audit must size off post-trim quantities
        except Exception as _e:
            print(f"  [Alpaca] Could not re-fetch positions after trim: {_e}")
    # ---- end DRIFT REBALANCING ----

    # ---- STOP AUDIT: one GTC broker stop per position, at the engine's trail ----
    # Runs even when the engine raised above: a bad engine pass must not also
    # leave the book without broker protection until tomorrow.
    _sep("EXECUTION STEP B: BROKER STOP AUDIT")
    try:
        from agents.risk_sl_tp_agent import run_stop_audit
        from execution.order_manager import load_position_state
        stop_audit_result = run_stop_audit(positions, load_position_state(),
                                           exited=exited_tickers, dry_run=DRY_RUN)
        print(f"  Protected={len(stop_audit_result['protected'])} | "
              f"Placed={len(stop_audit_result['placed'])} | "
              f"Replaced={len(stop_audit_result['replaced'])} | "
              f"Unprotected={len(stop_audit_result['unprotected'])} | "
              f"Errors={len(stop_audit_result['errors'])} | dry_run={DRY_RUN}")
        for _u in stop_audit_result["unprotected"]:
            print(f"  [Stop] UNPROTECTED {_u['ticker']}: {_u.get('reason')}")
    except Exception as _stop_err:
        print(f"  [Stop Audit] Failed: {_stop_err}")
    # ---- end STOP AUDIT ----

    return (equity, positions, exited_tickers, closed_positions,
            stop_audit_result, trimmed_tickers, failed_trims)


def run_entries(equity, positions, exited_tickers, trimmed_tickers, closed_positions,
                orders_submitted, pm_result, sentiment_results,
                research_results, strategy_signals, regime_signals, regime_result,
                strategy_score_map, funnel=None, stop_audit_result=None):
    """Phase 5 — entry gating, sizing, budget deployment and metadata persistence.

    `stop_audit_result` is run_exits' broker stop audit dict, mutated in
    place: the post-entry stop pass at the end of this phase merges its rows
    into it so the health check and run_data see one audit for the cycle.
    """
    from agents.entry_gates import EntryFacts, evaluate_entry, first_rejection
    from execution.trade_executor import buy_and_register

    from agents.budget import compute_deployment_budget, deployable_fraction
    from utils import funnel as fn

    if stop_audit_result is None:
        stop_audit_result = {}
    if funnel is None:
        funnel = fn.FunnelRecorder(datetime.now().isoformat(), list(strategy_signals.keys()))
    LAST_BUDGET.clear()

    held_tickers = {pos.get("symbol", "").replace(".", "-") for pos in positions}
    held_tickers -= exited_tickers
    weight_blocked_tickers = {
        pos.get("symbol", "").replace(".", "-")
        for pos in positions
        if float(pos.get("market_value", 0)) > 0
        and equity > 0
        and float(pos.get("market_value", 0)) / equity > config.MAX_POSITION_PCT
    } - trimmed_tickers

    pm_recs_list = pm_result.get("recommendations", [])
    pm_recs = {r["ticker"]: r for r in pm_recs_list}
    # Set by pm_agent whenever the veto gate did not actually run (budget
    # exhausted, API failure, unusable tool output). The verdicts below are
    # then research recommendations wearing a PM label.
    pm_degraded = bool(pm_result.get("degraded"))
    pm_sell_tickers = {t for t, r in pm_recs.items() if r["action"] == "SELL"}
    sent_map = {s["ticker"]: s.get("sentiment", "NEUTRAL") for s in sentiment_results}

    position_metadata = _load_position_metadata()

    from agents.rotation_agent import prioritize_deployments, classify_buy_outcome
    from db.database import recent_stop_exits
    score_map = strategy_score_map   # {ticker -> composite_score}, computed above

    # Design C re-entry cooldown: REENTRY_COOLDOWN_DAYS trading days after a
    # stop-out (software or broker) before the name can be bought again.
    cooldown_tickers = set(recent_stop_exits(config.REENTRY_COOLDOWN_DAYS))
    if cooldown_tickers:
        print(f"  [Cooldown] {', '.join(sorted(cooldown_tickers))}: stopped out in the "
              f"last {config.REENTRY_COOLDOWN_DAYS} trading days — not re-entered")

    # Expire stale queued signals so we never act on an outdated thesis.
    if not DRY_RUN:
        _expired = expire_pending_signals(config.PENDING_EXPIRY_DAYS)
        if _expired:
            print(f"  [Queue] Expired {_expired} stale pending signal(s)")
    pending_queue = load_pending_signals()
    if pending_queue:
        print("  [Queue] Pending awaiting budget: " + ", ".join(
            f"{p['ticker']}({(p.get('composite_score') or 0):+.2f})" for p in pending_queue))

    vol_size_mult = regime_result.get("size_multiplier", 1.0)

    new_entries_allowed  = regime_result.get('new_entries_allowed', True)
    current_regime       = regime_result.get('regime', 'BULL')
    # Same helper the PM prompt quotes — the two must never drift apart.
    regime_min_conviction = config.effective_min_conviction(current_regime)

    def _facts_for(ticker: str, pm_rec: dict) -> "EntryFacts":
        """Gather what agents/entry_gates.py needs about one candidate."""
        return EntryFacts(
            ticker=ticker,
            pm_action=pm_rec.get("action"),
            conviction=pm_rec.get("conviction", 0),
            composite=strategy_score_map.get(ticker),
            pm_degraded=pm_degraded,
            min_conviction=regime_min_conviction,
            regime=current_regime,
            strategy_buy=strategy_signals.get(ticker, {}).get("signal") == "BUY",
            regime_buy=regime_signals.get(ticker, {}).get("signal") == "BUY",
            sentiment=sent_map.get(ticker, "NEUTRAL"),
            held=ticker in held_tickers,
            in_cooldown=ticker in cooldown_tickers,
            weight_capped=ticker in weight_blocked_tickers,
            pm_sell=ticker in pm_sell_tickers,
            exited_today=ticker in exited_tickers,
            macro_stressed=_macro_stressed,
            credit_z=_credit_z,
        )

    def _conviction_gate_ok(conviction) -> bool:
        """Conviction bar for the pending queue, which has no PM record to pass."""
        from agents.entry_gates import _conviction_ok

        return _conviction_ok(EntryFacts(
            ticker="", conviction=conviction or 0, pm_degraded=pm_degraded,
            min_conviction=regime_min_conviction,
        ))

    def _signal_gate_ok(ticker: str) -> bool:
        """Signal evidence for the pending queue."""
        from agents.entry_gates import _signal_ok

        return _signal_ok(EntryFacts(
            ticker=ticker, pm_degraded=pm_degraded,
            composite=strategy_score_map.get(ticker),
            strategy_buy=strategy_signals.get(ticker, {}).get("signal") == "BUY",
            regime_buy=regime_signals.get(ticker, {}).get("signal") == "BUY",
        ))

    if pm_degraded:
        print(f"\n  [PM DEGRADED] Veto gate did not run — quant-only entries this "
              f"cycle: composite >= {config.SIGNAL_THRESHOLD} required, "
              f"PM conviction bar not applied.")

    if not new_entries_allowed:
        print(f"\n  [REGIME GATE] {current_regime} regime — no new entries this cycle.")
        print(f"  Holding existing positions: {list(held_tickers)}")
        # The gate used to close over the whole universe without naming a single
        # ticket it cancelled. Every candidate it stops is recorded, so a week of
        # silence has an auditable cause instead of an absence of evidence.
        funnel.record_many(
            [t for t in pm_recs if t not in held_tickers],
            fn.REJECTED, "REGIME_BLOCKED", stage=fn.STAGE_REGIME,
            detail=f"regime={current_regime}",
        )
    else:
        # Pre-compute how many tickers qualify for BUY (needed for equal-weight sizing).
        # Gate mirrors the execution loop exactly: strategy OR regime signal == BUY.
        #
        # This was one list comprehension testing eight conditions, and a name
        # failing any of them disappeared without a print, a log line or a row.
        # Same conditions, same order, same survivors — the only difference is
        # that the first condition a ticker fails is now written down.
        # The credit filter fails CLOSED, so a broken HYG/LQD download and a real
        # crisis both zero out every BUY. Same flag either way, but naming it
        # separates "the macro veto fired" from "nothing scored well enough".
        _first_signal = next(iter(strategy_signals.values()), {})
        _macro_stressed = bool(_first_signal.get("macro_stressed"))
        _credit_z = float(_first_signal.get("credit_spread_z", 0.0) or 0.0)
        if _macro_stressed:
            print(f"  [Macro] Credit stress active (z={_credit_z:+.2f}) — "
                  f"strategy BUYs suppressed universe-wide")

        _buy_candidates = []
        for t, pm_rec in pm_recs.items():
            _conv = pm_rec.get("conviction", 0)
            _score = strategy_score_map.get(t)
            _reject = first_rejection(evaluate_entry(_facts_for(t, pm_rec)))
            if _reject:
                funnel.record(t, fn.REJECTED, _reject.reason_code, stage=fn.STAGE_GATES,
                              detail=_reject.detail, score=_score, conviction=_conv)
            else:
                _buy_candidates.append(t)

        funnel.stage_count("pm_buy", sum(1 for r in pm_recs.values() if r.get("action") == "BUY"))
        funnel.stage_count("gates", len(_buy_candidates))
        if _buy_candidates:
            print(f"  [Embudo] {len(_buy_candidates)} candidato(s) pasan las compuertas: "
                  + ", ".join(_buy_candidates[:10]))

        # ── Free budget: BUDGET_ALLOCATION of the configured base ──
        # `_committed` is the market value of the book that survives this cycle.
        # Under the buying-power base the broker's figure is already net of it,
        # so it is not subtracted again — it only feeds the gross-exposure cap.
        _committed = sum(
            float(p.get("market_value", 0)) for p in positions
            if float(p.get("market_value", 0)) > 0
            and p.get("symbol", "").replace(".", "-") in held_tickers
        )
        _budget = compute_deployment_budget(
            equity, buying_power=ACCOUNT_SNAPSHOT.get("buying_power"),
            long_market_value=_committed,
        )
        free_budget = _budget["deployable"]
        _budget_start = free_budget
        deployable_frac = deployable_fraction(free_budget, equity)
        print(f"\n  [Budget] Base={_budget['base']} "
              f"BuyingPower=${float(ACCOUNT_SNAPSHOT.get('buying_power') or 0):,.0f} "
              f"Invested=${_committed:,.0f} Free=${free_budget:,.0f} "
              f"({deployable_frac:.2f}x equity, gross cap {config.MAX_GROSS_EXPOSURE:.2f}x)")
        if _budget["reason"]:
            print(f"  [Budget] {_budget['reason']}")

        # ── Shared, budget-aware buy executor (used by pending AND new buys) ──
        def _execute_buy(ticker, size_pct, pm_conv, budget_left, *, source="new",
                         funnel=None, fn=None, score=None):
            """Fund one BUY from remaining budget. Returns dollars spent (0 if skipped)."""
            def _note(code, detail, stage=None):
                if funnel is not None and fn is not None:
                    funnel.record(ticker, fn.REJECTED, code,
                                  stage=stage or fn.STAGE_ORDER, detail=detail,
                                  score=score, conviction=pm_conv)

            tgt_dollars = equity * (size_pct / 100.0)
            spend = min(tgt_dollars, budget_left)
            if spend < config.MIN_POSITION_PCT * equity:
                # The most invisible drop in the pipeline: a candidate that
                # cleared every gate went home on this line without printing.
                print(f"  [Skip] {ticker} ${spend:,.0f} below the "
                      f"${config.MIN_POSITION_PCT * equity:,.0f} minimum ticket")
                _note("BELOW_MIN_TICKET",
                      f"${spend:,.0f} < ${config.MIN_POSITION_PCT * equity:,.0f}",
                      stage=fn.STAGE_BUDGET if fn else None)
                return 0.0
            if DRY_RUN:
                print(f"  [DryRun] Would BUY {ticker} ({spend/equity*100:.1f}% of equity, "
                      f"conv={pm_conv}, src={source})")
                if funnel is not None and fn is not None:
                    funnel.record(ticker, fn.EXECUTED, "EXECUTED", stage=fn.STAGE_ORDER,
                                  detail=f"dry-run ${spend:,.0f}", score=score,
                                  conviction=pm_conv)
                _res = next((r for r in research_results if r["ticker"] == ticker), {})
                _ref_price = float(_res.get("entry_ref") or 0.0)
                append_conviction_log(datetime.now().strftime("%Y-%m-%d"), ticker,
                                      _res.get("conviction", pm_conv), _ref_price)
                return spend
            # Price fetch, submit, fill reconciliation and the local exit-engine
            # record now live in execution/trade_executor.py, shared with the
            # chat. assume_fill_on_timeout stays True here and only here: the
            # batch has to account the requested qty or the remaining budget
            # drifts. The chat cannot afford that assumption.
            _res_data = next((r for r in research_results
                              if r.get("ticker") == ticker), {})
            result = buy_and_register(
                ticker, spend, tag="main", atr_14=_res_data.get("atr_14"),
                note=f"src={source}", assume_fill_on_timeout=True,
            )
            if result.status in ("SKIPPED", "REJECTED", "ERROR"):
                _note(result.reason_code, result.detail)
                return 0.0

            try:
                filled_qty = result.filled_qty
                fill_price = result.fill_price
                _spent = result.spent
                orders_submitted.append({
                    "ticker": ticker, "qty": filled_qty, "price": round(fill_price, 2),
                    "order": result.order, "conviction": pm_conv,
                    "size_pct": _spent / equity * 100.0,
                })
                if funnel is not None and fn is not None:
                    funnel.record(ticker, fn.EXECUTED, "EXECUTED", stage=fn.STAGE_ORDER,
                                  detail=f"{filled_qty} @ ${fill_price:.2f}",
                                  score=score, conviction=pm_conv)
                append_conviction_log(datetime.now().strftime("%Y-%m-%d"), ticker,
                                      _res_data.get("conviction", pm_conv), fill_price)
                position_metadata[ticker] = {
                    "entry_price": round(fill_price, 2), "entry_date": datetime.now().isoformat(),
                    "holding_days": 0, "conviction": pm_conv,
                    "size_pct": _spent / equity * 100.0,
                }
                return _spent
            except Exception as e:
                print(f"  [Order] Error {ticker}: {e}")
                _note("ORDER_ERROR", f"{type(e).__name__}: {e}")
                return 0.0

        # ── STEP D: deploy free budget — PENDING first, then new BUYs, by score ──
        # (The CRITICAL block that halted every entry while one position sat
        # below a fixed loss threshold is gone: with an ATR-based stop that is a normal drawdown,
        # and a position the broker stop has not caught is a health alert,
        # not a reason to stop buying.)
        new_buy_tickers = list(_buy_candidates)
        deploy_order = prioritize_deployments(pending_queue, new_buy_tickers, score_map)
        print(f"  D) Deploying free budget ${free_budget:,.0f} "
              f"({len(pending_queue)} pending, {len(new_buy_tickers)} new):")

        # Pending items carry their own queue-specific gates (still
        # held/SELL/exited, cooldown, conviction, degraded signal). Filtering
        # those out BEFORE sizing means plan_entries only ever sees tickers
        # that are actually eligible to buy — a rejected pending item must not
        # consume a slot of gross room or budget it will never spend.
        sizing_order: list = []
        item_by_ticker: dict = {}
        for item in deploy_order:
            ticker = item["ticker"]
            if item["source"] == "pending":
                if ticker in held_tickers or ticker in pm_sell_tickers or ticker in exited_tickers:
                    print(f"  [Queue] {ticker}: skip — now held/SELL/exited")
                    funnel.record(ticker, fn.REJECTED, "ALREADY_HELD", stage=fn.STAGE_GATES,
                                  detail="en cola, ya en cartera/SELL/salida")
                    continue
                if ticker in cooldown_tickers:
                    print(f"  [Queue] {ticker}: skip — in cooldown after a stop-out")
                    funnel.record(ticker, fn.REJECTED, "COOLDOWN", stage=fn.STAGE_GATES,
                                  detail="en cola, stop reciente")
                    continue
                if not _conviction_gate_ok(item.get("conviction")):
                    print(f"  [Queue] {ticker}: skip — conviction below gate")
                    funnel.record(ticker, fn.REJECTED, "CONVICTION_BELOW_GATE",
                                  stage=fn.STAGE_GATES, conviction=item.get("conviction"),
                                  detail="en cola, convicción bajo la compuerta")
                    continue
                if pm_degraded and not _signal_gate_ok(ticker):
                    print(f"  [Queue] {ticker}: skip — PM degraded, composite below threshold")
                    funnel.record(ticker, fn.REJECTED, "NO_SIGNAL", stage=fn.STAGE_GATES,
                                  detail="en cola, PM degradado y composite bajo umbral")
                    continue
            sizing_order.append(ticker)
            item_by_ticker[ticker] = item

        # ── Slot-based sizing (design B): fill the book toward
        # TARGET_GROSS_EXPOSURE instead of dividing the budget by N.
        # agents.position_sizer.plan_entries is the single source of the
        # arithmetic — the daily backtest imports the same function, so
        # backtest and live can never size differently.
        from agents.position_sizer import plan_entries, realized_vol_63d, slot_size_pct

        try:
            from data.fetcher import get_closes
            # Same period string as strategy_agent/regime_agent — shares their
            # 1h price cache instead of triggering a second download.
            sigmas = realized_vol_63d(get_closes(sizing_order, period="2y"))
        except Exception as _ve:
            print(f"  [Sizing] realized_vol_63d failed, using neutral vol: {_ve}")
            sigmas = {}

        convictions: dict = {}
        for ticker in sizing_order:
            _conv = item_by_ticker[ticker].get("conviction")
            convictions[ticker] = (
                _conv if _conv is not None else pm_recs.get(ticker, {}).get("conviction")
            )

        sector_mults: dict = {}
        if config.SECTOR_ROTATION_ENABLED and sizing_order:
            try:
                from agents.sector_rotation_agent import get_sector_multipliers
                sector_mults = get_sector_multipliers(sizing_order)
            except Exception as _se:
                print(f"  [SectorRotation] Multipliers failed, using neutral tilt: {_se}")
                sector_mults = {}

        plan, plan_rejected = plan_entries(
            sizing_order, scores=score_map, convictions=convictions, sigmas=sigmas,
            regime_mult=vol_size_mult, equity=equity, long_market_value=_committed,
            deployable=free_budget, degraded=pm_degraded, sector_mults=sector_mults,
            ordered=True,
        )

        # Hard sector cap on top of the per-ticker tilt already folded into
        # plan_entries — the FINAL sizing step (do not renormalize afterwards
        # or the cap is re-violated), same contract as the old sector overlay.
        sizes_by_ticker = {p["ticker"]: p["size_pct"] for p in plan}
        if config.SECTOR_ROTATION_ENABLED and sizes_by_ticker:
            try:
                from agents.sector_rotation_agent import apply_sector_cap
                sizes_by_ticker = apply_sector_cap(
                    sizes_by_ticker, list(sizes_by_ticker.keys()),
                    config.MAX_SECTOR_WEIGHT * config.TARGET_GROSS_EXPOSURE,
                )
            except Exception as _ce:
                print(f"  [SectorRotation] Cap failed, using pre-cap sizes: {_ce}")
            sizes_by_ticker = {
                t: max(min(w, config.MAX_POSITION_PCT), config.MIN_POSITION_PCT)
                for t, w in sizes_by_ticker.items()
            }
        if sizes_by_ticker:
            print("     Slot sizing: " + ", ".join(
                f"{t}:{sizes_by_ticker[t]:.1%}" for t in sizes_by_ticker))

        # Candidates the slot plan did not fund: GROSS_TARGET_REACHED means the
        # book already reached its exposure target this cycle (rejected, not
        # queued — tomorrow starts from the same full book); NO_BUDGET means
        # the account itself ran short (queued, same as the old budget gate).
        deferred_new: list = []
        for ticker, reason in plan_rejected:
            item = item_by_ticker[ticker]
            pm_conv = convictions.get(ticker)
            if reason == "GROSS_TARGET_REACHED":
                print(f"  [Gross] {ticker}: gross exposure target reached — rejected")
                funnel.record(ticker, fn.REJECTED, "GROSS_TARGET_REACHED",
                              stage=fn.STAGE_BUDGET,
                              detail="objetivo de exposición bruta alcanzado",
                              score=score_map.get(ticker), conviction=pm_conv)
                continue
            if classify_buy_outcome(gates_passed=True, fundable=False) == "QUEUE":
                if item["source"] == "new":
                    print(f"  [Queue] {ticker}: approved BUY, no free budget → deferred to pending")
                    _info_size_pct = slot_size_pct(
                        vol_size_mult, pm_conv, sigmas.get(ticker),
                        degraded=pm_degraded, sector_mult=sector_mults.get(ticker, 1.0),
                    ) * 100
                    deferred_new.append({
                        "ticker": ticker, "composite_score": score_map.get(ticker, 0.0),
                        "conviction": pm_conv, "pm_action": "BUY", "size_pct": _info_size_pct,
                        "reason_deferred": "insufficient_budget",
                    })
                else:
                    print(f"  [Queue] {ticker}: still no budget → stays pending")
                funnel.record(ticker, fn.QUEUED, "NO_BUDGET", stage=fn.STAGE_BUDGET,
                              detail=f"libre ${free_budget:,.0f} < requerido",
                              score=score_map.get(ticker), conviction=pm_conv)

        for p in plan:
            ticker = p["ticker"]
            item = item_by_ticker[ticker]
            size_pct = sizes_by_ticker.get(ticker, p["size_pct"]) * 100
            pm_conv = convictions.get(ticker)
            spent = _execute_buy(ticker, size_pct, pm_conv, free_budget, source=item["source"],
                                 funnel=funnel, fn=fn, score=score_map.get(ticker))
            if spent > 0:
                free_budget -= spent
                if item["source"] == "pending":
                    if not DRY_RUN:
                        mark_pending_executed(ticker)
                    else:
                        print(f"  [DryRun] Would mark {ticker} pending→EXECUTED")

        # Persist brand-new BUYs we could not fund this cycle (dry-run = simulate only).
        if deferred_new:
            if DRY_RUN:
                print(f"  [DryRun] Would queue {len(deferred_new)} new signal(s): "
                      + ", ".join(d["ticker"] for d in deferred_new))
            else:
                save_pending_signals(datetime.now().isoformat(), deferred_new)
                _dualwrite_pending_json()

        # What was authorised versus what actually left the account. A budget
        # that goes unspent is a silent failure: the run reports "no new
        # positions" and nothing says the money was there the whole time.
        _deployed = max(_budget_start - free_budget, 0.0)
        LAST_BUDGET.clear()
        LAST_BUDGET.update({
            "deployable": _budget_start,
            "deployed": _deployed,
            "leftover": max(free_budget, 0.0),
            "base": _budget["base"],
            "degraded": _budget["degraded"],
            "reason": _budget["reason"],
            "equity": equity,
            "buying_power": float(ACCOUNT_SNAPSHOT.get("buying_power") or 0),
            # Slot-based sizing (design B): how full the book is against its
            # TARGET_GROSS_EXPOSURE target, for _collect_health_issues and the
            # funnel/Telegram reports.
            "target_gross_dollars": config.TARGET_GROSS_EXPOSURE * equity,
            "gross_before": _committed,
            "gross_after": _committed + _deployed,
        })
        _executed = funnel.by_outcome(fn.EXECUTED)
        _failed_at_order = [r for r in funnel.by_outcome(fn.REJECTED)
                            if r["stage"] == fn.STAGE_ORDER]
        funnel.stage_count("funded", len(_executed) + len(_failed_at_order))
        funnel.stage_count("executed", len(_executed))
        print(f"  [Budget] Desplegado ${LAST_BUDGET['deployed']:,.0f} de "
              f"${_budget_start:,.0f} — sin usar ${LAST_BUDGET['leftover']:,.0f}")

    # ---- POST-ENTRY STOP AUDIT: protect today's fills today ----
    # The morning audit ran at the end of run_exits, before these buys
    # existed, so a fresh entry had no broker stop until the next run — a
    # gap-down overnight was uncovered. Same audit, restricted to the tickers
    # bought this cycle, on freshly fetched positions. Not inside
    # buy_and_register on purpose: an assume_fill_on_timeout order has no
    # shares to hold yet, and the stop would 403. Never blocks the run.
    if orders_submitted:
        _sep("EXECUTION STEP C.1: BROKER STOP ON NEW ENTRIES")
        try:
            from agents.risk_sl_tp_agent import run_stop_audit
            from execution.alpaca import get_positions
            from execution.order_manager import load_position_state

            _new_tickers = {o["ticker"] for o in orders_submitted}
            _fresh = [p for p in get_positions()
                      if (p.get("symbol") or "").replace(".", "-") in _new_tickers]
            _post = run_stop_audit(_fresh, load_position_state(),
                                   exited=exited_tickers, dry_run=DRY_RUN)
            # A fill the broker does not list yet (a pending order accounted
            # as filled) has nothing to hold a stop against — name it so the
            # health alert does, rather than skipping it in silence.
            _listed = {(p.get("symbol") or "").replace(".", "-") for p in _fresh}
            for _t in sorted(_new_tickers - _listed):
                _post.setdefault("unprotected", []).append(
                    {"ticker": _t, "reason": "bought this run, not yet listed by the broker"})
            for _k in ("placed", "replaced", "protected", "unprotected", "errors"):
                stop_audit_result.setdefault(_k, []).extend(_post.get(_k) or [])
            print(f"  New entries: Placed={len(_post.get('placed') or [])} | "
                  f"Protected={len(_post.get('protected') or [])} | "
                  f"Unprotected={len(_post.get('unprotected') or [])} | "
                  f"Errors={len(_post.get('errors') or [])} | dry_run={DRY_RUN}")
        except Exception as _post_err:
            print(f"  [Stop Audit] Post-entry pass failed: {_post_err}")
    # ---- end POST-ENTRY STOP AUDIT ----

    _persist_holding_days(positions, exited_tickers, orders_submitted,
                          metadata=position_metadata)

    return orders_submitted, closed_positions


def _persist_holding_days(positions, exited_tickers, orders_submitted, *, metadata=None):
    """Age every tracked position by one run and drop the ones no longer held.

    Lives outside run_entries so a run with entries frozen still ages the
    book. `metadata` is run_entries' own working copy (it carries today's
    fills); without it the stored metadata is loaded fresh.
    """
    position_metadata = _load_position_metadata() if metadata is None else metadata
    held_tickers = {pos.get("symbol", "").replace(".", "-") for pos in positions}
    held_tickers -= set(exited_tickers)
    bought_today = {o["ticker"] for o in orders_submitted}
    for t in list(position_metadata.keys()):
        if t in held_tickers:
            position_metadata[t]["holding_days"] = position_metadata[t].get("holding_days", 0) + 1
        elif t not in bought_today:
            position_metadata.pop(t, None)
    _save_position_metadata(position_metadata)


def _collect_health_issues(pm_result, orders_submitted, strategy_signals,
                           regime_result, failed_trims, budget=None, stop_audit=None,
                           funnel_summary=None, research_stats=None,
                           entries_frozen=False):
    """Name every way this run completed without doing its job.

    Each of these used to be invisible: the run printed "Run complete" and
    exited 0 with no PM output, 403s on every write, and zero orders.
    """
    from execution.alpaca import FORBIDDEN_EVENTS

    issues = []

    recs = pm_result.get("recommendations", []) if isinstance(pm_result, dict) else []
    if not recs:
        issues.append(
            "STEP 4 (PM agent): returned ZERO recommendations — no BUY candidates "
            "can be built, so no entry could happen regardless of regime."
        )
    elif pm_result.get("degraded"):
        issues.append(
            f"STEP 4 (PM agent): fell back to raw research recommendations for "
            f"{len(recs)} tickers — the PM veto layer did not run."
        )

    # Research is chunked (Task 6): one bad chunk only HOLDs its own tickers,
    # but enough failed chunks still degrade the whole run's research quality.
    # Passed in, not read from agents.research_agent.LAST_RESEARCH_STATS: this
    # function judges the run it is handed, and reaching for module state made
    # it judge whichever run happened to have touched that global last — the
    # same anti-pattern already fixed for LAST_BUDGET below.
    research_stats = research_stats or {}
    if research_stats.get("degraded"):
        issues.append(
            f"STEP 1 (Research agent): {research_stats.get('fallback_tickers', 0)}/"
            f"{research_stats.get('tickers', 0)} tickers fell back to HOLD across "
            f"{research_stats.get('chunks_failed', 0)}/{research_stats.get('chunks', 0)} "
            f"failed chunk(s) — research quality for this run is degraded."
        )

    if FORBIDDEN_EVENTS:
        detail = ", ".join(
            f"{e['method']} {e['endpoint']}" + (f" [{e['symbol']}]" if e.get("symbol") else "")
            for e in FORBIDDEN_EVENTS[:5]
        )
        more = f" (+{len(FORBIDDEN_EVENTS) - 5} more)" if len(FORBIDDEN_EVENTS) > 5 else ""
        first_body = FORBIDDEN_EVENTS[0].get("detail", "")
        issues.append(
            f"EXECUTION (Alpaca): {len(FORBIDDEN_EVENTS)} request(s) refused 403 — "
            f"{detail}{more}. Broker said: {first_body}"
        )

    for ft in failed_trims:
        issues.append(
            f"EXECUTION STEP A (drift trim): {ft['ticker']} is at {ft['weight']:.1%} of "
            f"the portfolio vs a {config.MAX_POSITION_PCT:.0%} cap and the trim of "
            f"{ft['shares']} shares FAILED — {ft['error']}"
        )

    # A position without a broker stop is exposed to a gap between runs. This
    # used to halt every new entry (the CRITICAL block); now it is named and
    # sent, and the engine's next pass closes it in software if it has to.
    stop_audit = stop_audit or {}
    _unprotected = list(stop_audit.get("unprotected") or []) + list(stop_audit.get("errors") or [])
    if stop_audit.get("open_orders_unavailable"):
        issues.append(
            "EXECUTION STEP B (broker stop): open-order state was UNKNOWN, so no stop "
            "was placed or replaced this cycle (fail-closed). Every position rides on "
            "the software engine until the next run."
        )
    elif _unprotected:
        detail = ", ".join(
            f"{u['ticker']} ({u.get('reason') or u.get('error')})" for u in _unprotected[:8]
        )
        more = f" (+{len(_unprotected) - 8} more)" if len(_unprotected) > 8 else ""
        issues.append(
            f"EXECUTION STEP B (broker stop): {len(_unprotected)} position(s) have NO "
            f"resting broker stop — {detail}{more}. The software engine still evaluates "
            f"them every run, but a gap between runs is uncovered."
        )

    # Passed in, not read from LAST_BUDGET: this function judges the run it is
    # handed, and reaching for module state made it judge whichever run happened
    # to have touched that global last.
    budget = budget or {}
    if budget.get("degraded"):
        issues.append(
            f"BUDGET: the configured base ({config.BUDGET_BASE}) was unusable — "
            f"{budget.get('reason', 'no reason recorded')}. The cycle sized against "
            f"a fallback budget of ${budget.get('deployable', 0):,.0f}, so the "
            f"account is deploying less than it is configured to."
        )

    # Slot-based sizing (design B): the book is short of its gross-exposure
    # target AND the funnel says a candidate was actually turned away for lack
    # of gross room or budget (GROSS_TARGET_REACHED / NO_BUDGET). Either
    # reason alone is not a fault — a quiet market with nothing worth buying,
    # or a regime multiplier that deliberately shrinks every slot, both leave
    # gross exposure low without rejecting anyone on budget grounds. Only the
    # combination — room existed to trade further AND something was turned
    # away because of it — means the deployment actually fell short.
    funnel_summary = funnel_summary or {}
    _target_gross = budget.get("target_gross_dollars", 0.0)
    _gross_after = budget.get("gross_after", 0.0)
    # NO_BUDGET is recorded QUEUED, not REJECTED (a queued name is deferred to
    # tomorrow, not turned away) — funnel_summary carries it in the separate
    # queued_reasons bucket, so the codes that matter here are a union of both.
    _reason_codes = {r.get("code") for r in funnel_summary.get("reasons", [])}
    _reason_codes |= {r.get("code") for r in funnel_summary.get("queued_reasons", [])}
    _budget_limited = bool(_reason_codes & {"GROSS_TARGET_REACHED", "NO_BUDGET"})
    if _target_gross > 0 and _gross_after < 0.75 * _target_gross and _budget_limited:
        issues.append(
            f"BUDGET: gross exposure ${_gross_after:,.0f} is only "
            f"{_gross_after / _target_gross:.0%} of the ${_target_gross:,.0f} target after "
            f"this cycle, and at least one candidate was rejected for lack of gross room or "
            f"budget — the funnel report names which."
        )

    # A full book is the design working, not a miss: once gross exposure sits
    # at its target, every run has zero orders, a PM BUY that cleared the gate
    # and a GROSS_TARGET_REACHED funnel row — the rule below fired daily on
    # exactly that. Either the funnel says so, or gross_after is within 5% of
    # the target (the candidate may have died at an earlier gate).
    _book_full = ("GROSS_TARGET_REACHED" in _reason_codes
                  or (_target_gross > 0 and _gross_after >= 0.95 * _target_gross))

    strategy_buys = [t for t, v in strategy_signals.items() if v.get("signal") == "BUY"]
    # A frozen run buys nothing by design (config.ENTRIES_FROZEN).
    if (not orders_submitted and not entries_frozen
            and regime_result.get("new_entries_allowed", True)
            and strategy_buys and not _book_full):
        # "The pipeline should have traded and did not" is true but useless on
        # its own: it sent an operator hunting a broker fault when the trades
        # had in fact been eaten by the conviction gate. Name the gate.
        regime = regime_result.get("regime", "BULL")
        gate = config.effective_min_conviction(regime)
        pm_buys = [r for r in recs if r.get("action") == "BUY"]
        cleared = [r for r in pm_buys if (r.get("conviction") or 0) >= gate]
        head = (f"EXECUTION STEP C (entries): zero orders submitted while entries were "
                f"ALLOWED and the strategy agent had {len(strategy_buys)} BUY signal(s) "
                f"({', '.join(strategy_buys[:6])})")
        if not pm_buys:
            issues.append(
                f"{head} — the PM approved no BUY at all, so nothing reached sizing."
            )
        elif not cleared:
            detail = ", ".join(f"{r['ticker']}({r.get('conviction')})" for r in pm_buys[:6])
            issues.append(
                f"{head} — the PM approved {len(pm_buys)} BUY(s) [{detail}] but ALL fell "
                f"below the {regime} conviction gate of >= {gate}, so no candidate reached "
                f"sizing. Either the gate is too high for what research is scoring, or the "
                f"PM is under-rating conviction."
            )
        else:
            issues.append(
                f"{head} and {len(cleared)} PM BUY(s) cleared the >= {gate} conviction gate "
                f"— the pipeline should have traded and did not."
            )

    return issues


def run_reports(paper_trading, orders_submitted, closed_positions, research_results,
                sentiment_results, risk_report, pm_result, strategy_signals,
                regime_result, regime_signals, stop_audit_result, failed_trims=(),
                funnel=None, research_stats=None, entries_frozen=False):
    """Phase 6 — execution alerts, funnel report, persistence."""
    if paper_trading:
        _sep("EXECUTION ALERTS")
        from alerts.telegram_alert import send_execution_alerts
        send_execution_alerts(orders_submitted, closed_positions)
        print(f"  Orders submitted: {len(orders_submitted)}")
        print(f"  Positions closed: {len(closed_positions)}")

    # ── Why every analysed asset ended where it did ──────────────────────
    funnel_summary = {}
    if funnel is not None:
        _sep("EMBUDO DE DECISIÓN")
        funnel_summary = funnel.summary()
        for r in funnel_summary.get("reasons", []):
            print(f"  {r['label']:<38} {r['count']:>4}  "
                  + ", ".join(r["tickers"][:8])
                  + (f" +{len(r['tickers']) - 8}" if len(r["tickers"]) > 8 else ""))
        bottleneck = funnel_summary.get("bottleneck")
        if bottleneck:
            print(f"\n  Cuello de botella: {bottleneck['label']} ({bottleneck['count']})")
        try:
            from db.database import save_funnel_log
            save_funnel_log(funnel.run_timestamp, funnel.rows())
        except Exception as _fe:
            print(f"  [Funnel] Could not persist decision log: {_fe}")
        try:
            from alerts.telegram_alert import send_funnel_report
            send_funnel_report(funnel_summary,
                               regime=regime_result.get("regime", ""),
                               budget=LAST_BUDGET or None)
        except Exception as _fe:
            print(f"  [Funnel] Could not send report: {_fe}")

    _sep("SAVING RESULTS")
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "paper_trading": paper_trading,
        "research": research_results,
        "sentiment": sentiment_results,
        "risk": risk_report,
        "pm": pm_result,
        "strategy": {t: v for t, v in strategy_signals.items()},
        "regime": regime_result,
        "regime_signals": regime_signals,
        "stop_audit": stop_audit_result,
        "orders_submitted": orders_submitted,
        "closed_positions": closed_positions,
        "funnel": funnel_summary,
        "budget": dict(LAST_BUDGET),
    }
    # Persist to SQLite (key metrics + full JSON blob); JSON file remains the backup
    save_run(run_data)
    save_signals(datetime.now().strftime("%Y-%m-%d"), research_results)

    # Backfill conviction_log.return_30d/return_60d for rows old enough to
    # score. Skipped under DRY_RUN — nothing here touches the broker, but a
    # dry run's whole point is "nothing gets written", and this writes to
    # conviction_log same as any other persisted state.
    if not DRY_RUN:
        try:
            from reports.conviction_backfill import backfill_conviction_returns
            _n_backfilled = backfill_conviction_returns()
            if _n_backfilled:
                print(f"  [Conviction] Backfilled {_n_backfilled} row(s) with forward returns")
        except Exception as _backfill_e:
            print(f"  [Conviction] Backfill failed: {_backfill_e}")

    out_path = os.path.join(LOGS_DIR, "last_run.json")
    with open(out_path, "w") as f:
        json.dump(run_data, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")

    runs_dir = os.path.join(LOGS_DIR, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    archive_path = os.path.join(runs_dir, f"{ts}.json")
    shutil.copy2(out_path, archive_path)
    print(f"  Run archived to {archive_path}")

    try:
        from agents.llm_usage import run_summary
        _u = run_summary()
        print(f"  [LLM] {_u['calls']} calls | {_u['input']:,} in / {_u['output']:,} out tokens "
              f"| cache_read {_u['cache_read']:,} | est. cost ${_u['cost_usd']:.4f}")
    except Exception as _usage_err:
        print(f"  [LLM] usage summary unavailable: {_usage_err}")

    # Health check LAST: a run that completes without trading, without PM output
    # or with 403s on every write must not look identical to a healthy one.
    try:
        _issues = _collect_health_issues(pm_result, orders_submitted, strategy_signals,
                                         regime_result, list(failed_trims),
                                         budget=dict(LAST_BUDGET),
                                         stop_audit=stop_audit_result,
                                         funnel_summary=funnel_summary,
                                         research_stats=research_stats,
                                         entries_frozen=entries_frozen)
        if _issues:
            _sep("HEALTH ALERT")
            for _i in _issues:
                print(f"  [HEALTH] {_i}")
            from alerts.telegram_alert import send_health_alert
            _sent = send_health_alert(_issues)
            print(f"  Health alert: {'sent' if _sent else 'NOT sent (check Telegram config)'}")
    except Exception as _health_err:
        print(f"  [HEALTH] Health check itself failed: {_health_err}")

    print("\n  Run complete.")


def _run():
    from utils.funnel import FunnelRecorder

    run_timestamp = datetime.now().isoformat()
    tickers, research_results, sentiment_results, blended_sentiment = run_signals()
    # Captured right here, not read later inside _collect_health_issues: by
    # the time run_reports() (and its health check) run, another cycle could
    # in principle have touched agents.research_agent.LAST_RESEARCH_STATS —
    # same reasoning as LAST_BUDGET.
    import agents.research_agent as _research_agent
    research_stats = dict(_research_agent.LAST_RESEARCH_STATS)
    funnel = FunnelRecorder(run_timestamp, tickers)
    funnel.stage_count("universe", len(tickers))
    (risk_report, regime_result, regime_signals, strategy_signals,
     strategy_score_map, research_results) = run_gates(tickers, research_results, blended_sentiment)
    pm_result = run_decisions(tickers, research_results, sentiment_results, risk_report,
                              regime_result, regime_signals, strategy_signals)

    _sep("STEP 5: TELEGRAM ALERT")
    from alerts.telegram_alert import send_daily_summary
    print("  Sending daily summary to Telegram...")
    # CURRENT_POSITIONS is the book as of import: a PM SELL on one of these
    # names is a veto run_exits will act on, so the summary keeps its rationale.
    ok = send_daily_summary(research_results, sentiment_results, pm_result,
                            held_tickers=set(CURRENT_POSITIONS))
    print(f"  Telegram: {'sent' if ok else 'skipped/failed'}")

    orders_submitted = []
    closed_positions = []
    failed_trims = []        # over-cap positions whose trim did not go through
    stop_audit_result = {}   # broker stop audit; written to run_data, read by the health check
    entries_frozen = bool(config.ENTRIES_FROZEN)

    if PAPER_TRADING:
        (equity, positions, exited_tickers, closed_positions,
         stop_audit_result, trimmed_tickers, failed_trims) = run_exits(
            pm_result=pm_result, strategy_score_map=strategy_score_map)

        if entries_frozen:
            _sep("EXECUTION STEP C: ENTRIES FROZEN")
            print("  config.ENTRIES_FROZEN — no new buys until the first monthly rebalance.")
            _persist_holding_days(positions, exited_tickers, [])
        else:
            # stop_audit_result is mutated in place: the post-entry stop pass
            # merges the rows for today's fills into the morning audit's.
            orders_submitted, closed_positions = run_entries(
                equity, positions, exited_tickers, trimmed_tickers, closed_positions,
                orders_submitted, pm_result, sentiment_results,
                research_results, strategy_signals, regime_signals, regime_result,
                strategy_score_map, funnel=funnel, stop_audit_result=stop_audit_result)

    run_reports(PAPER_TRADING, orders_submitted, closed_positions, research_results,
                sentiment_results, risk_report, pm_result, strategy_signals,
                regime_result, regime_signals, stop_audit_result, failed_trims,
                funnel=funnel, research_stats=research_stats,
                entries_frozen=entries_frozen)


if __name__ == "__main__":
    main()
