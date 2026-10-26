"""
AI Hedge Fund — Operational Dashboard

Run:
    streamlit run dashboard/app.py

Reads the live system's JSON state under logs/ (position_metadata.json,
regime_state.json, last_run.json, runs/*.json, backtest_results.json) and
overlays live prices from yfinance. Set HF_LOGS_DIR to point at a logs/
directory other than <repo>/logs (useful when running from a git worktree).
"""
import json
import os
import subprocess
import sys
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
LOGS = Path(os.environ.get("HF_LOGS_DIR", ROOT / "logs"))

# ── Loaders (JSON — this system has no SQLite DB) ─────────────────────────────


def _load_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


@st.cache_data(ttl=300)
def load_positions() -> dict:
    return _load_json(LOGS / "position_metadata.json")


@st.cache_data(ttl=300)
def load_position_state() -> dict:
    return _load_json(LOGS / "position_state.json")


@st.cache_data(ttl=300)
def load_regime() -> dict:
    return _load_json(LOGS / "regime_state.json")


@st.cache_data(ttl=300)
def load_last_run() -> dict:
    return _load_json(LOGS / "last_run.json")


@st.cache_data(ttl=300)
def load_backtest() -> dict:
    return _load_json(LOGS / "backtest_results.json")


@st.cache_data(ttl=300)
def load_runs_history() -> list[dict]:
    """Per-run summaries from logs/runs/*.json (replaces the missing `runs` table)."""
    runs_dir = LOGS / "runs"
    rows: list[dict] = []
    if not runs_dir.exists():
        return rows
    for path in sorted(runs_dir.glob("*.json"), reverse=True):
        d = _load_json(path)
        if not d:
            continue
        risk = d.get("risk", {}) or {}
        regime = d.get("regime", {}) or {}
        ts = str(d.get("timestamp", path.stem))
        rows.append({
            "Run": ts[:16].replace("T", " "),
            "Regime": regime.get("regime", "—"),
            "Opened": len(d.get("orders_submitted", []) or []),
            "Closed": len(d.get("closed_positions", []) or []),
            "Sharpe": risk.get("sharpe_ratio"),
            "Status": risk.get("status", "—"),
        })
    return rows


@st.cache_data(ttl=300)
def load_run_file(path_str: str) -> dict:
    return _load_json(Path(path_str))


def available_runs() -> list[tuple[str, str | None]]:
    """Selector options: ('Latest …', None) then each logs/runs/*.json newest-first."""
    opts: list[tuple[str, str | None]] = [("Latest (last_run.json)", None)]
    runs_dir = LOGS / "runs"
    if runs_dir.exists():
        for path in sorted(runs_dir.glob("*.json"), reverse=True):
            d = _load_json(path)
            label = str(d.get("timestamp", path.stem))[:16].replace("T", " ")
            opts.append((label, str(path)))
    return opts


# ── Price helpers (yfinance) ──────────────────────────────────────────────────


@st.cache_data(ttl=60)
def get_current_prices(tickers: tuple) -> dict[str, float]:
    tickers = list(tickers)
    if not tickers:
        return {}
    try:
        data = yf.download(tickers, period="2d", auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        out = {}
        for t in tickers:
            if t in data.columns:
                s = data[t].dropna()
                if not s.empty:
                    out[t] = float(s.iloc[-1])
        return out
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_price_history(ticker: str, start: str | None = None, period: str | None = None) -> pd.Series:
    """Adjusted Close series for a single ticker."""
    try:
        if period:
            data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        else:
            data = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()
    except Exception:
        return pd.Series(dtype=float)


def _spy_normalized(start: str) -> pd.DataFrame:
    spy = get_price_history("SPY", start=start)
    if spy.empty:
        return pd.DataFrame()
    norm = spy / spy.iloc[0] * 100
    return norm.reset_index().set_axis(["Date", "SPY"], axis=1)


# ── Derived data ──────────────────────────────────────────────────────────────


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s)).date()  # handles full ISO timestamps
    except Exception:
        try:
            return date.fromisoformat(str(s)[:10])
        except Exception:
            return None


def position_pnl_pct(meta: dict, price_now: float) -> float:
    entry = meta.get("entry_price", 0) or 0
    if entry <= 0 or not price_now:
        return 0.0
    return (price_now / entry - 1) * 100


def book_unrealized_pct(positions: dict, prices: dict) -> float:
    """Size-weighted unrealized P&L of the book, in %. size_pct is a percent (e.g. 15.0)."""
    total = 0.0
    for ticker, meta in positions.items():
        entry = meta.get("entry_price", 0) or 0
        weight = (meta.get("size_pct", 0) or 0) / 100.0
        price_now = prices.get(ticker)
        if entry > 0 and price_now:
            total += weight * (price_now / entry - 1) * 100
    return total


def build_portfolio_equity_curve(positions: dict) -> pd.DataFrame:
    """Live book reconstructed from entries: level(t) = 100 * (1 + Σ w_i·(P_t/P_entry − 1))."""
    series = []
    for ticker, meta in positions.items():
        entry = meta.get("entry_price", 0) or 0
        weight = (meta.get("size_pct", 0) or 0) / 100.0
        if entry <= 0:
            continue
        start = _parse_date(meta.get("entry_date"))
        hist = get_price_history(ticker, start=start.isoformat() if start else None)
        if hist.empty:
            continue
        series.append((weight * (hist / entry - 1.0)).rename(ticker))
    if not series:
        return pd.DataFrame()
    combined = pd.concat(series, axis=1).sort_index()
    combined = combined.ffill().fillna(0.0)  # carry over non-trading days; 0 before entry
    level = (1.0 + combined.sum(axis=1)) * 100.0
    return level.reset_index().set_axis(["Date", "Portfolio"], axis=1)


def backtest_curve_df(backtest: dict) -> pd.DataFrame:
    wv = backtest.get("weekly_values", []) if backtest else []
    if not wv:
        return pd.DataFrame()
    df = pd.DataFrame(wv)
    df["Date"] = pd.to_datetime(df["date"])
    df["Portfolio"] = df["value"] / df["value"].iloc[0] * 100
    return df[["Date", "Portfolio"]]


# ── Agents tab helpers ────────────────────────────────────────────────────────


def _fmt_pct(v) -> str:
    """Format a fraction/ratio as a percent (0.05 -> '5.00%'). Non-numbers -> '—'."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "—"
    return f"{v:.2%}"


def _fmt_money(v) -> str:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "—"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def _fmt_num(v, nd: int = 2) -> str:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "—"
    return f"{v:.{nd}f}"


def classify_run_output(returncode: int, stdout: str, stderr: str) -> dict:
    """Interpret a pipeline run's exit code + output into a UX status dict.

    '[LOCK]' in stdout => a run was already in progress (friendly no-op, main.py exits 0).
    401 / auth markers => the known-invalid ANTHROPIC_API_KEY (surface even on exit 0).
    """
    out = stdout or ""
    err = stderr or ""
    if "[LOCK]" in out:
        return {"ok": True, "locked": True, "auth_error": False,
                "message": "A pipeline run is already in progress — nothing was started."}
    blob = (out + "\n" + err).lower()
    if any(k in blob for k in ("401", "authentication_error", "invalid x-api-key", "invalid api key")):
        return {"ok": False, "locked": False, "auth_error": True,
                "message": ("Authentication error: the ANTHROPIC_API_KEY in .env is invalid. "
                            "Replace it from console.anthropic.com.")}
    if returncode == 0:
        return {"ok": True, "locked": False, "auth_error": False,
                "message": "Pipeline finished successfully."}
    return {"ok": False, "locked": False, "auth_error": False,
            "message": f"Pipeline exited with code {returncode}. See output below."}


def run_pipeline(dry_run: bool, runner=subprocess.run) -> dict:
    """Shell out to main.py. dry_run adds --dry-run (gates order execution).

    `runner` is injectable for testing; defaults to subprocess.run. main.py self-guards
    with an fcntl lock and exits 0 if a run is already in progress, so this is safe to
    fire even while the scheduler is running.
    """
    cmd = [sys.executable, str(ROOT / "main.py")]
    if dry_run:
        cmd.append("--dry-run")
    proc = runner(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    return {
        "returncode": getattr(proc, "returncode", 1),
        "stdout": getattr(proc, "stdout", "") or "",
        "stderr": getattr(proc, "stderr", "") or "",
    }


def render_agents_tab():
    st.subheader("🤖 Agents — what they did and found")

    # Run controls
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            dry = st.checkbox("Dry run (no orders)", value=True,
                              help="Analyzes and writes findings but submits no paper orders.")
            go = st.button("▶ Run pipeline", type="primary")
        with c2:
            lr = load_last_run()
            st.caption(f"Paper trading: {lr.get('paper_trading', '—')} · last run: {lr.get('timestamp', '—')}")
            st.caption("Live runs call the Claude agents and fail if ANTHROPIC_API_KEY is invalid.")
        if go:
            with st.spinner("Running pipeline… agents call the Claude API; this can take a minute."):
                result = run_pipeline(dry)
            status = classify_run_output(result["returncode"], result["stdout"], result["stderr"])
            banner = (st.info if status["locked"]
                      else st.error if status["auth_error"]
                      else st.success if status["ok"]
                      else st.warning)
            banner(status["message"])
            with st.expander("Pipeline output", expanded=not status["ok"]):
                if result["stdout"]:
                    st.code(result["stdout"][-6000:])
                if result["stderr"]:
                    st.code(result["stderr"][-4000:])
            if status["ok"] and not status["locked"]:
                st.cache_data.clear()
                st.rerun()

    # Run selector
    runs = available_runs()
    choice = st.selectbox("View findings from run", runs, format_func=lambda o: o[0])
    run_data = load_last_run() if choice[1] is None else load_run_file(choice[1])
    if not run_data:
        st.info("No run data yet. Run the pipeline to populate.")
        return
    st.caption(
        f"Run {run_data.get('timestamp', '—')} · "
        f"paper_trading={run_data.get('paper_trading', '—')} · "
        f"regime={(run_data.get('regime') or {}).get('regime', '—')}"
    )

    # Research
    research = run_data.get("research") or []
    with st.expander(f"🔬 Research (Claude Sonnet) — {len(research)} tickers", expanded=True):
        if not research:
            st.caption("No research in this run.")
        for r in research:
            st.markdown(f"**{r.get('ticker', '—')}** · {r.get('recommendation', '—')} · "
                        f"{'⭐' * int(r.get('conviction', 0) or 0)}")
            if r.get("thesis"):
                st.write(r["thesis"])
            risks = r.get("key_risks") or []
            if risks:
                st.markdown("_Key risks:_ " + "; ".join(str(x) for x in risks))
            levels = {"entry": r.get("entry_ref"), "stop": r.get("sl_price")}
            shown = " · ".join(f"{k} {v}" for k, v in levels.items() if v is not None)
            if shown:
                st.caption(shown)
            st.divider()

    # Sentiment
    sentiment = run_data.get("sentiment") or []
    with st.expander(f"📰 Sentiment (Claude Haiku) — {len(sentiment)} tickers"):
        if not sentiment:
            st.caption("No sentiment in this run.")
        for s in sentiment:
            st.markdown(f"**{s.get('ticker', '—')}** · {s.get('sentiment', '—')} "
                        f"({_fmt_num(s.get('sentiment_score'))})")
            if s.get("summary"):
                st.caption(s["summary"])

    # Portfolio Manager
    pm = run_data.get("pm") or {}
    recs = pm.get("recommendations") or []
    with st.expander(f"🧠 Portfolio Manager — {len(recs)} recommendations"):
        if recs:
            st.dataframe(pd.DataFrame([{
                "Ticker": x.get("ticker"), "Action": x.get("action"),
                "Size %": x.get("size_pct"), "Conviction": x.get("conviction"),
                "Rationale": x.get("rationale"),
            } for x in recs]), hide_index=True, use_container_width=True)
        if pm.get("portfolio_comment"):
            st.markdown("**Portfolio comment**")
            st.write(pm["portfolio_comment"])
        if not recs and not pm.get("portfolio_comment"):
            st.caption("No PM output in this run.")

    # Risk
    risk = run_data.get("risk") or {}
    with st.expander("🛡️ Risk"):
        if not risk:
            st.caption("No risk output in this run.")
        else:
            a, b, c = st.columns(3)
            a.metric("VaR 95%", _fmt_pct(risk.get("var_95")))
            a.metric("Max Drawdown", _fmt_pct(risk.get("max_drawdown")))
            b.metric("Sharpe", _fmt_num(risk.get("sharpe_ratio")))
            b.metric("Avg Correlation", _fmt_num(risk.get("avg_correlation")))
            c.metric("Total Exposure", _fmt_pct(risk.get("total_exposure")))
            c.metric("Status", risk.get("status", "—"))
            violations = risk.get("violations") or []
            if violations:
                st.markdown("**Violations**")
                for v in violations:
                    st.markdown(f"- {v}")
            else:
                st.success("No risk violations.")

    # Broker stop audit (one GTC stop per position at the engine's trail)
    audit = run_data.get("stop_audit") or {}
    audit_rows = [
        {"Ticker": x.get("ticker"), "Outcome": bucket, "Stop": x.get("stop"),
         "Qty": x.get("qty"), "Order": x.get("order_id"),
         "Note": x.get("reason") or x.get("error") or x.get("via") or ""}
        for bucket in ("protected", "placed", "replaced", "unprotected", "errors")
        for x in (audit.get(bucket) or [])
    ]
    with st.expander(f"🎯 Broker stop audit — {len(audit_rows)} positions"):
        if audit_rows:
            st.dataframe(pd.DataFrame(audit_rows), hide_index=True, use_container_width=True)
            a, b, c = st.columns(3)
            a.metric("Protected", len(audit.get("protected") or []))
            b.metric("Placed / replaced",
                     len(audit.get("placed") or []) + len(audit.get("replaced") or []))
            c.metric("Unprotected", len(audit.get("unprotected") or []) + len(audit.get("errors") or []))
        elif audit.get("open_orders_unavailable"):
            st.warning("Open-order state was unknown — the audit failed closed this run.")
        else:
            st.caption("No stop audit in this run.")


# ── Page ──────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="AI Hedge Fund",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("📈 AI Hedge Fund — Dashboard")
    st.caption(f"Last refreshed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  ·  logs: {LOGS}")

    positions = load_positions()
    pos_state = load_position_state()
    regime = load_regime()
    last_run = load_last_run()
    tickers = list(positions.keys())
    prices = get_current_prices(tuple(tickers))

    # ── KPI bar ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Open Positions", len(positions))
    col2.metric("Unrealized P&L", f"{book_unrealized_pct(positions, prices):+.2f}%")
    col3.metric("Regime", regime.get("regime", "—"), delta=f"ADX {regime.get('adx', 0):.1f}")
    col4.metric("VIX", f"{regime.get('vix', 0):.1f}", delta=regime.get("vol_regime", "—"))
    col5.metric("Size Multiplier", f"{regime.get('size_multiplier', 1.0):.2f}x")

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "💼 Positions", "🔍 Signals", "🌡️ Regime", "🤖 Agents"])

    # ── TAB 1: Overview ──────────────────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            backtest = load_backtest()
            bt_df = backtest_curve_df(backtest)

            st.subheader("Backtest — Portfolio vs SPY")
            if bt_df.empty:
                st.info("No backtest_results.json found.")
            else:
                spy_bt = _spy_normalized(bt_df["Date"].iloc[0].date().isoformat())
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=bt_df["Date"], y=bt_df["Portfolio"],
                    name="Portfolio", line=dict(color="#00CC96", width=2)))
                if not spy_bt.empty:
                    fig.add_trace(go.Scatter(
                        x=spy_bt["Date"], y=spy_bt["SPY"],
                        name="SPY", line=dict(color="#636EFA", width=1, dash="dash")))
                fig.update_layout(
                    yaxis_title="Normalized (base=100)", xaxis_title="",
                    legend=dict(x=0, y=1), height=300, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"Period {backtest.get('period', '—')} · "
                    f"Total {backtest.get('total_return', 0):+.2%} · "
                    f"SPY {backtest.get('spy_total_return', 0):+.2%} · "
                    f"Alpha {backtest.get('alpha', 0):+.2%} · "
                    f"Sharpe {backtest.get('sharpe_ratio', 0):.2f}")

            st.subheader("Live Book vs SPY (since first entry)")
            live_df = build_portfolio_equity_curve(positions)
            if live_df.empty:
                st.info("No open positions to reconstruct a live curve.")
            else:
                spy_live = _spy_normalized(live_df["Date"].iloc[0].date().isoformat())
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=live_df["Date"], y=live_df["Portfolio"],
                    name="Book", line=dict(color="#00CC96", width=2)))
                if not spy_live.empty:
                    fig2.add_trace(go.Scatter(
                        x=spy_live["Date"], y=spy_live["SPY"],
                        name="SPY", line=dict(color="#636EFA", width=1, dash="dash")))
                fig2.update_layout(
                    yaxis_title="Normalized (base=100)", xaxis_title="",
                    legend=dict(x=0, y=1), height=300, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig2, use_container_width=True)

        with col_right:
            st.subheader("Run History")
            runs = load_runs_history()
            if runs:
                st.dataframe(pd.DataFrame(runs).head(12), hide_index=True, use_container_width=True)
            else:
                st.info("No runs/ history found.")

    # ── TAB 2: Positions ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("Open Positions")
        if not positions:
            st.info("No open positions.")
        else:
            rows = []
            pnl_values = []
            for ticker, meta in positions.items():
                price_now = prices.get(ticker, 0)
                entry = meta.get("entry_price", 0) or 0
                pnl_pct = position_pnl_pct(meta, price_now)
                pnl_values.append(pnl_pct)
                days_held = meta.get("holding_days")
                if days_held is None:
                    d = _parse_date(meta.get("entry_date"))
                    days_held = (date.today() - d).days if d else 0
                state = pos_state.get(ticker, {}) or {}
                stop = state.get("trail_stop") or state.get("sl_price")
                initial = state.get("initial_stop")
                rows.append({
                    "Ticker": ticker,
                    "Entry": entry,
                    "Current": price_now if price_now else float("nan"),
                    "P&L %": pnl_pct,
                    "Size": f"{meta.get('size_pct', 0):.0f}%",
                    "Conviction": "⭐" * int(meta.get("conviction", 3) or 0),
                    "Days Held": days_held,
                    "Decay": state.get("decay_count", 0) or 0,
                    "Initial Stop": initial if initial else float("nan"),
                    "Trail Stop": stop if stop else float("nan"),
                    "Protected": "✅" if state.get("broker_stop_order_id") else ("🟡" if stop else "🚫"),
                })

            df = pd.DataFrame(rows)
            styler = (
                df.style
                .format({"Entry": "${:,.2f}", "Current": "${:,.2f}", "P&L %": "{:+.2f}%", "Stop": "${:,.2f}"})
                .map(lambda v: "color: #00CC96" if v >= 0 else "color: #EF553B", subset=["P&L %"])
            )
            st.dataframe(styler, use_container_width=True, hide_index=True)

            colors = ["#00CC96" if p >= 0 else "#EF553B" for p in pnl_values]
            fig_pnl = go.Figure(go.Bar(
                x=[r["Ticker"] for r in rows], y=pnl_values,
                marker_color=colors, text=[f"{p:+.1f}%" for p in pnl_values], textposition="auto"))
            fig_pnl.update_layout(
                title="Unrealized P&L by Position", yaxis_title="P&L %",
                height=280, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_pnl, use_container_width=True)

    # ── TAB 3: Signals ───────────────────────────────────────────────────────
    with tab3:
        st.subheader("Latest Signals")
        if not last_run:
            st.info("No last_run.json found. Run main.py to populate.")
        else:
            st.caption(f"From run: {last_run.get('timestamp', '—')}")
            research = {r["ticker"]: r for r in last_run.get("research", [])}
            sentiment = {s["ticker"]: s for s in last_run.get("sentiment", [])}
            pm = {r["ticker"]: r for r in last_run.get("pm", {}).get("recommendations", [])}
            strategy = last_run.get("strategy", {}) or {}
            reg_sig = last_run.get("regime_signals", {}) or {}
            all_tickers = sorted(set(research) | set(strategy) | set(pm) | set(reg_sig))

            sig_rows = []
            for t in all_tickers:
                r = research.get(t, {})
                s = sentiment.get(t, {})
                p = pm.get(t, {})
                strat = strategy.get(t, {})
                rs = reg_sig.get(t, {})
                sig_rows.append({
                    "Ticker": t,
                    "Research": r.get("recommendation", "—"),
                    "Conv": r.get("conviction"),
                    "Sentiment": s.get("sentiment", "—"),
                    "Sent Score": s.get("sentiment_score"),
                    "Strategy": strat.get("signal", "—"),
                    "Score": strat.get("composite_score"),
                    "Regime Sig": rs.get("signal", "—"),
                    "Rank": rs.get("rank"),
                    "PM": p.get("action", "—"),
                })
            sig_df = pd.DataFrame(sig_rows)

            def color_rec(val):
                if val == "BUY":
                    return "background-color: #1a472a; color: #00CC96"
                if val == "SELL":
                    return "background-color: #4a1a1a; color: #EF553B"
                return ""

            st.dataframe(
                sig_df.style
                .format({"Sent Score": "{:+.2f}", "Score": "{:+.2f}"}, na_rep="—")
                .map(color_rec, subset=["Research", "PM"]),
                use_container_width=True, hide_index=True,
            )

        st.subheader("Conviction vs Unrealized P&L (open book)")
        conv_rows = []
        for ticker, meta in positions.items():
            price_now = prices.get(ticker)
            if not price_now:
                continue
            conv_rows.append({
                "ticker": ticker,
                "conviction": meta.get("conviction", 0),
                "pnl_pct": position_pnl_pct(meta, price_now),
            })
        if conv_rows:
            cdf = pd.DataFrame(conv_rows)
            fig_conv = px.scatter(
                cdf, x="conviction", y="pnl_pct", color="ticker", text="ticker",
                labels={"conviction": "Conviction (1-5)", "pnl_pct": "Unrealized P&L %"},
            )
            fig_conv.update_traces(textposition="top center", marker=dict(size=12))
            fig_conv.update_xaxes(dtick=1, range=[0.5, 5.5])
            fig_conv.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_conv.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig_conv, use_container_width=True)
        else:
            st.info("No priced open positions to plot.")

    # ── TAB 4: Regime ────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Market Regime — Current State")
        if not regime:
            st.warning("No regime_state.json found.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                emoji = {"BULL": "🟢", "BEAR": "🔴"}.get(regime.get("regime", ""), "⚪")
                st.metric("Regime", f"{emoji} {regime.get('regime', '—')}")
                st.metric("Breadth", f"{regime.get('breadth_pct', 1.0):.0%}")
                st.metric("ADX(14)", f"{regime.get('adx', 0):.2f} (obs. only)")
                st.metric("New Entries", "✅ Yes" if regime.get("new_entries_allowed") else "🚫 No")
            with c2:
                st.metric("SPY Price", f"${regime.get('spy_price', 0):,.2f}")
                st.metric("SPY EMA200", f"${regime.get('spy_ema200', 0):,.2f}")
                st.metric("Buffer %", f"{regime.get('buffer_pct', 0) * 100:.2f}%")
                st.metric("Days in Regime", regime.get("days_in_regime", 0))
            with c3:
                st.metric("VIX / Vol", f"{regime.get('vix', 0):.1f}", delta=regime.get("vol_regime", "—"))
                st.metric("Size Multiplier", f"{regime.get('size_multiplier', 1.0):.2f}x")
                st.metric("Credit Signal", regime.get("credit_signal", "—"))
                st.metric("Switches YTD", regime.get("switch_count_ytd", 0))

            st.subheader("SPY — Price vs EMA200 (1Y)")
            spy = get_price_history("SPY", period="1y")
            if spy.empty:
                st.warning("Could not load SPY history.")
            else:
                ema200 = spy.ewm(span=200, adjust=False).mean()
                sma20 = spy.rolling(20).mean()
                fig_spy = go.Figure()
                fig_spy.add_trace(go.Scatter(x=spy.index, y=spy, name="SPY", line=dict(color="#636EFA", width=2)))
                fig_spy.add_trace(go.Scatter(x=ema200.index, y=ema200, name="EMA200", line=dict(color="#EF553B", width=1)))
                fig_spy.add_trace(go.Scatter(x=sma20.index, y=sma20, name="SMA20", line=dict(color="#FFA15A", width=1, dash="dot")))
                fig_spy.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend=dict(x=0, y=1))
                st.plotly_chart(fig_spy, use_container_width=True)

    # ── TAB 5: Agents ────────────────────────────────────────────────────────
    with tab5:
        render_agents_tab()


if __name__ == "__main__":
    main()
