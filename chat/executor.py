"""
chat/executor.py — Agent dispatcher and response formatter for the AI hedge fund terminal.

Public API:
    execute_routing(routing: dict, ctx: dict, force_fresh: bool = False) -> tuple[str, dict | None]
"""

import os
from datetime import datetime

from config import TICKERS as ALL_TICKERS


# ---------------------------------------------------------------------------
# Helper: extract positions data for the risk agent and the stop audit
# ---------------------------------------------------------------------------

def _positions_for_risk(ctx: dict) -> dict:
    equity = ctx.get("account", {}).get("equity", 0)
    if equity <= 0:
        return {}
    return {
        p["symbol"]: p["market_value"] / equity
        for p in ctx.get("live_positions", [])
        if p.get("market_value", 0) > 0
    }


def _positions_for_stop_audit(ctx: dict) -> list:
    """ctx live_positions as the broker-shaped rows run_stop_audit reads."""
    return [
        {
            "symbol": str(p["symbol"]).replace("-", "."),
            "current_price": p["current_price"],
            "qty": p["qty"],
            "qty_available": p.get("qty_available", p["qty"]),
            "market_value": p["market_value"],
        }
        for p in ctx.get("live_positions", [])
    ]


# ---------------------------------------------------------------------------
# Formatter helpers
# ---------------------------------------------------------------------------

def _format_portfolio_summary(ctx: dict, lang: str = "es") -> str:
    account = ctx.get("account", {})
    portfolio_state = ctx.get("portfolio_state", {})

    equity = account.get("equity", portfolio_state.get("equity_real", 0))
    cash = account.get("cash", portfolio_state.get("cash", 0))
    regime = portfolio_state.get("regime", "UNKNOWN")

    lines = []
    lines.append(
        f"Equity: ${equity:,.2f}  |  Cash: ${cash:,.2f}  |  Régimen: {regime}"
    )
    lines.append("")

    positions = ctx.get("live_positions", [])
    open_pos_label = "Open positions" if lang == "en" else "Posiciones abiertas"
    lines.append(f"{open_pos_label} ({len(positions)}):")
    if positions:
        for p in positions:
            sym = p.get("symbol", "?")
            qty = p.get("qty", 0)
            price = p.get("current_price", 0)
            plpc = p.get("unrealized_plpc", 0)
            sign = "+" if plpc >= 0 else ""
            lines.append(f"  {sym:<8} qty={qty}  precio=${price:.2f}  {sign}{plpc*100:.2f}%")
    else:
        no_pos_label = "no positions" if lang == "en" else "sin posiciones"
        lines.append(f"  ({no_pos_label})")

    orders = ctx.get("open_orders", [])
    open_ord_label = "Open orders" if lang == "en" else "Órdenes abiertas"
    lines.append("")
    lines.append(f"{open_ord_label} ({len(orders)}):")
    if orders:
        for o in orders:
            sym = o.get("symbol", "?")
            side = o.get("side", "?")
            qty = o.get("qty", 0)
            status = o.get("status", "?")
            lines.append(f"  {sym:<8} {side}  qty={qty}  estado={status}")
    else:
        lines.append("  (sin órdenes)")

    # Pending next session from portfolio_state or position_state
    pending = portfolio_state.get("pending_next_session", [])
    if not pending:
        pending = ctx.get("position_state", {}).get("pending_next_session", [])
    if pending:
        pending_label = "Pending next session" if lang == "en" else "Pendientes próxima sesión"
        lines.append("")
        lines.append(f"{pending_label}: {', '.join(pending)}")

    return "\n".join(lines)


def _format_research(results: list, lang: str) -> str:
    if not results:
        return "No hay resultados de research disponibles." if lang == "es" else "No research results available."

    header = "Análisis fundamental:" if lang == "es" else "Fundamental Analysis:"
    lines = [header, ""]

    for r in results:
        ticker = r.get("ticker", "?")
        rec = r.get("recommendation", "?")
        conviction = r.get("conviction", "?")
        thesis = r.get("thesis", "N/A")
        risks = r.get("key_risks", [])
        entry_ref = r.get("entry_ref")
        sl_price = r.get("sl_price")
        tp3_price = r.get("tp3_price")

        lines.append(f"{'='*40}")
        lines.append(f"{ticker}  →  {rec}  (convicción: {conviction}/5)")
        lines.append(f"  Tesis: {thesis}")
        if risks:
            lines.append(f"  Riesgos: {', '.join(risks)}")
        if entry_ref is not None:
            lines.append(f"  Precio ref: ${entry_ref:.2f}")
        if sl_price is not None:
            lines.append(f"  Stop-loss: ${sl_price:.2f}")
        if tp3_price is not None:
            lines.append(f"  Take-profit (TP3): ${tp3_price:.2f}")
        lines.append("")

    return "\n".join(lines)


def _format_sentiment(results: list, lang: str) -> str:
    if not results:
        return "No hay resultados de sentimiento disponibles." if lang == "es" else "No sentiment results available."

    header = "Análisis de sentimiento:" if lang == "es" else "Sentiment Analysis:"
    lines = [header, ""]

    for r in results:
        ticker = r.get("ticker", "?")
        sentiment = r.get("sentiment", "?")
        score = r.get("sentiment_score", 0.0)
        summary = r.get("summary", "N/A")

        sign = "+" if score >= 0 else ""
        lines.append(f"{ticker:<8}  {sentiment:<8}  score={sign}{score:.2f}")
        lines.append(f"  {summary}")
        lines.append("")

    return "\n".join(lines)


def _format_strategy(results: dict, lang: str) -> str:
    if not results:
        return "No hay señales de estrategia disponibles." if lang == "es" else "No strategy signals available."

    header = "Señales de estrategia:" if lang == "es" else "Strategy Signals:"
    lines = [header, ""]

    for ticker, r in results.items():
        signal = r.get("signal", "?")
        composite = r.get("composite_score", 0.0)
        mom_z = r.get("mom_z", 0.0)
        val_z = r.get("val_z", 0.0)

        lines.append(
            f"{ticker:<8}  {signal:<5}  composite={composite:+.4f}  "
            f"mom_z={mom_z:+.4f}  val_z={val_z:+.4f}"
        )

    return "\n".join(lines)


def _format_regime(result: dict, lang: str) -> str:
    if not result:
        return "No hay datos de régimen disponibles." if lang == "es" else "No regime data available."

    header = "Régimen de mercado:" if lang == "es" else "Market Regime:"
    lines = [header, ""]

    regime = result.get("regime", "?")
    spy_price = result.get("spy_price", 0.0)
    spy_ema200 = result.get("spy_ema200", 0.0)
    vix = result.get("vix", 0.0)
    vol_regime = result.get("vol_regime", "?")
    size_mult = result.get("size_multiplier", 1.0)
    days_in = result.get("days_in_regime", 0)
    confirmed = result.get("confirmed", False)

    lines.append(f"  Régimen:         {regime}")
    lines.append(f"  SPY precio:      ${spy_price:.2f}  |  EMA200: ${spy_ema200:.2f}")
    lines.append(f"  VIX:             {vix:.2f}  |  Vol régimen: {vol_regime}")
    lines.append(f"  Mult. tamaño:    {size_mult:.2f}x")
    lines.append(f"  Días en régimen: {days_in}")
    lines.append(f"  Confirmado:      {'Sí' if confirmed else 'No'}")

    return "\n".join(lines)


def _format_risk(result: dict, lang: str) -> str:
    if not result:
        return "No hay análisis de riesgo disponible." if lang == "es" else "No risk analysis available."

    header = "Análisis de riesgo del portafolio:" if lang == "es" else "Portfolio Risk Analysis:"
    lines = [header, ""]

    status = result.get("status", "?")
    var_95 = result.get("var_95", 0.0)
    max_dd = result.get("max_drawdown", 0.0)
    sharpe = result.get("sharpe_ratio", 0.0)
    avg_corr = result.get("avg_correlation", 0.0)
    total_exp = result.get("total_exposure", 0.0)
    violations = result.get("violations", [])

    lines.append(f"  Estado:            {status}")
    lines.append(f"  VaR 95%:           {var_95:.2%}")
    lines.append(f"  Max drawdown:      {max_dd:.2%}")
    lines.append(f"  Sharpe ratio:      {sharpe:.2f}")
    lines.append(f"  Corr. promedio:    {avg_corr:.4f}")
    lines.append(f"  Exposición total:  {total_exp:.2%}")

    if violations:
        lines.append("")
        lines.append("  Violaciones:")
        for v in violations:
            lines.append(f"    - {v}")

    return "\n".join(lines)


def _format_stop_audit(result: dict, state: dict, positions: list, lang: str) -> str:
    """Per-position stop protection, from a dry-run pass of run_stop_audit."""
    if not result:
        return ("No hay auditoría de stops disponible." if lang == "es"
                else "No stop audit available.")

    header = ("Auditoría de stops (motor de salidas):" if lang == "es"
              else "Stop audit (exit engine):")
    lines = [header, ""]
    if result.get("open_orders_unavailable"):
        lines.append("  ⚠ estado de órdenes abiertas desconocido — nada verificado (fail-closed)"
                     if lang == "es" else
                     "  ⚠ open-order state unknown — nothing verified (fail-closed)")
        return "\n".join(lines)

    labels = {
        "protected": ("stop GTC activo al nivel del trailing", "GTC stop resting at the trail"),
        "placed": ("SIN stop en el bróker — la corrida lo colocaría", "NO broker stop — the run would place one"),
        "replaced": ("stop del bróker desfasado — la corrida lo subiría", "broker stop stale — the run would raise it"),
        "unprotected": ("SIN protección", "UNPROTECTED"),
        "errors": ("error", "error"),
    }
    by_ticker = {}
    for bucket, rows in result.items():
        if isinstance(rows, list):
            for r in rows:
                by_ticker[r["ticker"]] = (bucket, r)
    n_ok = sum(1 for b, _ in by_ticker.values() if b == "protected")
    lines.append(f"  Protegidas: {n_ok}/{len(by_ticker)}" if lang == "es"
                 else f"  Protected: {n_ok}/{len(by_ticker)}")
    lines.append("")
    for p in positions:
        ticker = str(p["symbol"]).replace(".", "-")
        bucket, row = by_ticker.get(ticker, ("unprotected", {"reason": "no evaluada"}))
        rec = state.get(ticker) or {}
        price = float(p.get("current_price") or 0)
        trail = rec.get("trail_stop")
        lines.append(f"{'─' * 40}")
        lines.append(f"{ticker}  ${price:,.2f}")
        if trail:
            dist = (price - float(trail)) / price if price else 0.0
            lines.append(f"  Trailing stop: ${float(trail):,.2f}  ({dist:.1%} "
                         f"{'abajo' if lang == 'es' else 'below'})"
                         + (f"  · decay {rec.get('decay_count', 0)}/5" if rec.get("decay_count") else ""))
        else:
            lines.append("  Sin registro del motor de salidas" if lang == "es"
                         else "  No exit-engine record")
        label = labels[bucket][0 if lang == "es" else 1]
        extra = row.get("reason") or row.get("error")
        lines.append(f"  Bróker: {label}" + (f" ({extra})" if extra and bucket in ("unprotected", "errors") else ""))
        lines.append("")
    return "\n".join(lines)


def _format_pm(result: dict, lang: str) -> str:
    if not result:
        return "No hay decisión del PM disponible." if lang == "es" else "No PM decision available."

    header = "Decisión del Portfolio Manager:" if lang == "es" else "Portfolio Manager Decision:"
    lines = [header, ""]

    comment = result.get("portfolio_comment", "")
    if comment:
        lines.append(f"  {comment}")
        lines.append("")

    recommendations = result.get("recommendations", [])
    actionable = [r for r in recommendations if r.get("action") != "HOLD"]

    if not actionable:
        hold_msg = "No hay acciones recomendadas este ciclo (todo HOLD)." if lang == "es" else "No actions recommended this cycle (all HOLD)."
        lines.append(f"  {hold_msg}")
    else:
        lines.append("  Recomendaciones:")
        for r in actionable:
            ticker = r.get("ticker", "?")
            action = r.get("action", "?")
            conviction = r.get("conviction", "?")
            rationale = r.get("rationale", "N/A")
            lines.append(f"  {ticker:<8}  {action}  (conv: {conviction}/5)")
            lines.append(f"    {rationale}")
            lines.append("")

    return "\n".join(lines)


def _format_regime_signals(results: dict, lang: str) -> str:
    if not results:
        return "No hay señales de régimen disponibles." if lang == "es" else "No regime signals available."

    header = "Señales ajustadas al régimen:" if lang == "es" else "Regime-Adjusted Signals:"
    lines = [header, ""]

    # Sort by composite_score descending when available, else alphabetically
    def _sort_key(item):
        _, v = item
        return -(v.get("composite_score") or v.get("score") or 0.0)

    sorted_items = sorted(results.items(), key=_sort_key)

    for ticker, r in sorted_items:
        signal = r.get("signal", "?")
        regime = r.get("regime", "?")
        strategy = r.get("strategy", "")
        composite = r.get("composite_score") or r.get("score")

        score_str = f"  score={composite:+.4f}" if composite is not None else ""
        strat_str = f"  [{strategy}]" if strategy else ""
        lines.append(f"{ticker:<8}  {signal:<5}{score_str}  régimen={regime}{strat_str}")

    return "\n".join(lines)


def _format_pending(rows: list, lang: str) -> str:
    if not rows:
        return ("No hay señales pendientes en la cola." if lang == "es"
                else "No pending signals queued.")

    header = ("Cola de compras aprobadas sin presupuesto:" if lang == "es"
              else "Approved buys awaiting budget:")
    lines = [header, ""]
    for r in rows:
        lines.append(
            f"  {r.get('ticker', '?'):<8} conv={r.get('conviction', '?')}/5  "
            f"score={r.get('composite_score') or 0.0:+.4f}  "
            f"size={r.get('size_pct') or 0.0:.2f}%  "
            f"desde {str(r.get('created_at', ''))[:10]}"
        )
    lines.append("")
    lines.append("Para ejecutar una: «compra <TICKER>» — se propone tamaño y niveles."
                 if lang == "es" else
                 "To act on one: \u201cbuy <TICKER>\u201d — size and levels are proposed.")
    return "\n".join(lines)


def _format_action_preview(action: dict, ctx: dict, lang: str) -> str:
    action_type = action.get("type")
    ticker = action.get("ticker", "?")
    qty = action.get("qty")
    sl_price = action.get("sl_price")
    tp_price = action.get("tp_price")

    if action_type == "close_position":
        # Look up qty from live positions
        live_qty = qty
        if live_qty is None:
            for p in ctx.get("live_positions", []):
                if p.get("symbol") == ticker:
                    live_qty = p.get("qty", "?")
                    break
        qty_str = str(live_qty) if live_qty is not None else "?"
        if lang == "es":
            return f"Solicitud: cerrar posición en {ticker} ({qty_str} acciones)"
        return f"Request: close position in {ticker} ({qty_str} shares)"

    elif action_type == "cancel_order":
        # Look up order details from open orders
        order_side = "?"
        order_qty = "?"
        order_id = "?"
        for o in ctx.get("open_orders", []):
            if o.get("symbol") == ticker:
                order_side = o.get("side", "?")
                order_qty = o.get("qty", "?")
                order_id = o.get("id", "?")
                break
        if lang == "es":
            return f"Solicitud: cancelar orden {ticker} {order_side} {order_qty} acciones (id: {order_id})"
        return f"Request: cancel order {ticker} {order_side} {order_qty} shares (id: {order_id})"

    elif action_type == "sell_partial":
        pct = action.get("pct")
        if qty is not None:
            amount = f"{qty} acciones" if lang == "es" else f"{qty} shares"
        elif pct is not None:
            amount = f"{float(pct) * 100 if float(pct) <= 1 else float(pct):.0f}%"
        else:
            amount = "?"
        if lang == "es":
            return f"Solicitud: vender {amount} de {ticker} @ mercado"
        return f"Request: sell {amount} of {ticker} @ market"

    elif action_type == "submit_buy":
        qty_str = str(qty) if qty is not None else "?"
        if lang == "es":
            return f"Solicitud: comprar {qty_str} acciones de {ticker} @ mercado"
        return f"Request: buy {qty_str} shares of {ticker} @ market"

    elif action_type == "submit_oco":
        qty_str = str(qty) if qty is not None else "?"
        sl_str = f"${sl_price:.2f}" if sl_price is not None else "?"
        tp_str = f"${tp_price:.2f}" if tp_price is not None else "?"
        if lang == "es":
            return f"Solicitud: OCO en {ticker} qty={qty_str}  SL={sl_str}  TP={tp_str}"
        return f"Request: OCO on {ticker} qty={qty_str}  SL={sl_str}  TP={tp_str}"

    else:
        if lang == "es":
            return f"Acción pendiente: {action_type} — {ticker}"
        return f"Pending action: {action_type} — {ticker}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_routing(routing: dict, ctx: dict, force_fresh: bool = False) -> tuple[str, dict | None]:
    """
    Dispatch to the correct agent(s) based on routing["agents"], format the
    response as human-readable text, and return (response_text, pending_action_or_None).

    Never raises — individual agent failures are caught and reported inline.
    """
    agents = routing.get("agents", ["portfolio_summary"])
    lang = routing.get("response_language", "es")
    tickers = routing.get("tickers") or []
    action = routing.get("action", {})

    # Normalize: routing["agents"] can be a str or a list (handle both)
    if isinstance(agents, str):
        agents = [agents]

    sections = []
    pending_action = None

    # Budget denial from the router must reach the user, not just the log
    notice = routing.get("budget_notice")
    if notice:
        sections.append(notice)

    for agent_name in agents:

        # ------------------------------------------------------------------
        # portfolio_summary — format ctx directly, no agent call
        # ------------------------------------------------------------------
        if agent_name == "portfolio_summary":
            sections.append(_format_portfolio_summary(ctx, lang))

        # ------------------------------------------------------------------
        # Action agents — build preview, set pending_action
        # ------------------------------------------------------------------
        elif agent_name in ("action_close", "action_cancel_order", "action_buy",
                            "action_oco", "action_sell_partial"):
            if action and action.get("type"):
                if pending_action is not None:
                    # Router should never return multiple actions — skip subsequent ones
                    continue
                sections.append(_format_action_preview(action, ctx, lang))
                pending_action = action
            else:
                sections.append(
                    "Acción recibida pero sin detalles suficientes para ejecutar."
                    if lang == "es"
                    else "Action received but insufficient details to execute."
                )

        # ------------------------------------------------------------------
        # propose_trade — the full decision panel for one name
        # ------------------------------------------------------------------
        elif agent_name == "propose_trade":
            try:
                from chat.proposal import build_proposal, render

                t = tickers[0] if tickers else None
                if not t:
                    sections.append(
                        "¿Sobre qué ticker? Dime el símbolo — ej. «¿compro NVDA?»."
                        if lang == "es" else
                        "Which ticker? Name the symbol — e.g. \u201cshould I buy NVDA?\u201d."
                    )
                else:
                    proposal = build_proposal(t, ctx, force_fresh=force_fresh)
                    sections.append(render(proposal, lang))
                    if pending_action is None:
                        # The proposal rides along so chat/actions.py can journal
                        # the sizing and the gate verdicts that were on screen.
                        _act = proposal.to_action()
                        _act["_proposal"] = proposal
                        pending_action = _act
            except Exception as e:
                sections.append(f"[propose_trade] Error: {e}")

        # ------------------------------------------------------------------
        # list_pending — BUYs the pipeline approved but could not fund
        # ------------------------------------------------------------------
        elif agent_name == "list_pending":
            try:
                from db.database import load_pending_signals

                sections.append(_format_pending(load_pending_signals() or [], lang))
            except Exception as e:
                sections.append(f"[list_pending] Error: {e}")

        # ------------------------------------------------------------------
        # research
        # ------------------------------------------------------------------
        elif agent_name == "research":
            try:
                from agents.research_agent import run_research
                t = tickers if tickers else ALL_TICKERS
                results = run_research(t)
                sections.append(_format_research(results, lang))
            except Exception as e:
                sections.append(f"[research] Error: {e}")

        # ------------------------------------------------------------------
        # sentiment
        # ------------------------------------------------------------------
        elif agent_name == "sentiment":
            try:
                from agents.sentiment_agent import run_sentiment
                t = tickers if tickers else ALL_TICKERS
                results = run_sentiment(t)
                sections.append(_format_sentiment(results, lang))
            except Exception as e:
                sections.append(f"[sentiment] Error: {e}")

        # ------------------------------------------------------------------
        # strategy
        # ------------------------------------------------------------------
        elif agent_name == "strategy":
            try:
                from agents.strategy_agent import generate_signals
                # ALWAYS the full universe. generate_signals z-scores its factors
                # ACROSS the list it is given: with one ticker pandas' std() is
                # NaN, the divisor falls back to 1e-9, composite comes out 0.0
                # and 0.0 >= SIGNAL_THRESHOLD is never true — so asking for one
                # name used to answer "HOLD, composite +0.0000" every time,
                # whatever the name. Compute over the universe, then show the
                # rows that were asked about.
                results = generate_signals(ALL_TICKERS)
                if tickers:
                    results = {t: v for t, v in results.items() if t in tickers}
                sections.append(_format_strategy(results, lang))
            except Exception as e:
                sections.append(f"[strategy] Error: {e}")

        # ------------------------------------------------------------------
        # regime
        # ------------------------------------------------------------------
        elif agent_name == "regime":
            try:
                from agents.regime_agent import detect_regime
                result = detect_regime()
                sections.append(_format_regime(result, lang))
            except Exception as e:
                sections.append(f"[regime] Error: {e}")

        # ------------------------------------------------------------------
        # regime_strategy — with optional cache bust via force_fresh
        # ------------------------------------------------------------------
        elif agent_name == "regime_strategy":
            if force_fresh:
                today = datetime.today().strftime("%Y-%m-%d")
                cache_path = os.path.join(
                    os.path.dirname(__file__), "..", "logs", f"regime_signals_{today}.json"
                )
                if os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass

            try:
                from agents.regime_strategy_agent import generate_regime_signals
                regime = ctx.get("portfolio_state", {}).get("regime") or "BULL"
                results = generate_regime_signals(regime)
                # regime_strategy returns dict of ticker -> signal_dict
                sections.append(_format_regime_signals(results, lang))
            except Exception as e:
                sections.append(f"[regime_strategy] Error: {e}")

        # ------------------------------------------------------------------
        # risk
        # ------------------------------------------------------------------
        elif agent_name == "risk":
            try:
                from agents.risk_agent import analyze_portfolio_risk
                positions_weights = _positions_for_risk(ctx)
                if positions_weights:
                    result = analyze_portfolio_risk(positions_weights)
                else:
                    result = {"status": "OK", "var_95": 0.0, "max_drawdown": 0.0,
                              "sharpe_ratio": 0.0, "avg_correlation": 0.0,
                              "total_exposure": 0.0, "violations": []}
                sections.append(_format_risk(result, lang))
            except Exception as e:
                sections.append(f"[risk] Error: {e}")

        # ------------------------------------------------------------------
        # risk_sl_tp
        # ------------------------------------------------------------------
        elif agent_name == "risk_sl_tp":
            # A status question must not place orders: dry_run classifies
            # every position against its resting broker order and sends nothing.
            try:
                from agents.risk_sl_tp_agent import run_stop_audit
                from execution.order_manager import load_position_state
                positions_list = _positions_for_stop_audit(ctx)
                state = load_position_state()
                result = (run_stop_audit(positions_list, state, exited=set(), dry_run=True)
                          if positions_list else {})
                sections.append(_format_stop_audit(result, state, positions_list, lang))
            except Exception as e:
                sections.append(f"[risk_sl_tp] Error: {e}")

        # ------------------------------------------------------------------
        # pm — full chain: research → risk → pm
        # ------------------------------------------------------------------
        elif agent_name == "pm":
            try:
                from agents.research_agent import run_research
                from agents.risk_agent import analyze_portfolio_risk
                from agents.pm_agent import run_pm

                from agents.sentiment_agent import run_sentiment

                t = tickers if tickers else ALL_TICKERS
                research_results = run_research(t)
                sentiment_results = run_sentiment(t)

                positions_weights = _positions_for_risk(ctx)
                risk_report = analyze_portfolio_risk(positions_weights) if positions_weights else {}

                # sentiment_results and min_conviction were both missing. The
                # second is the bug config.effective_min_conviction exists to
                # prevent: the PM approving at conviction 3 while the executor
                # demands 4, so every BUY it approved at exactly 3 was dropped.
                import config
                _regime = ctx.get("portfolio_state", {}).get("regime") or "BULL"
                result = run_pm(research_results, risk_report,
                                sentiment_results=sentiment_results,
                                min_conviction=config.effective_min_conviction(_regime))
                sections.append(_format_pm(result, lang))
            except Exception as e:
                sections.append(f"[pm] Error: {e}")

        # ------------------------------------------------------------------
        # Unknown agent
        # ------------------------------------------------------------------
        else:
            sections.append(
                f"Agente desconocido: '{agent_name}'"
                if lang == "es"
                else f"Unknown agent: '{agent_name}'"
            )

    response_text = "\n\n".join(s for s in sections if s)
    return response_text, pending_action

