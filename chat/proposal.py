"""Build a full trade proposal for one ticker: analysis, size, levels, gates.

Default path reads TODAY'S RUN, it does not recompute it. Two reasons, and the
second is not an optimisation:

1. agents/strategy_agent.py's z-scores are CROSS-SECTIONAL. Call
   generate_signals(["MSFT"]) and pandas' std() over one element is NaN, the
   divisor falls back to 1e-9, composite comes out 0.0, and 0.0 >= 0.3 is never
   true — so a single-ticker call reports HOLD for every name, forever. The
   chat has been doing exactly that. The composite therefore comes from the
   cached run, where it was computed against the whole universe.
2. The gate verdicts shown to the user should be the ones the pipeline actually
   applied this morning, not a second opinion computed from different inputs.

`force_fresh` re-runs research, sentiment and the PM for this one ticker — all
three are per-name and safe alone — but still takes the composite from the
cached run.
"""
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from agents.entry_gates import EntryFacts, evaluate_entry, failures
from chat import pretrade
from chat.sizing import peer_count_from_run, suggest_size


@dataclass
class Proposal:
    ticker: str
    side: str = "buy"
    source: str = "cached_run"       # cached_run | fresh | no_data
    as_of: str = ""

    regime: str = "UNKNOWN"
    vol_regime: str = "?"
    size_multiplier: float = 1.0

    composite: "float | None" = None
    strategy_signal: str = "?"
    regime_signal: str = "?"

    pm_action: "str | None" = None
    conviction: int = 0
    rationale: str = ""
    recommendation: str = "?"
    thesis: str = ""
    key_risks: list = field(default_factory=list)
    sentiment: str = "NEUTRAL"
    sentiment_score: float = 0.0

    atr_14: float = 0.0
    ref_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    rr: float = 0.0

    size: object = None
    risk_sized_qty: int = 0

    entry_gates: list = field(default_factory=list)
    chat_gates: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def hard_failures(self) -> list:
        return pretrade.hard_failures(self.chat_gates)

    @property
    def soft_failures(self) -> list:
        return pretrade.soft_failures(self.chat_gates) + failures(self.entry_gates)

    @property
    def blocked(self) -> bool:
        return bool(self.hard_failures)

    @property
    def qty(self) -> int:
        return getattr(self.size, "qty", 0) or 0

    def to_action(self) -> dict:
        return {
            "type": "submit_buy",
            "ticker": self.ticker,
            "qty": self.qty,
            "sl_price": self.sl_price or None,
            "tp_price": self.tp_price or None,
        }


# ── inputs ──────────────────────────────────────────────────────────────────

def _last_run() -> dict:
    try:
        from db.database import load_recent_runs

        runs = load_recent_runs(1)
        return runs[0] if runs else {}
    except Exception as e:
        print(f"  [proposal] no se pudo leer la última corrida: {e}")
        return {}


def _regime(ctx: dict, run: dict) -> dict:
    """detect_regime() is day-cached, so this is cheap; the run blob is the fallback."""
    try:
        from agents.regime_agent import detect_regime

        return detect_regime() or {}
    except Exception:
        return run.get("regime") or {"regime": ctx.get("portfolio_state", {}).get("regime")}


def _pick(rows, ticker: str) -> dict:
    for r in rows or []:
        if r.get("ticker") == ticker:
            return r
    return {}


def _sector_exposure(ticker: str, ctx: dict) -> float:
    """Market value already held in this ticker's sector."""
    try:
        from agents.sector_rotation_agent import _get_ticker_sector

        target = _get_ticker_sector(ticker)
        total = 0.0
        for p in ctx.get("live_positions", []):
            if _get_ticker_sector(p.get("symbol", "")) == target:
                total += float(p.get("market_value", 0.0))
        return total
    except Exception:
        return 0.0


def _fresh_analysis(ticker: str, ctx: dict, regime: dict) -> tuple:
    """Re-run the per-name agents. Never fans out over the universe."""
    from agents.pm_agent import run_pm
    from agents.research_agent import run_research
    from agents.sentiment_agent import run_sentiment

    research = run_research([ticker]) or []
    sentiment = run_sentiment([ticker]) or []

    risk_report = {}
    try:
        from agents.risk_agent import analyze_portfolio_risk

        equity = ctx.get("account", {}).get("equity", 0) or 0
        weights = {p["symbol"]: p["market_value"] / equity
                   for p in ctx.get("live_positions", [])
                   if equity > 0 and p.get("market_value", 0) > 0}
        if weights:
            risk_report = analyze_portfolio_risk(weights)
    except Exception as e:
        print(f"  [proposal] riesgo de cartera no disponible: {e}")

    # sentiment_results and min_conviction were both missing from the chat's
    # run_pm call. The second one is the bug config.effective_min_conviction
    # exists to prevent: the PM approving at 3 while the executor demands 4.
    pm = run_pm(research, risk_report, sentiment_results=sentiment,
                min_conviction=config.effective_min_conviction(
                    regime.get("regime") or "BULL"))
    return research, sentiment, pm


# ── public ──────────────────────────────────────────────────────────────────

def build_proposal(ticker: str, ctx: dict, *, side: str = "buy",
                   force_fresh: bool = False) -> Proposal:
    """Everything needed to decide on one trade, in one object. Never raises."""
    ticker = (ticker or "").upper().strip()
    p = Proposal(ticker=ticker, side=side)

    run = _last_run()
    regime = _regime(ctx, run)
    p.regime = regime.get("regime") or "UNKNOWN"
    p.vol_regime = regime.get("vol_regime") or "?"
    p.size_multiplier = float(regime.get("size_multiplier") or 1.0)

    # The composite is cross-sectional and must come from the batch run.
    strategy = (run.get("strategy") or {}).get(ticker, {})
    p.composite = strategy.get("composite_score")
    p.strategy_signal = strategy.get("signal", "?")
    p.regime_signal = (run.get("regime_signals") or {}).get(ticker, {}).get("signal", "?")
    p.as_of = run.get("timestamp", "")
    if not run:
        p.source = "no_data"
        p.notes.append("No hay corrida previa en la base — sin composite ni veredicto "
                       "del PM. Usa --fresh para analizar este nombre ahora.")

    if force_fresh:
        try:
            research, sentiment, pm = _fresh_analysis(ticker, ctx, regime)
            p.source = "fresh"
            p.as_of = datetime.now().isoformat(timespec="seconds")
        except Exception as e:
            print(f"  [proposal] análisis fresco falló: {e}")
            research, sentiment, pm = (run.get("research"), run.get("sentiment"),
                                       run.get("pm"))
            p.notes.append(f"Análisis fresco falló ({e}) — usando la última corrida.")
    else:
        research, sentiment, pm = (run.get("research"), run.get("sentiment"),
                                   run.get("pm"))

    res = _pick(research, ticker)
    p.recommendation = res.get("recommendation", "?")
    p.thesis = res.get("thesis", "")
    p.key_risks = res.get("key_risks") or []
    p.atr_14 = float(res.get("atr_14") or 0.0)

    sent = _pick(sentiment, ticker)
    p.sentiment = sent.get("sentiment", "NEUTRAL")
    p.sentiment_score = float(sent.get("sentiment_score") or 0.0)

    pm_rec = _pick((pm or {}).get("recommendations"), ticker)
    p.pm_action = pm_rec.get("action")
    p.conviction = int(pm_rec.get("conviction") or res.get("conviction") or 0)
    p.rationale = pm_rec.get("rationale", "")

    # ── price, levels ──
    try:
        from execution.trade_executor import resolve_price

        p.ref_price = resolve_price(ticker)
    except Exception as e:
        print(f"  [proposal] sin precio para {ticker}: {e}")
        p.ref_price = 0.0

    if p.atr_14 > 0 and p.ref_price > 0:
        # The exit engine's initial stop (design C); no take-profit exists any
        # more, so there is no R:R to quote — the chandelier trail is the exit.
        p.sl_price = round(p.ref_price - config.STOP_ATR_MULT * p.atr_14, 2)
    elif p.ref_price > 0:
        p.notes.append("Sin ATR en la corrida — sin nivel de stop; la compra se "
                       "registrará al vuelo con el ATR vivo en la próxima corrida.")

    # ── size ──
    account = ctx.get("account", {})
    equity = float(account.get("equity") or 0.0)
    positions = ctx.get("live_positions", [])
    current_mv = sum(float(x.get("market_value") or 0.0)
                     for x in positions if x.get("symbol") == ticker)
    long_mv = sum(max(float(x.get("market_value") or 0.0), 0.0) for x in positions)

    p.size = suggest_size(
        ticker=ticker, price=p.ref_price, equity=equity,
        buying_power=float(account.get("buying_power") or 0.0),
        long_market_value=long_mv, vol_size_mult=p.size_multiplier,
        peer_count=peer_count_from_run(run), current_mv=current_mv,
        sector_mv=_sector_exposure(ticker, ctx),
    )

    # Fixed-risk sizing, shown as a contrast only. Executing it would give the
    # chat a different size than the pipeline for the same name.
    if p.sl_price > 0 and p.ref_price > 0 and equity > 0:
        try:
            from agents.risk_agent import compute_position_size

            p.risk_sized_qty = compute_position_size(equity, p.ref_price, p.sl_price)
        except Exception:
            p.risk_sized_qty = 0

    # ── gates ──
    held = {x.get("symbol") for x in positions}
    bought_today = {o.get("ticker") for o in (run.get("orders_submitted") or [])}
    p.entry_gates = evaluate_entry(EntryFacts(
        ticker=ticker,
        pm_action=p.pm_action,
        conviction=p.conviction,
        composite=p.composite,
        pm_degraded=bool((pm or {}).get("degraded")),
        min_conviction=config.effective_min_conviction(p.regime),
        regime=p.regime,
        strategy_buy=p.strategy_signal == "BUY",
        regime_buy=p.regime_signal == "BUY",
        sentiment=p.sentiment,
        held=ticker in held,
        macro_stressed=bool(strategy.get("macro_stressed")),
        credit_z=float(strategy.get("credit_spread_z") or 0.0),
    ))
    p.chat_gates = pretrade.evaluate_chat_gates(
        kind="buy", ticker=ticker, ctx=ctx, size=p.size, equity=equity,
        held_today=ticker in bought_today,
    )
    return p


# ── rendering ───────────────────────────────────────────────────────────────

def _gate_line(g) -> str:
    mark = "✓" if g.passed else ("⛔" if g.severity == "hard" else "⚠")
    detail = f" — {g.detail}" if g.detail else ""
    return f"    {mark} {g.name}{detail}"


def render(p: Proposal, lang: str = "es") -> str:
    """The panel the user reads before answering s/n."""
    L = []
    src = {"cached_run": f"corrida {p.as_of[:16]}", "fresh": "análisis fresco",
           "no_data": "sin corrida previa"}.get(p.source, p.source)
    L.append(f"PROPUESTA — {p.side.upper()} {p.ticker}   [{src}]")
    L.append("=" * 58)

    L.append(f"  Régimen: {p.regime} ({p.vol_regime}, x{p.size_multiplier:.2f})   "
             f"Precio: ${p.ref_price:,.2f}")
    comp = f"{p.composite:+.4f}" if p.composite is not None else "n/d"
    L.append(f"  Señal: estrategia={p.strategy_signal} régimen={p.regime_signal}  "
             f"composite={comp}")
    L.append(f"  PM: {p.pm_action or 'n/d'} (convicción {p.conviction}/5)   "
             f"Research: {p.recommendation}   Sentimiento: {p.sentiment} "
             f"({p.sentiment_score:+.2f})")
    if p.thesis:
        L.append(f"  Tesis: {p.thesis}")
    if p.rationale:
        L.append(f"  PM dice: {p.rationale}")
    if p.key_risks:
        L.append(f"  Riesgos: {', '.join(p.key_risks[:3])}")

    L.append("")
    s = p.size
    if s and s.qty > 0:
        L.append(f"  TAMAÑO SUGERIDO: {s.qty} acciones ≈ ${s.qty * s.price:,.0f}")
        L.append(f"    reparto {s.alloc_pct:.2%} del equity  "
                 f"(budget {s.deployable_frac:.2f}x ÷ n={s.n}; "
                 f"n_floor={s.n_floor}, candidatos hoy={s.peer_count})")
        L.append(f"    limita: {s.binding}")
        for cap in s.caps:
            if cap.name == "target":
                continue
            L.append(f"      {cap.label}: tope ${cap.limit:,.0f} · "
                     f"libre ${cap.headroom:,.0f}")
    else:
        L.append("  TAMAÑO SUGERIDO: 0 acciones — no hay presupuesto para esta entrada")
    if s and s.budget_reason:
        L.append(f"    nota de presupuesto: {s.budget_reason}")

    if p.sl_price:
        risk = p.ref_price - p.sl_price
        L.append("")
        L.append(f"  NIVELES: stop inicial ${p.sl_price:,.2f} (-{risk / p.ref_price:.1%}, "
                 f"{config.STOP_ATR_MULT}×ATR)  trailing {config.TRAIL_ATR_MULT}×ATR bajo el máximo")
        if p.risk_sized_qty:
            L.append(f"    contraste: a 1% de riesgo por operación serían "
                     f"{p.risk_sized_qty} acciones")

    hard = p.hard_failures
    soft = p.soft_failures
    L.append("")
    if hard:
        L.append("  BLOQUEOS (no anulables):")
        L.extend(_gate_line(g) for g in hard)
    if soft:
        L.append("  AVISOS (anulables escribiendo 'override'):")
        L.extend(_gate_line(g) for g in soft)
    if not hard and not soft:
        L.append("  COMPUERTAS: todas pasan.")

    for note in p.notes:
        L.append(f"  · {note}")

    return "\n".join(L)
