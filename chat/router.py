"""
chat/router.py — Claude-based question router for the AI hedge fund terminal.

Public API:
    route_question(question: str, ctx: dict) -> dict
"""

import copy

from utils.llm import get_client
from utils.env import load_project_env

# This checkout's .env only. Walking up to "handle worktree paths" is what
# handed test processes the parent checkout's production credentials.
load_project_env()

_client = get_client()  # real Anthropic client, or the offline stub when LLM_MODE=stub

_FALLBACK = {
    "agents": ["portfolio_summary"],
    "tickers": [],
    "action": {
        "type": None,
        "ticker": None,
        "qty": None,
        "pct": None,
        "sl_price": None,
        "tp_price": None,
    },
    "response_language": "es",
    "reasoning": "Router fallback — defaulting to portfolio summary.",
}

# Shown verbatim to the user when the governor denies the routing call. The
# terminal is interactive: "presupuesto diario agotado" is actionable, a
# generic router error is not.
_BUDGET_NOTICE = (
    "⚠️ Presupuesto diario agotado (gasto LLM) — no puedo clasificar la pregunta "
    "ni invocar agentes de análisis. Respondo solo con el estado del portafolio "
    "desde el contexto ya cargado."
)

_SYSTEM_PROMPT = """You are a routing agent for an AI hedge fund terminal. Given a user question, a live portfolio context snapshot, and optional conversation history, decide which specialist agents should respond and whether the question implies a trade action.

AVAILABLE AGENTS:
1. "research" — Fundamental analysis: investment thesis, conviction, recommendation for tickers. Use for: "¿vale la pena comprar X?", "análisis de NVDA", "qué opinas de META", "research on AAPL".
2. "sentiment" — News sentiment score and summary. Use for: "qué dicen las noticias de X", "cómo está el sentimiento de META", "news sentiment TSLA".
3. "strategy" — Quantitative signals: momentum Z-score, value Z-score, composite score, BUY/HOLD. Use for: "qué señal tiene X", "composite score de AAPL", "señal de estrategia para MSFT".
4. "regime" — Market regime: BULL/BEAR, VIX, size multiplier. Use for: "cómo está el mercado", "qué régimen estamos", "VIX actual", "market regime", "¿es momento de comprar?".
5. "regime_strategy" — Regime-adjusted ranked signals. Use for: "cuáles son las mejores acciones según el régimen", "top picks del régimen actual", "qué tickers tienen señal BUY ahora".
6. "risk" — Portfolio-level risk: VaR 95%, max drawdown, Sharpe ratio, correlation. Use for: "cómo está el riesgo", "cuál es el VaR", "Sharpe ratio", "¿hay violaciones de riesgo?".
7. "risk_sl_tp" — Per-position stop-loss / take-profit audit AND exit logic explanation. Use for: "SL y TP de mis posiciones", "¿están protegidas mis posiciones?", "auditoría de stops", "revisar stops", "hasta qué punto mantienes X", "qué factores consideras para cerrar", "cuáles son los criterios de salida", "por qué cerrarías X", "qué determina el cierre de una posición", "explica tu lógica de salida", "a qué precio saldrías de X", "cuándo cerrarías X".
8. "pm" — Portfolio Manager synthesis: BUY/SELL/HOLD verdicts AND reasoning about portfolio decisions. Use for: "qué debería hacer con mi portafolio", "decisión del PM", "aprueba el PM comprar X", "por qué mantienes X abierta", "qué te haría cerrar X", "justifica tener X", "cuál es la tesis detrás de X". WARNING: this is expensive — only route here when user explicitly asks for portfolio-level reasoning or decisions.
9. "portfolio_summary" — Answer from context only, no agent call needed. Use for: "cómo está mi portafolio", "resumen de posiciones", "equity", "¿cuánto tengo?", "qué posiciones tengo", "mis órdenes abiertas". Only use this for factual portfolio state questions that need NO reasoning or analysis.
10. "action_close" — Close an existing position entirely. Use ONLY when user explicitly says: "cierra X", "vende X", "close X", "sell X", "salir de X", "exit X".
11. "action_cancel_order" — Cancel an open order. Use ONLY when user says: "cancela la orden de X", "cancel order X".
12. "action_buy" — Submit a market buy order. Use ONLY when user says: "compra X acciones de Y", "buy N shares of Y".
13. "action_oco" — Submit an OCO order. Use ONLY when user says: "pon OCO para X", "set OCO on X", with explicit prices.
14. "propose_trade" — Build a FULL trade proposal for ONE ticker: analysis, suggested share count, SL/TP levels and the verdict of every risk gate, then ask the user to confirm. Use whenever the user asks WHETHER or HOW MUCH to buy rather than ordering an explicit quantity: "¿compro NVDA?", "analiza AAPL para comprar", "cuántas acciones de MSFT debería comprar", "propón una entrada en META", "¿vale la pena entrar en X?", "¿me conviene X?", "should I buy X".
15. "action_sell_partial" — Sell PART of an existing position. Use when the user says: "vende la mitad de X", "recorta 30% de X", "trim X", "vende 10 acciones de X" while they hold more than 10.
16. "list_pending" — Show the pipeline's queued BUY candidates (approved but never funded). Use for: "qué tienes pendiente", "cola de señales", "pending signals", "qué querías comprar y no pudiste".

RULES:
- Record the routing decision with the route_decision tool.
- action_* agents must NOT be combined with analytical agents (agents list should contain only one action_* entry when routing to actions).
- IMPORTANT: Only use "portfolio_summary" for pure factual questions (show positions, show equity). For ANY question asking about reasoning, factors, criteria, logic, or "why" — use the appropriate analytical agent.
- If a follow-up question (e.g., "pero dime más", "explícame", "y los factores?") references the previous question in CONVERSATION HISTORY, route to the same agent(s) used in that previous turn.
- Extract ticker symbols in uppercase (e.g., AAPL, NVDA, BRK-B). Ignore articles and prepositions.
- If no tickers mentioned, check if question refers to a position in the context — infer ticker from context if obvious.
- Detect language: use "es" for Spanish questions, "en" for English.
- For action_* routing: set action.type to "close_position", "cancel_order", "submit_buy", "submit_oco" or "sell_partial" respectively. Set action.ticker, qty, sl_price, tp_price from the question when available.
- For "action_sell_partial": set action.qty when the user names a share count, or action.pct when they name a fraction ("la mitad" -> 0.5, "30%" -> 0.3, "un tercio" -> 0.33). Never set both.
- "propose_trade" MUST leave action.type and action.qty null. The sizing engine computes the quantity from equity, budget, regime and the risk caps — never invent one.
- Prefer "propose_trade" over "action_buy" unless the user states an explicit number of shares. "compra NVDA" is a proposal; "compra 20 acciones de NVDA" is an action.
- "propose_trade" and "list_pending" are analytical: they may not be combined with any action_* agent."""

_ROUTER_TOOL = {
    "name": "route_decision",
    "description": "Record the routing decision for the user's question.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "agents": {"type": "array", "items": {"type": "string"}},
            "tickers": {"type": "array", "items": {"type": "string"}},
            "action": {
                "type": "object",
                "properties": {
                    "type": {"type": ["string", "null"]},
                    "ticker": {"type": ["string", "null"]},
                    "qty": {"type": ["number", "null"]},
                    "pct": {"type": ["number", "null"]},
                    "sl_price": {"type": ["number", "null"]},
                    "tp_price": {"type": ["number", "null"]},
                },
                "required": ["type", "ticker", "qty", "pct", "sl_price", "tp_price"],
                "additionalProperties": False,
            },
            "response_language": {"type": "string", "enum": ["es", "en"]},
            "reasoning": {"type": "string"},
        },
        "required": ["agents", "tickers", "action", "response_language", "reasoning"],
        "additionalProperties": False,
    },
}


def _build_context_summary(ctx: dict) -> str:
    """Build a compact text summary of the context dict (< 300 tokens)."""
    lines = []

    # Account / portfolio_state info
    account = ctx.get("account", {})
    portfolio_state = ctx.get("portfolio_state", {})

    equity = account.get("equity") if "equity" in account else portfolio_state.get("equity_real", 0)
    cash = account.get("cash") if "cash" in account else portfolio_state.get("cash", 0)
    regime = portfolio_state.get("regime", "UNKNOWN")

    lines.append(f"Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | Regime: {regime}")

    # Open positions
    positions = ctx.get("live_positions", [])
    if positions:
        lines.append("Positions:")
        for p in positions:
            sym = p.get("symbol", "?")
            qty = p.get("qty", 0)
            price = p.get("current_price", 0)
            plpc = p.get("unrealized_plpc", 0)
            sign = "+" if plpc >= 0 else ""
            lines.append(f"  {sym} qty={qty} price=${price:.2f} pnl={sign}{plpc*100:.1f}%")
    else:
        lines.append("Positions: none")

    # Open orders
    orders = ctx.get("open_orders", [])
    if orders:
        lines.append("Open orders:")
        for o in orders:
            sym = o.get("symbol", "?")
            side = o.get("side", "?")
            qty = o.get("qty", 0)
            status = o.get("status", "?")
            lines.append(f"  {sym} side={side} qty={qty} status={status}")
    else:
        lines.append("Open orders: none")

    return "\n".join(lines)


def route_question(question: str, ctx: dict, history: list | None = None) -> dict:
    """
    Route a user question to the appropriate specialist agents.

    Args:
        question: The user's natural language question.
        ctx: A ContextSnapshot dict from chat.context.load_context().
        history: Optional list of recent turns, each a dict with keys
                 "question" (str) and "agents" (list[str]). Last 3 turns max.
                 Used so the router can understand follow-up questions.

    Returns:
        A routing decision dict with keys: agents, tickers, action,
        response_language, reasoning.
    """
    try:
        context_summary = _build_context_summary(ctx)

        history_block = ""
        if history:
            lines = ["CONVERSATION HISTORY (most recent last):"]
            for turn in history[-3:]:
                agents_str = ", ".join(turn.get("agents", []))
                lines.append(f"  Q: {turn['question']}")
                lines.append(f"  A: [routed to: {agents_str}]")
            history_block = "\n".join(lines) + "\n\n"

        user_message = (
            f"PORTFOLIO CONTEXT:\n{context_summary}\n\n"
            f"{history_block}"
            f"USER QUESTION: {question}"
        )

        import config
        from agents.cost_governor import authorize, est_tokens

        decision = authorize(
            "chat_router", config.CLAUDE_SENTIMENT_MODEL,
            est_tokens_in=est_tokens(_SYSTEM_PROMPT, user_message),
            est_max_tokens_out=512,
        )
        if not decision.allowed:
            print(f"  [router] {_BUDGET_NOTICE}")
            blocked = copy.deepcopy(_FALLBACK)
            blocked["reasoning"] = "Presupuesto diario agotado — routing por defecto."
            blocked["budget_notice"] = _BUDGET_NOTICE
            return blocked

        # .stream() not .create(): consistent with research/pm/sentiment
        # agents, which hit the SDK's non-streaming >10min guard once their
        # ceilings scaled with the universe. 512 tokens never approaches it
        # here, but the call shape stays uniform across every agent.
        with _client.messages.stream(
            model=decision.model,  # intent classification is a Haiku task; may degrade further
            max_tokens=decision.max_tokens,
            temperature=0.2,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            tools=[_ROUTER_TOOL],
            tool_choice={"type": "tool", "name": "route_decision"},
        ) as stream:
            message = stream.get_final_message()
        from agents.llm_usage import record_usage
        record_usage("chat_router", decision.model, message)

        result = next(b for b in message.content if b.type == "tool_use").input

        # Ensure required keys are present; fill missing with fallback values
        result.setdefault("agents", ["portfolio_summary"])
        result.setdefault("tickers", [])
        result.setdefault("action", {
            "type": None,
            "ticker": None,
            "qty": None,
            "pct": None,
            "sl_price": None,
            "tp_price": None,
        })
        result.setdefault("response_language", "es")
        result.setdefault("reasoning", "")

        return result

    except Exception as e:
        print(f"  [router] Error routing question: {e}")
        return copy.deepcopy(_FALLBACK)
