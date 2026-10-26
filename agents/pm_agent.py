import os
import sys as _sys

from utils.llm import get_client
from utils.env import load_project_env

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import config
from agents.cost_governor import authorize, est_tokens, notify_degraded
from agents.llm_usage import record_usage
from utils.retry import with_retry

load_project_env()

client = get_client()  # real Anthropic client, or the offline stub when LLM_MODE=stub

_PM_SYSTEM = [{
    "type": "text",
    "text": (
        "You are the Portfolio Manager for a systematic equity strategy. "
        "Your role is VETO GATE only: decide BUY / SELL / HOLD and conviction for each ticker, "
        "and record the decision with the record_decisions tool. "
        "Position SIZING is handled mechanically — always set size_pct to 0.0.\n\n"
        "Rules:\n"
        "- size_pct MUST be 0.0 for every ticker\n"
        "- BUY = approve entry; SELL = veto/exit; HOLD = skip\n"
        "- SELL on deteriorating fundamentals or negative risk-reward\n"
        "- Respect risk constraints — SELL if violations exist"
    ),
    # NOTE: no cache_control — prompt (~200 tok) is far below Sonnet's
    # 2048-token minimum cacheable prefix; a marker here never caches.
}]

_PM_TOOL = {
    "name": "record_decisions",
    "description": "Record the PM's BUY/SELL/HOLD decision for every ticker.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                        "size_pct": {"type": "number"},
                        "conviction": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                        "rationale": {"type": "string"},
                    },
                    "required": ["ticker", "action", "size_pct", "conviction", "rationale"],
                    "additionalProperties": False,
                },
            },
            "portfolio_comment": {"type": "string"},
        },
        "required": ["recommendations", "portfolio_comment"],
        "additionalProperties": False,
    },
}


# One decision is an action, a conviction and a one-line rationale. Measured at
# ~150 output tokens each, so 220 leaves half again as much headroom per ticker.
# The floor keeps small universes (and the single-ticker calls in the suite) on
# the ceiling this call has always used.
_TOKENS_PER_DECISION = 220
_MIN_OUTPUT_TOKENS = 8192
_MAX_OUTPUT_TOKENS = 32000


def _output_ceiling(n_tickers: int) -> int:
    """Output ceiling for a batch of `n_tickers`.

    This was hard-coded at 8192, sized for a 20-name universe. _extract_decisions
    raises when the model returns fewer recommendations than tickers asked for,
    and every raise falls back to raw research with degraded=True — so a
    truncated response does not fail loudly, it quietly removes the PM veto from
    the cycle. Research and sentiment already learned this lesson; see
    research_agent._output_ceiling and tests/test_batch_truncation.py.
    """
    return max(_MIN_OUTPUT_TOKENS,
               min(_MAX_OUTPUT_TOKENS, _TOKENS_PER_DECISION * n_tickers))


def _extract_decisions(message, expected: int) -> dict:
    """Pull the record_decisions payload out of `message`, or raise ValueError saying why not.

    A forced tool call that hits the token ceiling comes back as
    stop_reason="max_tokens" with a TRUNCATED tool_use block whose `.input`
    is `{}` — no exception, no error, just a PM that recommends nothing and
    silently blocks every entry. Every shape that is not a complete decision
    set has to become an explicit failure here.
    """
    if getattr(message, "stop_reason", None) == "max_tokens":
        raise ValueError("truncated: stop_reason=max_tokens")
    try:
        block = next(b for b in message.content if b.type == "tool_use")
    except StopIteration:
        # StopIteration would also be swallowed by `except Exception`, but only
        # by accident, and it reports no useful reason. Name it.
        raise ValueError("no tool_use block in response") from None
    result = block.input
    if not isinstance(result, dict):
        raise ValueError(f"tool input is {type(result).__name__}, not dict")
    recs = result.get("recommendations")
    if not isinstance(recs, list) or not recs:
        raise ValueError("recommendations missing or empty")
    if len(recs) != expected:
        raise ValueError(f"{len(recs)} recommendations, expected {expected}")
    return result


def _research_fallback(research_results: list, comment: str) -> dict:
    """PM verdicts straight from research when the synthesis call is unavailable.

    `degraded` is the contract with the executor, and it is set for *every*
    reason we end up here — a truncated tool call and an exhausted budget both
    mean the veto gate did not run. main.py raises a health alert on it and,
    more importantly, degrades to quant-only entries: the PM conviction bar is
    dropped (the convictions below are *research* convictions, not PM verdicts)
    and the strategy composite must clear SIGNAL_THRESHOLD on its own. See
    `_conviction_gate_ok` / `_signal_gate_ok` in main.run_entries — a fallback
    run must never be indistinguishable from a healthy one.
    """
    return {
        "recommendations": [
            {"ticker": r["ticker"], "action": r["recommendation"],
             "size_pct": 3.0 if r["recommendation"] == "BUY" else 0.0,
             "conviction": r["conviction"], "rationale": r["thesis"]}
            for r in research_results
        ],
        "portfolio_comment": comment,
        "degraded": True,
    }


_PM_FAILED_COMMENT = "PM synthesis failed; using research recommendations directly."


def run_pm(research_results: list, risk_report: dict,
           sentiment_results: list | None = None,
           min_conviction: int | None = None) -> dict:
    """Synthesize research + risk into BUY/SELL/HOLD verdicts.

    min_conviction is the bar the *executor* will apply this cycle
    (config.effective_min_conviction(regime)). Quoting the base gate here while
    the engine enforced a higher regime gate silently voided every approval.
    """
    if min_conviction is None:
        min_conviction = config.MIN_PM_CONVICTION
    research_summary = []
    for r in research_results:
        research_summary.append(
            f"- {r['ticker']}: {r['recommendation']} (conviction {r['conviction']}/5) — {r['thesis']}"
        )

    regime_section = ''
    if risk_report.get('regime_context'):
        regime_section = f"\n\nREGIME CONTEXT:\n{risk_report['regime_context']}"
    if risk_report.get('alpha_context'):
        regime_section += f"\n\nALPHA SIGNALS (PEAD / Yield Curve / Reversal Risk):\n{risk_report['alpha_context']}"

    sentiment_section = ''
    if sentiment_results:
        lines = []
        for s in sentiment_results:
            score = s.get('sentiment_score', 0.0)
            label = s.get('sentiment', 'NEUTRAL')
            summary = s.get('summary', '')
            lines.append(f"- {s['ticker']}: {label} ({score:+.2f}) — {summary}")
        sentiment_section = (
            "\n\nSENTIMENT SUMMARY (modulates conviction ±1; does NOT override BUY/SELL):\n"
            + "\n".join(lines)
        )

    prompt = f"""You are the Portfolio Manager for a systematic equity strategy.
Your role is VETO GATE only: decide BUY / SELL / HOLD and conviction for each ticker.
Position SIZING is handled mechanically by the strategy engine — do NOT set size_pct.

RESEARCH RESULTS:
{chr(10).join(research_summary)}

RISK REPORT:
- VaR 95%: {risk_report.get('var_95', 'N/A')}
- Max Drawdown: {risk_report.get('max_drawdown', 'N/A')}
- Sharpe Ratio: {risk_report.get('sharpe_ratio', 'N/A')}
- Avg Correlation: {risk_report.get('avg_correlation', 'N/A')}
- Total Exposure: {risk_report.get('total_exposure', 'N/A')}
- Status: {risk_report.get('status', 'N/A')}
- Violations: {risk_report.get('violations', [])}{regime_section}

Record your decisions with the record_decisions tool.

Rules:
- Include ALL {len(research_results)} tickers
- size_pct MUST be 0.0 for every ticker — sizing is handled by the engine
- BUY = you approve entry (engine will size it); SELL = veto / exit; HOLD = skip this cycle
- Only BUY high conviction (>={min_conviction}) stocks with strong fundamentals.
  The engine DISCARDS any BUY below {min_conviction} without trading it — a BUY you
  cannot justify at {min_conviction}+ conviction is a HOLD, not a lower-conviction BUY
- SELL any position with deteriorating fundamentals or negative risk-reward
- Respect risk constraints — SELL if violations exist{sentiment_section}"""

    decision = authorize(
        "pm", config.CLAUDE_PM_MODEL,
        est_tokens_in=est_tokens(_PM_SYSTEM[0]["text"], prompt),
        # max_tokens is a ceiling, not a reservation, but the governor prices the
        # worst case off it. 2048 was not enough for 20 tickers each with a
        # rationale string: the forced tool call got truncated mid-JSON. 8192 was
        # then hard-coded, which is the same bug one universe later — see
        # _output_ceiling.
        est_max_tokens_out=_output_ceiling(len(research_results)),
    )

    def _call_api():
        # .stream() not .create(): _output_ceiling reaches 26400 at n=120,
        # over the SDK's non-streaming >10min guard ("Streaming is required
        # for operations that may take longer than 10 minutes"). Same shape
        # as .create() via get_final_message() — nothing downstream changes.
        with client.messages.stream(
            model=decision.model,          # may arrive degraded
            max_tokens=decision.max_tokens,
            temperature=0.2,  # trading decisions: low sampling variance
            system=_PM_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            tools=[_PM_TOOL],
            tool_choice={"type": "tool", "name": "record_decisions"},
        ) as stream:
            message = stream.get_final_message()
        record_usage("pm", decision.model, message)
        return message

    if decision.tier != "FULL":
        notify_degraded("pm", decision)

    if not decision.allowed:
        print(f"  [PM] Presupuesto agotado ({decision.reason}) — usando veredictos de research.")
        result = _research_fallback(
            research_results,
            "Presupuesto LLM diario agotado; usando recomendaciones de research directamente.",
        )
    else:
        try:
            # Transport-level failures are worth two more shots (three
            # attempts total, 2s then 5s backoff); a bad *payload* is not —
            # strict tool use makes that failure deterministic, so
            # _extract_decisions below goes straight to the fallback without
            # retrying.
            message = with_retry(_call_api, "pm", max_retries=3, delays=(2, 5))
        except Exception as e:
            print(f"  [PM] API call failed after 3 attempts ({e}), using fallback.")
            message = None

        if message is None:
            result = _research_fallback(research_results, _PM_FAILED_COMMENT)
        else:
            try:
                result = _extract_decisions(message, len(research_results))
            except ValueError as e:
                print(f"  [PM] Invalid tool output ({e}) — using fallback.")
                result = _research_fallback(research_results, _PM_FAILED_COMMENT)

    for rec in result.get("recommendations", []):
        rec["size_pct"] = 0.0  # PM is veto-only — size always 0 here; engine computes it
        if rec.get("action") not in {"BUY", "SELL", "HOLD"}:
            rec["action"] = "HOLD"

    return result
