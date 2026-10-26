import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.llm import get_client
from utils.env import load_project_env

from data.fetcher import get_stock_info, get_atr, get_stock_price
from config import TICKERS
import config
from agents.cost_governor import authorize, est_tokens, notify_degraded
from agents.llm_usage import record_usage
from utils.retry import with_retry

load_project_env()

client = get_client()  # real Anthropic client, or the offline stub when LLM_MODE=stub

# Snapshot of the most recent run_research() call — same pattern as
# main.LAST_BUDGET. main._collect_health_issues and main.run_signals read
# this to report when enough chunks fell back to HOLD that the run should be
# considered degraded, without threading a return value through every caller.
LAST_RESEARCH_STATS: dict = {}

# record_usage() writes one row via db.database.save_llm_usage, which opens
# its own sqlite3 connection per call (timeout=10, WAL). WAL already lets
# concurrent writers retry against each other for up to that timeout, and a
# single-row INSERT is fast enough that contention among RESEARCH_MAX_WORKERS
# threads is very unlikely to matter in practice — but this call site is new
# (research used to be one call, now it is up to RESEARCH_MAX_WORKERS at
# once), so this lock removes any doubt for the cost of serializing a few
# fast inserts rather than betting reliability on SQLite's busy-timeout retry.
_usage_lock = threading.Lock()

_RESEARCH_SYSTEM = [{
    "type": "text",
    "text": (
        "You are a professional equity research analyst. "
        "Analyze stocks based on fundamental data and record your analysis for "
        "every ticker with the record_research tool: recommendation (BUY/SELL/HOLD), "
        "conviction (1-5), a 2-3 sentence thesis and three key risks."
    ),
    # NOTE: no cache_control — Anthropic's minimum cacheable prefix is 2048
    # tokens (Sonnet) / 4096 (Haiku); this prompt is ~80 tokens, so a marker
    # here never caches anything. Re-add only if the static prefix grows past
    # the minimum (verify with response.usage.cache_creation_input_tokens > 0).
}]

_RESEARCH_TOOL = {
    "name": "record_research",
    "description": "Record the structured research analysis for every ticker.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "analyses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "recommendation": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                        "conviction": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                        "thesis": {"type": "string"},
                        "key_risks": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["ticker", "recommendation", "conviction", "thesis", "key_risks"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["analyses"],
        "additionalProperties": False,
    },
}


# One analysis is a recommendation, a conviction, a 2-3 sentence thesis and
# three risks. Twenty of them measured 3300-4096 output tokens across July —
# ~200 each — so 320 leaves half again as much headroom per ticker. The floor
# keeps small universes (and the single-ticker calls in the test suite) on the
# ceiling this call has always used.
_TOKENS_PER_ANALYSIS = 320
_MIN_OUTPUT_TOKENS = 4096
_MAX_OUTPUT_TOKENS = 32000


def _output_ceiling(n_tickers: int) -> int:
    """Output ceiling for a batch of `n_tickers`.

    This was a hard-coded 4096 sized for a 20-ticker universe. When the universe
    grew to 50 the call hit the ceiling on every run: stop_reason=max_tokens, a
    truncated tool_use payload, and every ticker silently downgraded to a
    conviction-1 HOLD that could never clear the entry gate.
    """
    return max(_MIN_OUTPUT_TOKENS,
               min(_MAX_OUTPUT_TOKENS, _TOKENS_PER_ANALYSIS * n_tickers))


def _extract_analyses(message) -> list:
    """Pull the record_research payload out of `message`, or raise ValueError saying why not.

    A forced tool call that hits the token ceiling comes back as
    stop_reason="max_tokens" with a TRUNCATED tool_use block — no exception, no
    error, just a payload that is empty or cut off mid-array. Reading that as
    "the model had nothing to say about these tickers" is what turned a sizing
    bug into five weeks of vetoed cycles. Every shape that is not a complete
    response has to become an explicit failure here.
    """
    if getattr(message, "stop_reason", None) == "max_tokens":
        raise ValueError("truncated: stop_reason=max_tokens (raise the output ceiling)")
    try:
        block = next(b for b in message.content if b.type == "tool_use")
    except StopIteration:
        # StopIteration would also be swallowed by `except Exception`, but only
        # by accident, and it reports no useful reason. Name it.
        raise ValueError("no tool_use block in response") from None
    result = block.input
    if not isinstance(result, dict):
        raise ValueError(f"tool input is {type(result).__name__}, not dict")
    analyses = result.get("analyses")
    if not isinstance(analyses, list):
        raise ValueError("analyses missing or not a list")
    return analyses


def _fundamentals_block(ticker: str) -> str:
    info = get_stock_info(ticker)
    return (
        f'{ticker} ({info.get("name", ticker)}):\n'
        f'- Sector: {info.get("sector", "N/A")}\n'
        f'- P/E Ratio: {info.get("pe_ratio", "N/A")}\n'
        f'- ROE: {info.get("roe", "N/A")}\n'
        f'- Revenue Growth: {info.get("revenue_growth", "N/A")}\n'
        f'- Market Cap: {info.get("market_cap", "N/A")}\n'
        f'- Free Cash Flow: {info.get("free_cashflow", "N/A")}'
    )


def _hold_fallback(ticker: str, thesis: str = "Analysis failed.") -> dict:
    """Deterministic verdict when no analysis is available — never trade blind."""
    return {
        "ticker": ticker,
        "recommendation": "HOLD",
        "conviction": 1,
        "thesis": thesis,
        "key_risks": ["Data unavailable"],
    }


def _enrich_exit_levels(result: dict, ticker: str) -> dict:
    """ATR14 and the reference price the executor sizes and stops against.

    sl_price is the exit engine's initial stop (config.STOP_ATR_MULT × ATR14
    below the reference) so the PM prompt and the chat quote the level the
    position will actually carry. The TP levels are gone with the ladder;
    the keys stay None because the signals table still has the columns.
    """
    try:
        atr_14 = get_atr(ticker, period=14)
        price_hist = get_stock_price(ticker, period="1d")
        current_price = float(price_hist["Close"].iloc[-1]) if not price_hist.empty else 0.0
        result["atr_14"]    = round(atr_14, 4)
        result["entry_ref"] = round(current_price, 2)
        if atr_14 > 0 and current_price > 0:
            result["sl_price"] = round(current_price - config.STOP_ATR_MULT * atr_14, 2)
        else:
            result["sl_price"] = None
    except Exception:
        result["atr_14"] = result["sl_price"] = None
    result["tp1_price"] = result["tp2_price"] = result["tp3_price"] = None
    return result


def _build_prompt(chunk: list, blocks: dict) -> str:
    """The research prompt for one chunk of tickers."""
    return (
        f"Analyze the following {len(chunk)} stocks as investment opportunities.\n\n"
        "Fundamental data:\n\n"
        + "\n\n".join(blocks[t] for t in chunk)
        + f"\n\nRecord your analysis with the record_research tool. "
        f"Include ALL {len(chunk)} tickers, in the same order."
    )


def _call_chunk(chunk: list, prompt: str, decision, attempt: int) -> list:
    """One .stream() call for `chunk`. Raises on any unusable response.

    max_tokens grows with `attempt` (0, 1, 2, ...) so a retry after a
    stop_reason=max_tokens truncation gets more headroom instead of hitting
    the same ceiling again — capped at _MAX_OUTPUT_TOKENS regardless of how
    degraded the governor's starting ceiling already was.
    """
    max_tokens = min(int(decision.max_tokens * (1 + 0.5 * attempt)), _MAX_OUTPUT_TOKENS)
    # .stream() not .create(): the SDK refuses a non-streaming call once
    # max_tokens is high enough to estimate >10min, which _output_ceiling
    # reaches at n=20 chunks near the cap — "Streaming is required for
    # operations that may take longer than 10 minutes." get_final_message()
    # blocks and returns the same accumulated Message .create() would, so
    # nothing downstream changes shape.
    with client.messages.stream(
        model=decision.model,          # may arrive degraded (MINIMAL forces Haiku)
        max_tokens=max_tokens,
        temperature=0.2,  # trading decisions: low sampling variance
        system=_RESEARCH_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        tools=[_RESEARCH_TOOL],
        tool_choice={"type": "tool", "name": "record_research"},
    ) as stream:
        message = stream.get_final_message()
    with _usage_lock:
        record_usage("research", decision.model, message)
    return _extract_analyses(message)


def _run_chunk(chunk: list, blocks: dict) -> tuple:
    """Authorize and run one chunk end to end. Returns (analyses, fallback).

    `fallback` is True whenever this chunk's tickers are hold-fallbacks
    rather than real analyses — either the governor blocked the chunk, or
    every retry came back unusable. Either way only THIS chunk's tickers are
    affected; the rest of the universe is unaffected.

    authorize()/notify_degraded() sit INSIDE the guard: the real
    cost_governor swallows its own failures, but a bug there used to escape
    this function, and run_research re-raised it — aborting the whole daily
    run in main.run_signals, exits and the broker stop audit included. A
    governor exception is now the same per-chunk HOLD fallback as an
    unusable response.
    """
    prompt = _build_prompt(chunk, blocks)
    attempt = 0

    try:
        decision = authorize(
            "research", config.CLAUDE_RESEARCH_MODEL,
            est_tokens_in=est_tokens(_RESEARCH_SYSTEM[0]["text"], prompt),
            est_max_tokens_out=_output_ceiling(len(chunk)),
        )
        if decision.tier != "FULL":
            # Deduped per (agent, day, tier) in cost_governor — calling this
            # once per chunk (up to RESEARCH_MAX_WORKERS times per run) still
            # sends at most one Telegram message per tier step, not one per
            # chunk.
            notify_degraded("research", decision)

        if not decision.allowed:
            print(f"  [Research] Presupuesto agotado ({decision.reason}) — "
                  f"fallback HOLD para {len(chunk)} tickers.")
            return (
                [_hold_fallback(t, "Presupuesto LLM diario agotado; sin análisis fundamental.")
                 for t in chunk],
                True,
            )

        def _attempt():
            nonlocal attempt
            try:
                return _call_chunk(chunk, prompt, decision, attempt)
            finally:
                attempt += 1

        analyses = with_retry(
            _attempt,
            f"research chunk [{chunk[0]}..{chunk[-1]}]",
            max_retries=1 + config.RESEARCH_CHUNK_RETRIES,
            delays=(2, 5),
        )
        return analyses, False
    except Exception as e:
        print(f"  [Research] Chunk unusable ({e}) — HOLD fallback for {len(chunk)} tickers.")
        return [_hold_fallback(t, f"Research chunk unusable: {e}") for t in chunk], True


def run_research(tickers: list = None) -> list:
    if tickers is None:
        tickers = TICKERS

    # Reset immediately, not only on a successful return: LAST_RESEARCH_STATS
    # otherwise keeps whatever the PREVIOUS run_research() call left behind if
    # something here raises before the final update at the bottom (the
    # fundamentals fetch, say) — stale, falsely-healthy stats for a caller
    # that only checks this dict.
    n = len(tickers)
    LAST_RESEARCH_STATS.clear()
    LAST_RESEARCH_STATS.update({
        "tickers": n,
        "chunks": 0,
        "chunks_failed": 0,
        "fallback_tickers": 0,
        "degraded": False,
    })

    # Fundamentals come from yfinance — keep the parallel fetch; only the LLM
    # analysis is chunked (RESEARCH_CHUNK_SIZE tickers per call).
    blocks: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fundamentals_block, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                blocks[ticker] = future.result()
            except Exception as e:
                print(f"  [Research] Fundamentals error on {ticker}: {e}")
                blocks[ticker] = f"{ticker}:\n- No fundamental data available"

    chunks = [tickers[i:i + config.RESEARCH_CHUNK_SIZE]
              for i in range(0, len(tickers), config.RESEARCH_CHUNK_SIZE)]
    LAST_RESEARCH_STATS["chunks"] = len(chunks)

    # One failed/blocked chunk used to mean one failed/blocked universe: a
    # single call carried all 120 tickers, so a single bad response HOLDed
    # everything. Each chunk is now authorized, called and retried on its own
    # — a bad chunk only HOLDs the tickers inside it.
    chunk_analyses: dict = {}
    chunk_fallback: dict = {}
    forced_degraded = False
    try:
        with ThreadPoolExecutor(max_workers=config.RESEARCH_MAX_WORKERS) as executor:
            futures = {executor.submit(_run_chunk, chunk, blocks): i
                       for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    analyses, fallback = future.result()
                except Exception as e:
                    # _run_chunk guards the governor and the call itself; this
                    # is something unmodeled escaping that guard. Still only
                    # this chunk's tickers — never the whole run.
                    print(f"  [Research] Chunk {i} failed outside its guard ({e}) — "
                          f"HOLD fallback for {len(chunks[i])} tickers.")
                    analyses = [_hold_fallback(t, f"Research chunk failed: {e}")
                                for t in chunks[i]]
                    fallback = True
                chunk_analyses[i] = analyses
                chunk_fallback[i] = fallback
    except Exception as e:
        # The pool itself broke. Research must never abort the daily run
        # (main.run_signals has no guard, and the exits + broker stop audit
        # come after it): every chunk still missing becomes a HOLD fallback,
        # and the run is degraded whatever the fallback ratio says.
        print(f"  [Research] Chunk pool failed ({e}) — HOLD fallback for every "
              f"chunk not yet returned; run marked degraded.")
        forced_degraded = True

    for i, chunk in enumerate(chunks):
        if i not in chunk_analyses:
            chunk_analyses[i] = [_hold_fallback(t, "Research chunk never returned.")
                                 for t in chunk]
            chunk_fallback[i] = True

    all_analyses = [a for i in range(len(chunks)) for a in chunk_analyses[i]]
    by_ticker = {a.get("ticker"): a for a in all_analyses if isinstance(a, dict)}

    missing = [t for t in tickers if t not in by_ticker]
    if missing:
        # A response that stopped cleanly but skipped tickers is a smaller
        # version of the same failure, and it deserves one line that states the
        # scale rather than N lines that bury it.
        print(f"  [Research] {len(missing)}/{len(tickers)} tickers missing from the "
              f"batched response — HOLD fallback for: {', '.join(missing)}")

    valid_recs = {"BUY", "SELL", "HOLD"}
    results = []
    # Rebuilt in the ORIGINAL ticker order regardless of chunk completion
    # order (as_completed above is unordered) or in-chunk response order.
    for ticker in tickers:
        result = by_ticker.get(ticker)
        if not result:
            results.append(_hold_fallback(ticker, "Ticker absent from the research batch."))
            continue
        result["ticker"] = ticker
        if result.get("recommendation") not in valid_recs:
            result["recommendation"] = "HOLD"
        try:
            result["conviction"] = max(1, min(5, int(result.get("conviction", 3))))
        except (TypeError, ValueError):
            result["conviction"] = 3
        results.append(_enrich_exit_levels(result, ticker))

    chunks_failed = sum(1 for f in chunk_fallback.values() if f)
    fallback_tickers = sum(len(chunks[i]) for i, f in chunk_fallback.items() if f)
    degraded = (fallback_tickers / n) > config.RESEARCH_DEGRADED_FALLBACK_PCT if n else False
    LAST_RESEARCH_STATS.update({
        "tickers": n,
        "chunks": len(chunks),
        "chunks_failed": chunks_failed,
        "fallback_tickers": fallback_tickers,
        "degraded": degraded or forced_degraded,
    })

    return results
