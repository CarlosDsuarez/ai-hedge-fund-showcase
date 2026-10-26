import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.llm import get_client
from utils.env import load_project_env

import config
from config import TICKERS
from agents.cost_governor import authorize, est_tokens, notify_degraded
from agents.llm_usage import record_usage

load_project_env()

client = get_client()  # real Anthropic client, or the offline stub when LLM_MODE=stub

_SENTIMENT_SYSTEM = [{
    "type": "text",
    "text": (
        "You are a financial news sentiment analyst. "
        "Analyze headlines and record the sentiment for every ticker with the "
        "record_sentiment tool: score (-1.0 to 1.0), label and a one-sentence summary."
    ),
    # NOTE: no cache_control — prompt (~55 tok) is far below Haiku's 4096-token
    # minimum cacheable prefix; a marker here never caches. See research_agent.
}]

_SENTIMENT_TOOL = {
    "name": "record_sentiment",
    "description": "Record the news sentiment analysis for every ticker.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "sentiment_score": {"type": "number"},
                        "sentiment": {"type": "string",
                                      "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"]},
                        "summary": {"type": "string"},
                    },
                    "required": ["ticker", "sentiment_score", "sentiment", "summary"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}


# One result is a score, a label and a one-sentence summary. Twenty of them
# measured 1045-1246 output tokens across July — ~60 each — so 110 leaves most
# of a result's worth of headroom per ticker. The floor keeps small universes
# (and the single-ticker calls in the test suite) on the ceiling this call has
# always used.
_TOKENS_PER_RESULT = 110
_MIN_OUTPUT_TOKENS = 2048
_MAX_OUTPUT_TOKENS = 16000


def _output_ceiling(n_tickers: int) -> int:
    """Output ceiling for a batch of `n_tickers`. See research_agent._output_ceiling."""
    return max(_MIN_OUTPUT_TOKENS,
               min(_MAX_OUTPUT_TOKENS, _TOKENS_PER_RESULT * n_tickers))


def _extract_results(message) -> list:
    """Pull the record_sentiment payload out of `message`, or raise ValueError saying why not.

    A forced tool call that hits the token ceiling comes back as
    stop_reason="max_tokens" with a TRUNCATED tool_use block and no exception.
    Read naively that looks exactly like a universe with no news — a
    tradeable-looking neutral rather than the outage it actually is.
    """
    if getattr(message, "stop_reason", None) == "max_tokens":
        raise ValueError("truncated: stop_reason=max_tokens (raise the output ceiling)")
    try:
        block = next(b for b in message.content if b.type == "tool_use")
    except StopIteration:
        raise ValueError("no tool_use block in response") from None
    payload = block.input
    if not isinstance(payload, dict):
        raise ValueError(f"tool input is {type(payload).__name__}, not dict")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("results missing or not a list")
    return results


def _fetch_headlines(ticker: str) -> list:
    from data import synthetic
    if synthetic.enabled():
        return synthetic.headlines(ticker)
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        headlines = []
        for item in root.findall(".//item"):
            title = item.find("title")
            if title is not None and title.text:
                headlines.append(title.text.strip())
            if len(headlines) >= 5:
                break
        return headlines
    except Exception:
        return []


def _neutral(ticker: str, summary: str) -> dict:
    """Zero-signal result: sentiment must never guess when it cannot read."""
    return {
        "ticker": ticker,
        "sentiment_score": 0.0,
        "sentiment": "NEUTRAL",
        "summary": summary,
    }


def run_sentiment(tickers: list = None) -> list:
    if tickers is None:
        tickers = TICKERS

    # Headlines come from Yahoo RSS — keep the parallel fetch; only the LLM
    # scoring is batched into a single call (20 requests -> 1).
    headlines_map: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_headlines, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                headlines_map[ticker] = future.result()
            except Exception as e:
                print(f"  [Sentiment] Headlines error on {ticker}: {e}")
                headlines_map[ticker] = []

    results_map = {
        t: _neutral(t, "No headlines available.")
        for t in tickers if not headlines_map.get(t)
    }
    with_news = [t for t in tickers if headlines_map.get(t)]

    if with_news:
        sections = []
        for t in with_news:
            headlines_text = "\n".join(f"- {h}" for h in headlines_map[t])
            sections.append(f"{t}:\n{headlines_text}")
        prompt = (
            f"Analyze the sentiment of the news headlines for these {len(with_news)} tickers:\n\n"
            + "\n\n".join(sections)
            + f"\n\nRecord the sentiment with the record_sentiment tool. "
            f"Include ALL {len(with_news)} tickers, in the same order."
        )

        # One authorization for the whole batch — see the note in research_agent.
        decision = authorize(
            "sentiment", config.CLAUDE_SENTIMENT_MODEL,
            est_tokens_in=est_tokens(_SENTIMENT_SYSTEM[0]["text"], prompt),
            est_max_tokens_out=_output_ceiling(len(with_news)),
        )
        if decision.tier != "FULL":
            notify_degraded("sentiment", decision)

        if not decision.allowed:
            print(f"  [Sentiment] Presupuesto agotado ({decision.reason}) — fallback NEUTRAL para {len(with_news)} tickers.")
            for t in with_news:
                results_map[t] = _neutral(t, "Presupuesto LLM diario agotado; sentimiento no evaluado.")
            return [results_map[t] for t in tickers]

        try:
            # .stream() not .create(): stays under the SDK's non-streaming
            # >10min guard today (13200 tokens at n=120) but the same trap
            # research_agent and pm_agent hit is only a universe or thinking
            # budget away — same fix here before it lands here too.
            with client.messages.stream(
                model=decision.model,          # may arrive degraded
                max_tokens=decision.max_tokens,
                temperature=0.2,  # trading decisions: low sampling variance
                system=_SENTIMENT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                tools=[_SENTIMENT_TOOL],
                tool_choice={"type": "tool", "name": "record_sentiment"},
            ) as stream:
                message = stream.get_final_message()
            record_usage("sentiment", decision.model, message)
            scored = _extract_results(message)
        except Exception as e:
            print(f"  [Sentiment] Batched call failed ({e}) — NEUTRAL fallback for "
                  f"{len(with_news)} tickers.")
            for t in with_news:
                results_map[t] = _neutral(t, f"Sentiment batch unusable: {e}")
            return [results_map[t] for t in tickers]

        by_ticker = {r.get("ticker"): r for r in scored if isinstance(r, dict)}

        missing = [t for t in with_news if t not in by_ticker]
        if missing:
            # A response that stopped cleanly but skipped tickers is a smaller
            # version of the same failure, and it deserves one line that states
            # the scale rather than N lines that bury it.
            print(f"  [Sentiment] {len(missing)}/{len(with_news)} tickers missing from the "
                  f"batched response — NEUTRAL fallback for: {', '.join(missing)}")

        for ticker in with_news:
            result = by_ticker.get(ticker)
            if not result:
                results_map[ticker] = _neutral(ticker, "Ticker absent from the sentiment batch.")
                continue
            result["ticker"] = ticker
            try:
                result["sentiment_score"] = max(-1.0, min(1.0, float(result.get("sentiment_score", 0.0))))
            except (TypeError, ValueError):
                result["sentiment_score"] = 0.0
            if result.get("sentiment") not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
                result["sentiment"] = "NEUTRAL"
            result.setdefault("summary", "")
            results_map[ticker] = result

    return [results_map[t] for t in tickers]
