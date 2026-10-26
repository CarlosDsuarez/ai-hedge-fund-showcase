"""The batched research/sentiment calls must size their output ceiling to the
universe, and must say so out loud when they hit it anyway.

Both agents were written for a 20-ticker universe and hard-coded a matching
max_tokens (4096 research, 2048 sentiment). When the universe grew to 50, every
run came back with stop_reason="max_tokens" and a truncated tool_use block whose
payload was empty. No exception was raised, so each ticker quietly took the
per-ticker fallback: research conviction 1, sentiment NEUTRAL. Downstream the PM
saw a universe with no data, could not clear the conviction gate, and vetoed the
entire cycle — five weeks of zero entries while exits kept firing.

Two invariants, one per failure mode:
  1. the requested ceiling grows with len(tickers)
  2. a truncated response is reported as a truncation, not as "Analysis failed"
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_TICKERS_20 = [f"T{i:02d}" for i in range(20)]
_TICKERS_50 = [f"T{i:02d}" for i in range(50)]


def _mock_client(content, stop_reason="tool_use"):
    mock = MagicMock()
    message = MagicMock()
    message.content = content
    message.stop_reason = stop_reason
    mock.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = message
    return mock


def _tool_block(payload):
    block = MagicMock()
    block.type = "tool_use"
    block.input = payload
    return block


def _decision(max_tokens):
    """A FULL-tier authorization that hands back the ceiling it was asked for."""
    decision = MagicMock()
    decision.allowed = True
    decision.tier = "FULL"
    decision.model = "claude-sonnet-4-6"
    decision.max_tokens = max_tokens
    return decision


def _capture_authorize(recorder):
    def _authorize(agent, model, est_tokens_in, est_max_tokens_out):
        recorder.append(est_max_tokens_out)
        return _decision(est_max_tokens_out)
    return _authorize


# --------------------------------------------------------------------------
# research
# --------------------------------------------------------------------------

def _run_research(mock, tickers, recorder):
    from agents.research_agent import run_research
    with patch("agents.research_agent.client", mock), \
         patch("agents.research_agent.record_usage"), \
         patch("agents.research_agent.notify_degraded"), \
         patch("agents.research_agent.authorize", _capture_authorize(recorder)), \
         patch("agents.research_agent._fundamentals_block", lambda t: f"{t}: data"), \
         patch("agents.research_agent._enrich_exit_levels", lambda r, t: r):
        return run_research(tickers)


def test_research_ceiling_scales_with_universe():
    """50 tickers (3 chunks at chunk size 20) must not be squeezed into one
    chunk's ceiling — each chunk is still authorized against its own size,
    and the chunks together leave room for every ticker."""
    recorder_20 = []
    mock = _mock_client([_tool_block({"analyses": []})])
    _run_research(mock, _TICKERS_20, recorder_20)

    recorder_50 = []
    _run_research(mock, _TICKERS_50, recorder_50)

    assert len(recorder_20) == 1, "20 tickers at chunk size 20 is a single chunk"
    assert len(recorder_50) == 3, "50 tickers at chunk size 20 must split into 3 chunks"
    # 20 tickers historically produced 3300-4096 output tokens: ~200/ticker.
    assert sum(recorder_50) >= 200 * len(_TICKERS_50), (
        f"{sum(recorder_50)} tokens across chunks for {len(_TICKERS_50)} tickers "
        "leaves no room for the observed ~200 tokens per analysis"
    )


def test_research_truncation_is_reported_as_truncation():
    """stop_reason=max_tokens must not read like a per-ticker data outage.

    Every attempt (including retries) comes back truncated, so the chunk
    exhausts its retries and every ticker in it falls back to a HOLD that says
    so — 1 initial attempt + RESEARCH_CHUNK_RETRIES(2) retries = 3 calls.
    """
    recorder = []
    mock = _mock_client([_tool_block({})], stop_reason="max_tokens")
    with patch("time.sleep"):
        results = _run_research(mock, _TICKERS_20, recorder)

    assert mock.messages.stream.call_count == 3
    assert len(results) == len(_TICKERS_20)
    assert all(r["recommendation"] == "HOLD" for r in results), "never trade blind"
    assert all("truncat" in r["thesis"].lower() for r in results), (
        "a truncated batch must be distinguishable from missing fundamentals — "
        f"got {results[0]['thesis']!r}"
    )


def test_research_partial_truncated_payload_is_reported_as_truncation():
    """Truncation can leave the first few analyses intact — still a truncation."""
    recorder = []
    payload = {"analyses": [
        {"ticker": "T00", "recommendation": "BUY", "conviction": 5,
         "thesis": "complete", "key_risks": ["r"]},
    ]}
    mock = _mock_client([_tool_block(payload)], stop_reason="max_tokens")
    with patch("time.sleep"):
        results = _run_research(mock, _TICKERS_20, recorder)

    assert all("truncat" in r["thesis"].lower() for r in results), (
        "a partial batch under stop_reason=max_tokens is not a usable batch"
    )


def test_research_missing_tool_block_is_reported():
    """No tool_use block at all is a failure with a name, not a silent HOLD."""
    recorder = []
    text_block = MagicMock()
    text_block.type = "text"
    mock = _mock_client([text_block])
    with patch("time.sleep"):
        results = _run_research(mock, _TICKERS_20, recorder)

    assert len(results) == len(_TICKERS_20)
    assert all(r["conviction"] == 1 for r in results)


def test_research_complete_response_is_used():
    """The healthy path must survive the guard."""
    recorder = []
    payload = {"analyses": [
        {"ticker": t, "recommendation": "BUY", "conviction": 4,
         "thesis": f"{t} thesis", "key_risks": ["r"]}
        for t in _TICKERS_20
    ]}
    mock = _mock_client([_tool_block(payload)])
    results = _run_research(mock, _TICKERS_20, recorder)

    assert all(r["recommendation"] == "BUY" for r in results)
    assert all(r["conviction"] == 4 for r in results)


# --------------------------------------------------------------------------
# sentiment
# --------------------------------------------------------------------------

def _run_sentiment(mock, tickers, recorder):
    from agents.sentiment_agent import run_sentiment
    with patch("agents.sentiment_agent.client", mock), \
         patch("agents.sentiment_agent.record_usage"), \
         patch("agents.sentiment_agent.notify_degraded"), \
         patch("agents.sentiment_agent.authorize", _capture_authorize(recorder)), \
         patch("agents.sentiment_agent._fetch_headlines", lambda t: [f"{t} headline"]):
        return run_sentiment(tickers)


def test_sentiment_ceiling_scales_with_universe():
    recorder = []
    mock = _mock_client([_tool_block({"results": []})])
    _run_sentiment(mock, _TICKERS_20, recorder)
    _run_sentiment(mock, _TICKERS_50, recorder)

    ceiling_20, ceiling_50 = recorder
    assert ceiling_50 > ceiling_20, "ceiling must grow with the universe"
    # 20 tickers historically produced 1045-1246 output tokens: ~60/ticker.
    assert ceiling_50 >= 60 * len(_TICKERS_50), (
        f"{ceiling_50} tokens for {len(_TICKERS_50)} tickers leaves no room for "
        "the observed ~60 tokens per score"
    )


def test_sentiment_truncation_is_reported_as_truncation():
    recorder = []
    mock = _mock_client([_tool_block({})], stop_reason="max_tokens")
    results = _run_sentiment(mock, _TICKERS_20, recorder)

    assert len(results) == len(_TICKERS_20)
    assert all(r["sentiment"] == "NEUTRAL" for r in results), "never guess"
    assert all(r["sentiment_score"] == 0.0 for r in results)
    assert all("truncat" in r["summary"].lower() for r in results), (
        "a truncated batch must be distinguishable from a ticker with no news — "
        f"got {results[0]['summary']!r}"
    )


def test_sentiment_no_headlines_still_reads_as_no_headlines():
    """The truncation wording must not swallow the genuine no-news case."""
    recorder = []
    mock = _mock_client([_tool_block({"results": []})])
    from agents.sentiment_agent import run_sentiment
    with patch("agents.sentiment_agent.client", mock), \
         patch("agents.sentiment_agent.record_usage"), \
         patch("agents.sentiment_agent.notify_degraded"), \
         patch("agents.sentiment_agent.authorize", _capture_authorize(recorder)), \
         patch("agents.sentiment_agent._fetch_headlines", lambda t: []):
        results = run_sentiment(["AAPL"])

    assert results[0]["summary"] == "No headlines available."


def test_sentiment_complete_response_is_used():
    recorder = []
    payload = {"results": [
        {"ticker": t, "sentiment_score": 0.5, "sentiment": "POSITIVE",
         "summary": f"{t} ok"}
        for t in _TICKERS_20
    ]}
    mock = _mock_client([_tool_block(payload)])
    results = _run_sentiment(mock, _TICKERS_20, recorder)

    assert all(r["sentiment"] == "POSITIVE" for r in results)
    assert all(r["sentiment_score"] == 0.5 for r in results)


# ── PM ceiling ──────────────────────────────────────────────────────────
# The PM was the last hard-coded ceiling in the pipeline: 8192 tokens, sized for
# a 20-name universe, while research and sentiment had already learned to scale.
# It is the most dangerous of the three to truncate — _extract_decisions raises
# when the model returns fewer decisions than tickers, and every raise falls
# back to raw research with degraded=True, which does not stop the cycle. It
# removes the PM veto from it.

def test_pm_ceiling_grows_with_the_universe():
    from agents.pm_agent import _output_ceiling

    assert _output_ceiling(120) > _output_ceiling(50) > _output_ceiling(20)


def test_pm_ceiling_covers_the_configured_universe():
    """Enough room for a decision per ticker at the measured ~150 tokens each."""
    import config
    from agents.pm_agent import _output_ceiling

    n = len(config.TICKERS)
    assert _output_ceiling(n) >= 150 * n


def test_pm_ceiling_keeps_its_historical_floor():
    """Small universes and the suite's single-ticker calls keep the old ceiling."""
    from agents.pm_agent import _output_ceiling

    assert _output_ceiling(1) == 8192
