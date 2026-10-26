"""Showcase reference implementations: synthetic data, stub LLM, 12-1 momentum."""
import pandas as pd
import pytest

import config
from data import synthetic
from utils import llm


@pytest.fixture
def synthetic_mode(monkeypatch):
    monkeypatch.setenv("DATA_MODE", "synthetic")


# ── synthetic data ──────────────────────────────────────────────────────

def test_synthetic_prices_are_deterministic_and_well_formed(synthetic_mode):
    a = synthetic.ohlcv("AAPL", "1y")
    b = synthetic.ohlcv("AAPL", "1y")
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert (a["High"] >= a[["Open", "Close"]].max(axis=1)).all()
    assert (a["Low"] <= a[["Open", "Close"]].min(axis=1)).all()
    assert len(a) > 200


def test_fetcher_routes_to_synthetic_without_network(synthetic_mode):
    from data import fetcher
    closes = fetcher.get_closes(["AAPL", "MSFT"], period="2y")
    assert set(closes.columns) == {"AAPL", "MSFT"}
    info = fetcher.get_stock_info("AAPL")
    assert info["name"].endswith("(synthetic)") and info["price"] > 0


# ── reference strategy ──────────────────────────────────────────────────

def test_momentum_is_twelve_minus_one():
    from agents.strategy_agent import _compute_momentum
    n = config.MOMENTUM_WINDOW + 1
    closes = pd.DataFrame({"UP": range(1, n + 1), "FLAT": [10.0] * n}, dtype=float)
    mom = _compute_momentum(closes)
    expected_up = closes["UP"].iloc[-config.MOMENTUM_SKIP - 1] / closes["UP"].iloc[0] - 1
    assert mom["UP"] == pytest.approx(expected_up)
    assert mom["FLAT"] == pytest.approx(0.0)


def test_generate_signals_contract(synthetic_mode):
    from agents.strategy_agent import generate_signals
    out = generate_signals(config.TICKERS)
    assert set(out) == set(config.TICKERS)
    keys = {"signal", "composite_score", "mom_z", "val_z", "sent_z", "macro_stressed",
            "credit_spread_z", "strategy", "vwap_signal", "mom_signal"}
    for row in out.values():
        assert keys <= set(row)
        assert row["signal"] in ("BUY", "HOLD")
    assert any(r["signal"] == "BUY" for r in out.values())


def test_regime_signals_mark_top_n(synthetic_mode, tmp_path, monkeypatch):
    from agents import regime_strategy_agent as rsa
    monkeypatch.setattr(rsa, "LOGS_DIR", str(tmp_path))
    out = rsa.generate_regime_signals("BULL")
    buys = [t for t, r in out.items() if r["signal"] == "BUY"]
    assert len(buys) == rsa.TOP_N_BUY
    assert all(out[t]["rank"] <= rsa.TOP_N_BUY for t in buys)


# ── stub LLM ────────────────────────────────────────────────────────────

def test_stub_llm_answers_forced_tool_call_per_ticker(monkeypatch):
    monkeypatch.setenv("LLM_MODE", "stub")
    from agents.research_agent import _RESEARCH_TOOL, _extract_analyses
    client = llm.get_client()
    prompt = "Analyze:\nAAPL: price 1\nMSFT: price 2\n"
    with client.messages.stream(model="x", max_tokens=10, messages=[{"role": "user", "content": prompt}],
                                tools=[_RESEARCH_TOOL],
                                tool_choice={"type": "tool", "name": "record_research"}) as s:
        msg = s.get_final_message()
    analyses = _extract_analyses(msg)
    assert [a["ticker"] for a in analyses] == ["AAPL", "MSFT"]
    assert all(a["recommendation"] in ("BUY", "SELL", "HOLD") for a in analyses)


# ── reference backtest ──────────────────────────────────────────────────

def test_reference_backtest_runs_on_synthetic_data(synthetic_mode):
    from backtest import reference_backtest as rb
    out = rb.run(config.TICKERS[:8], "2024-01-01", "2024-12-31")
    eq = out["equity"]
    assert len(eq) > 200 and eq.notna().all() and (eq > 0).all()
    assert set(out["metrics"]) == {"cagr", "ann_vol", "sharpe", "max_drawdown"}
    assert out["metrics"]["max_drawdown"] <= 0
