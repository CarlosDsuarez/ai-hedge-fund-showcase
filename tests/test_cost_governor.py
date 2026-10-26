# tests/test_cost_governor.py — pre-flight LLM budget governor
import os
import sqlite3
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents import cost_governor as cg

SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"

# sonnet worst case for (1000 in, 500 out) = 1000/1e6*3 + 500/1e6*15 = 0.0105
EST_IN, EST_OUT = 1000, 500


def setup_function(_fn):
    cg.reset_cache()


@pytest.fixture
def budget(monkeypatch):
    """Deterministic $1.00/day budget with enforcement on."""
    monkeypatch.setattr(config, "DAILY_LLM_BUDGET_USD", 1.00)
    monkeypatch.setattr(config, "BUDGET_TIER_REDUCED_PCT", 0.40)
    monkeypatch.setattr(config, "BUDGET_TIER_MINIMAL_PCT", 0.15)
    monkeypatch.setattr(config, "BUDGET_TIER_BLOCKED_PCT", 0.05)
    monkeypatch.setattr(config, "BUDGET_ENFORCEMENT", True)
    return monkeypatch


def _spend(monkeypatch, amount):
    monkeypatch.setattr(cg, "spent_today", lambda day=None: amount)


# ── the degradation ladder ──────────────────────────────────────────────

def test_full_tier_leaves_request_untouched(budget):
    _spend(budget, 0.10)  # 90% left
    d = cg.authorize("research", SONNET, EST_IN, EST_OUT)

    assert d.allowed is True
    assert d.tier == "FULL"
    assert d.model == SONNET
    assert d.max_tokens == EST_OUT


def test_full_tier_enables_thinking_when_max_tokens_allows_it(budget):
    _spend(budget, 0.10)
    d = cg.authorize("pm", SONNET, EST_IN, 4096)

    assert d.tier == "FULL"
    assert d.thinking_budget == 1024
    # a small ceiling can't host a 1024-token thinking budget
    assert cg.authorize("research", SONNET, EST_IN, 512).thinking_budget is None


def test_reduced_tier_shrinks_max_tokens_and_kills_thinking(budget):
    _spend(budget, 0.75)  # 25% left → REDUCED
    d = cg.authorize("research", SONNET, EST_IN, EST_OUT)

    assert d.allowed is True
    assert d.tier == "REDUCED"
    assert d.model == SONNET          # model not degraded at this tier
    assert d.max_tokens == 300        # 500 * 0.6
    assert d.thinking_budget is None


def test_minimal_tier_forces_haiku_and_drops_deep_dive(budget):
    _spend(budget, 0.90)  # 10% left → MINIMAL
    d = cg.authorize("pm", SONNET, EST_IN, EST_OUT)

    assert d.allowed is True
    assert d.tier == "MINIMAL"
    assert d.model == config.CLAUDE_SENTIMENT_MODEL == HAIKU
    assert d.max_tokens == 200        # 500 * 0.4
    assert d.thinking_budget is None


def test_blocked_tier_denies_the_call(budget):
    _spend(budget, 0.98)  # 2% left → BLOCKED
    d = cg.authorize("research", SONNET, EST_IN, EST_OUT)

    assert d.allowed is False
    assert d.tier == "BLOCKED"
    assert d.reason == "budget_exhausted"


def test_estimate_above_remaining_blocks_even_inside_a_healthy_tier(budget):
    _spend(budget, 0.80)  # 20% left → REDUCED by fraction…
    d = cg.authorize("pm", SONNET, 100_000, 100_000)  # …but this call costs $1.80

    assert d.allowed is False
    assert d.tier == "BLOCKED"


def test_tier_boundaries_are_inclusive_on_the_upper_side(budget):
    _spend(budget, 0.60)  # exactly 40% left → still FULL
    assert cg.authorize("research", SONNET, EST_IN, EST_OUT).tier == "FULL"

    _spend(budget, 0.85)  # exactly 15% left → still REDUCED
    assert cg.authorize("research", SONNET, EST_IN, EST_OUT).tier == "REDUCED"

    _spend(budget, 0.95)  # exactly 5% left → still MINIMAL
    assert cg.authorize("research", SONNET, EST_IN, EST_OUT).tier == "MINIMAL"


# ── observation mode ────────────────────────────────────────────────────

@pytest.mark.parametrize("spent", [0.10, 0.75, 0.90, 0.99])
def test_enforcement_disabled_never_degrades(budget, spent):
    budget.setattr(config, "BUDGET_ENFORCEMENT", False)
    _spend(budget, spent)
    d = cg.authorize("research", SONNET, EST_IN, EST_OUT)

    assert d.allowed is True
    assert d.tier == "FULL"
    assert d.model == SONNET
    assert d.max_tokens == EST_OUT
    assert d.reason == "enforcement_disabled"


# ── resilience ──────────────────────────────────────────────────────────

def test_db_failure_reports_zero_spend():
    with patch("db.database._conn", side_effect=sqlite3.OperationalError("disk gone")):
        assert cg.spent_today("2026-07-29") == 0.0


def test_db_failure_authorizes_at_full(budget):
    # spend unknown → 0.0 → whole budget appears available
    with patch("db.database._conn", side_effect=sqlite3.OperationalError("disk gone")):
        d = cg.authorize("research", SONNET, EST_IN, EST_OUT)

    assert d.allowed is True
    assert d.tier == "FULL"


def test_internal_failure_degrades_to_governor_error(budget):
    def _boom():
        raise RuntimeError("boom")

    budget.setattr(cg, "remaining_today", _boom)
    d = cg.authorize("research", SONNET, EST_IN, EST_OUT)

    assert d.allowed is True
    assert d.tier == "FULL"
    assert d.reason == "governor_error"
    assert d.model == SONNET
    assert d.max_tokens == EST_OUT


def test_zero_budget_is_treated_as_governance_off(budget):
    budget.setattr(config, "DAILY_LLM_BUDGET_USD", 0.0)
    d = cg.authorize("research", SONNET, EST_IN, EST_OUT)

    assert d.allowed is True
    assert d.reason == "budget_not_configured"


# ── estimation & DB reads ───────────────────────────────────────────────

def test_estimate_cost_is_worst_case():
    assert cg.estimate_cost(SONNET, 1_000_000, 1_000_000) == pytest.approx(18.0)
    assert cg.estimate_cost(HAIKU, 1_000_000, 1_000_000) == pytest.approx(6.0)
    # unknown model falls back to Sonnet pricing (conservative)
    assert cg.estimate_cost("mystery-model", 1_000_000, 0) == pytest.approx(3.0)


def _seed(db_file, rows, with_cost_column=False):
    """Create an llm_usage table (optionally pre-A3, i.e. without cost_usd)."""
    extra = "cost_usd REAL NOT NULL DEFAULT 0.0" if with_cost_column else "dummy INTEGER"
    value_col = "cost_usd" if with_cost_column else "dummy"
    conn = sqlite3.connect(db_file)
    conn.execute(f"""
        CREATE TABLE llm_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, agent TEXT NOT NULL, model TEXT NOT NULL,
            tokens_in INTEGER NOT NULL, tokens_out INTEGER NOT NULL,
            cache_read INTEGER NOT NULL DEFAULT 0, {extra}
        )
    """)
    conn.executemany(
        f"INSERT INTO llm_usage (ts, agent, model, tokens_in, tokens_out, {value_col})"
        f" VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def test_spent_today_sums_cost_usd_column(tmp_path, monkeypatch):
    import db.database as db

    dbf = str(tmp_path / "t.db")
    _seed(dbf, [
        ("2026-07-29T09:00:00", "research", SONNET, 1000, 500, 0.02),
        ("2026-07-29T10:00:00", "pm", SONNET, 1000, 500, 0.03),
        ("2026-07-28T10:00:00", "pm", SONNET, 1000, 500, 9.99),  # other day
    ], with_cost_column=True)
    monkeypatch.setattr(db, "DB_PATH", dbf)

    assert cg.spent_today("2026-07-29") == pytest.approx(0.05)


def test_spent_today_derives_cost_when_column_absent(tmp_path, monkeypatch):
    """Pre-A3 schema: reconstruct spend from tokens instead of reporting $0."""
    import db.database as db

    dbf = str(tmp_path / "t.db")
    _seed(dbf, [
        ("2026-07-29T09:00:00", "research", SONNET, 1_000_000, 0, 0),
        ("2026-07-29T10:00:00", "sentiment", HAIKU, 1_000_000, 0, 0),
    ])
    monkeypatch.setattr(db, "DB_PATH", dbf)

    assert cg.spent_today("2026-07-29") == pytest.approx(4.0)  # 3.0 sonnet + 1.0 haiku


def test_spent_today_is_cached_for_60s(tmp_path, monkeypatch):
    import db.database as db

    dbf = str(tmp_path / "t.db")
    _seed(dbf, [("2026-07-29T09:00:00", "research", SONNET, 1_000_000, 0, 0.5)],
          with_cost_column=True)
    monkeypatch.setattr(db, "DB_PATH", dbf)

    assert cg.spent_today("2026-07-29") == pytest.approx(0.5)

    conn = sqlite3.connect(dbf)
    conn.execute("INSERT INTO llm_usage (ts, agent, model, tokens_in, tokens_out, cost_usd)"
                 " VALUES ('2026-07-29T11:00:00', 'pm', ?, 0, 0, 10.0)", (SONNET,))
    conn.commit()
    conn.close()

    assert cg.spent_today("2026-07-29") == pytest.approx(0.5)   # served from cache
    cg.reset_cache()
    assert cg.spent_today("2026-07-29") == pytest.approx(10.5)  # cache dropped → fresh read


def test_remaining_today_can_go_negative(budget):
    _spend(budget, 2.50)
    assert cg.remaining_today() == pytest.approx(-1.50)
