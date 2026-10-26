import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agents.rotation_agent as rotation_agent
from agents.rotation_agent import prioritize_deployments, classify_buy_outcome


def test_weekly_rotation_is_retired():
    """Design C: rank decay in the exit engine replaces the Monday laggard sweep."""
    assert not hasattr(rotation_agent, "is_rotation_day")
    assert not hasattr(rotation_agent, "select_rotation_exits")




# (d) freed cash deployed to highest-score PENDING first, then new by score
def test_prioritize_deployments_pending_first_then_new_by_score():
    pending = [
        {"ticker": "Q1", "composite_score": 0.50, "conviction": 4},
        {"ticker": "Q2", "composite_score": 0.90, "conviction": 5},
    ]
    new = ["N1", "N2"]
    scores = {"N1": 0.99, "N2": 0.10}                     # N1 outscores everything
    order = prioritize_deployments(pending, new, scores)
    assert [d["ticker"] for d in order] == ["Q2", "Q1", "N1", "N2"]
    assert order[0]["source"] == "pending"               # pending class wins regardless
    assert order[0]["ticker"] == "Q2"                    # highest-score pending first


# (a) budget-only skip → QUEUE ; (b) policy veto → REJECT (never queued)
def test_classify_buy_outcome():
    assert classify_buy_outcome(gates_passed=True, fundable=False) == "QUEUE"
    assert classify_buy_outcome(gates_passed=True, fundable=True) == "EXECUTE"
    assert classify_buy_outcome(gates_passed=False, fundable=True) == "REJECT"
    assert classify_buy_outcome(gates_passed=False, fundable=False) == "REJECT"
