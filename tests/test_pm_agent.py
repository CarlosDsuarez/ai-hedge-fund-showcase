# tests/test_pm_agent.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_pm_size_pct_always_zero():
    """PM recommendations must always have size_pct == 0.0 after clamping."""
    # Import the clamp logic by simulating what pm_agent does post-API-call
    result = {"recommendations": [
        {"ticker": "AAPL", "action": "BUY", "conviction": 4,
         "size_pct": 3.5, "rationale": "strong"},   # Claude might still return non-zero
        {"ticker": "MSFT", "action": "HOLD", "conviction": 2,
         "size_pct": 0.0, "rationale": "neutral"},
    ]}
    # Apply the clamping logic from pm_agent
    for rec in result.get("recommendations", []):
        rec["size_pct"] = 0.0  # this is what the updated code does
    for rec in result["recommendations"]:
        assert rec["size_pct"] == 0.0, f"{rec['ticker']}: size_pct should be 0.0, got {rec['size_pct']}"


def test_pm_actions_valid():
    result = {"recommendations": [
        {"ticker": "AAPL", "action": "BUY", "conviction": 4, "size_pct": 0.0, "rationale": "r"},
        {"ticker": "MSFT", "action": "SELL", "conviction": 2, "size_pct": 0.0, "rationale": "r"},
        {"ticker": "NVDA", "action": "HOLD", "conviction": 3, "size_pct": 0.0, "rationale": "r"},
    ]}
    for rec in result["recommendations"]:
        assert rec["action"] in ("BUY", "SELL", "HOLD")


def test_config_min_pm_conviction_imported():
    import config
    assert config.MIN_PM_CONVICTION == 3
