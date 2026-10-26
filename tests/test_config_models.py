import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_claude_research_model_in_config():
    import config
    assert hasattr(config, "CLAUDE_RESEARCH_MODEL"), "CLAUDE_RESEARCH_MODEL missing from config.py"
    assert "claude" in config.CLAUDE_RESEARCH_MODEL

def test_claude_sentiment_model_in_config():
    import config
    assert hasattr(config, "CLAUDE_SENTIMENT_MODEL"), "CLAUDE_SENTIMENT_MODEL missing from config.py"
    assert "claude" in config.CLAUDE_SENTIMENT_MODEL
