import numpy as np
import pandas as pd
import pytest

import config

from agents import regime_agent
from agents.regime_agent import detect_regime, _default_result


def test_detect_regime_returns_vix_fields(synthetic_regime_market):
    result = detect_regime()
    for field in ("vix", "vol_regime", "size_multiplier", "credit_signal"):
        assert field in result, f"detect_regime() missing '{field}' field"


def test_vix_field_is_positive_float(synthetic_regime_market):
    result = detect_regime()
    assert isinstance(result["vix"], float), "vix should be a float"
    assert result["vix"] > 0, "vix should be positive"


def test_vol_regime_values(synthetic_regime_market):
    result = detect_regime()
    assert result["vol_regime"] in ("LOW", "MED", "HIGH", "EXTREME"), (
        f"vol_regime must be LOW/MED/HIGH/EXTREME, got: {result['vol_regime']}"
    )


def test_size_multiplier_matches_vol_regime(monkeypatch):
    """size_multiplier is the direction x breadth x VIX product — not a function
    of vol_regime alone. A BULL regime with strong breadth (70% of the universe
    above its own SMA200) and a VIX between the elevated and extreme thresholds
    sizes at breadth-strong x VIX-elevated, with a direction multiplier of 1.0."""
    vix_level = (config.VIX_ELEVATED + config.VIX_EXTREME) / 2
    n = 260
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = np.linspace(300, 500, n)
    cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Open"], ["SPY", "^VIX"]])
    raw = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for f in ("Close", "High", "Low", "Open"):
        raw[(f, "SPY")] = close + (1.0 if f == "High" else -1.0 if f == "Low" else 0.0)
        raw[(f, "^VIX")] = vix_level

    universe_idx = pd.bdate_range("2024-01-01", periods=260)
    up = {f"UP{i}": np.linspace(100, 200, 260) for i in range(28)}
    down = {f"DN{i}": np.linspace(200, 100, 260) for i in range(12)}
    universe_closes = pd.DataFrame({**up, **down}, index=universe_idx)

    monkeypatch.setattr(regime_agent.yf, "download", lambda *a, **k: raw)
    monkeypatch.setattr(regime_agent, "_fetch_universe_closes", lambda: universe_closes)
    monkeypatch.setattr(regime_agent, "_load_cache", lambda: {})
    monkeypatch.setattr(regime_agent, "_save_cache", lambda *a, **k: None)

    result = regime_agent.detect_regime()
    assert result["regime"] == "BULL"
    assert result["breadth_pct"] == pytest.approx(0.70)
    assert result["vix"] == pytest.approx(vix_level)
    assert result["vol_regime"] == "HIGH"
    assert result["size_multiplier"] == pytest.approx(0.75)


def test_default_result_has_vix_fields():
    d = _default_result()
    assert d["vix"] == 20.0
    assert d["vol_regime"] == "MED"
    assert d["size_multiplier"] == 1.0
    assert d["credit_signal"] == "NORMAL"
