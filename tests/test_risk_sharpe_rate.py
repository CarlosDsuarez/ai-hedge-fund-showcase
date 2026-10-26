import pandas as pd
import numpy as np
import config
from agents.risk_agent import _calculate_sharpe


def test_sharpe_uses_config_risk_free_rate():
    np.random.seed(42)
    daily_rf_config = config.RISK_FREE_ANNUAL / 252
    returns = pd.Series(np.random.normal(0.001, 0.01, 252))

    expected_sharpe = float(
        (returns - daily_rf_config).mean() / (returns - daily_rf_config).std() * (252 ** 0.5)
    )
    actual_sharpe = _calculate_sharpe(returns)

    assert abs(actual_sharpe - expected_sharpe) < 1e-6, (
        f"Sharpe used wrong rate. Expected ~{expected_sharpe:.4f}, got {actual_sharpe:.4f}"
    )
