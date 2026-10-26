from agents import regime_agent


def test_whipsaw_guard_blocks_after_six_switches():
    """After 6 switches in a calendar year, further switches must be blocked."""
    cache_at_limit = {
        "regime": "BULL",
        "switch_count_ytd": 6,
        "switch_year": 2026,
    }
    result = regime_agent._apply_whipsaw_guard(
        candidate_regime="BEAR",
        prev_cache=cache_at_limit,
    )
    assert result == "BULL", (
        f"Whipsaw guard should block switch after 6 YTD, got: {result}"
    )


def test_whipsaw_guard_allows_switch_under_limit():
    cache_under_limit = {
        "regime": "BULL",
        "switch_count_ytd": 3,
        "switch_year": 2026,
    }
    result = regime_agent._apply_whipsaw_guard(
        candidate_regime="BEAR",
        prev_cache=cache_under_limit,
    )
    assert result == "BEAR", "Should allow switch when count < 6"


def test_whipsaw_guard_resets_for_new_year():
    """Counter from a previous year should be ignored (reset to 0)."""
    cache_old_year = {
        "regime": "BULL",
        "switch_count_ytd": 6,
        "switch_year": 2025,   # previous year
    }
    result = regime_agent._apply_whipsaw_guard(
        candidate_regime="BEAR",
        prev_cache=cache_old_year,
    )
    assert result == "BEAR", "Old-year switch_count_ytd should not block switches in new year"


def test_whipsaw_guard_no_prior_cache():
    """Empty cache (first run) should not block any switch."""
    result = regime_agent._apply_whipsaw_guard(
        candidate_regime="BEAR",
        prev_cache={},
    )
    assert result == "BEAR", "No prior cache should not block switch"


def test_whipsaw_guard_coerces_a_stale_regime_label_at_the_switch_cap():
    """A pre-Task-4 CHOPPY/SIDEWAYS label has nothing valid to preserve.

    At the switch cap the guard normally keeps the previous regime instead
    of the candidate — but a legacy 4-state label is not a direction, so
    "keeping" it would leak CHOPPY/SIDEWAYS into final_regime. There is
    nothing to guard in that case; the freshly computed candidate applies.
    """
    for stale_regime in ("CHOPPY", "SIDEWAYS"):
        cache_at_limit = {
            "regime": stale_regime,
            "switch_count_ytd": 6,
            "switch_year": 2026,
        }
        result = regime_agent._apply_whipsaw_guard(
            candidate_regime="BEAR",
            prev_cache=cache_at_limit,
        )
        assert result in ("BULL", "BEAR"), (
            f"stale prev regime {stale_regime!r} must not leak through, got: {result}"
        )
        assert result == "BEAR", "with nothing valid to preserve, the candidate applies"


def test_detect_regime_result_has_switch_count_ytd(synthetic_regime_market):
    result = regime_agent.detect_regime()
    assert "switch_count_ytd" in result, "detect_regime() must return switch_count_ytd"
    assert "switch_year" in result, "detect_regime() must return switch_year"
    assert isinstance(result["switch_count_ytd"], int)
