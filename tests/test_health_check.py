

def test_health_check_has_check_anthropic_key_function():
    import health_check
    assert hasattr(health_check, "_check_anthropic_key"), (
        "health_check must expose _check_anthropic_key(warnings_list)"
    )


def test_check_anthropic_key_warns_when_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import health_check
    warnings_list = []
    health_check._check_anthropic_key(warnings_list)
    assert any("ANTHROPIC_API_KEY" in w for w in warnings_list), (
        "Should warn when ANTHROPIC_API_KEY is missing"
    )


def test_check_anthropic_key_no_warning_when_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    import health_check
    warnings_list = []
    health_check._check_anthropic_key(warnings_list)
    assert warnings_list == [], "Should not warn when key is present"
