"""agents/entry_gates.py — the entry predicate main.py used to inline.

The equivalence test below re-implements the ORIGINAL if/elif chain from
main.py's run_entries and drives both versions over a matrix of fact sets. If
the extraction changed a verdict or a reason code anywhere, this fails: the
funnel's reason codes are stored in db.funnel_log and rendered by
utils.funnel.REASON_LABELS, so a silent rename would rewrite history's meaning.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.entry_gates import EntryFacts, evaluate_entry, failures, first_rejection


def _original_reject(f: EntryFacts):
    """main.py's chain as it stood before the extraction. Returns (code, detail) or None."""
    def _conviction_gate_ok(conviction):
        if f.pm_degraded:
            return True
        return (conviction or 0) >= f.min_conviction

    def _signal_gate_ok():
        if f.pm_degraded:
            return f.strategy_buy and (f.composite or 0.0) >= config.SIGNAL_THRESHOLD
        return f.strategy_buy or f.regime_buy

    if f.pm_action != "BUY":
        return (("PM_SELL" if f.pm_action == "SELL" else "PM_NOT_BUY"),
                f"acción={f.pm_action}")
    elif not _conviction_gate_ok(f.conviction):
        return ("CONVICTION_BELOW_GATE",
                f"convicción {f.conviction} < {f.min_conviction} ({f.regime})")
    elif not _signal_gate_ok():
        if f.macro_stressed:
            return ("MACRO_CREDIT_STRESS", f"z-score HYG/LQD={f.credit_z:+.2f}")
        return ("NO_SIGNAL",
                f"composite={f.composite if f.composite is not None else 0.0:+.2f} "
                f"< {config.SIGNAL_THRESHOLD}")
    elif f.sentiment == "NEGATIVE":
        return ("SENTIMENT_NEGATIVE", "titulares negativos")
    elif f.held:
        return ("ALREADY_HELD", "")
    elif f.weight_capped:
        return ("WEIGHT_CAPPED", f"> {config.MAX_POSITION_PCT:.0%} del equity")
    elif f.pm_sell:
        return ("PM_SELL", "")
    elif f.exited_today:
        return ("EXITED_TODAY", "")
    return None


def _matrix():
    """Every combination that changes a branch, not every combination that exists."""
    axes = {
        "pm_action": ["BUY", "SELL", "HOLD", None],
        "conviction": [0, 3, 5],
        "composite": [None, 0.1, 0.9],
        "pm_degraded": [False, True],
        "strategy_buy": [False, True],
        "regime_buy": [False, True],
        "sentiment": ["NEUTRAL", "NEGATIVE"],
        "macro_stressed": [False, True],
    }
    flags = {"held": [False, True], "weight_capped": [False, True],
             "pm_sell": [False, True], "exited_today": [False, True]}

    keys = list(axes)
    for combo in itertools.product(*axes.values()):
        base = dict(zip(keys, combo, strict=True))
        # Late-stage flags matter only for names that clear the early gates, so
        # vary them one at a time instead of multiplying the whole matrix.
        yield EntryFacts(ticker="AAPL", min_conviction=3, regime="BULL",
                         credit_z=-1.75, **base)
        for flag, values in flags.items():
            for v in values[1:]:
                yield EntryFacts(ticker="AAPL", min_conviction=3, regime="BULL",
                                 credit_z=-1.75, **base, **{flag: v})


def test_extraction_matches_the_original_chain():
    checked = 0
    for facts in _matrix():
        expected = _original_reject(facts)
        got = first_rejection(evaluate_entry(facts))
        if expected is None:
            assert got is None, f"{facts} should pass, got {got}"
        else:
            assert got is not None, f"{facts} should be rejected as {expected[0]}"
            assert (got.reason_code, got.detail) == expected, f"mismatch on {facts}"
        checked += 1
    assert checked > 500, "the matrix collapsed — it is no longer proving anything"


# ── individual gates ────────────────────────────────────────────────────────

def _passing(**over):
    base = dict(ticker="AAPL", pm_action="BUY", conviction=4, composite=0.9,
                strategy_buy=True, min_conviction=3, regime="BULL")
    base.update(over)
    return EntryFacts(**base)


def test_a_clean_candidate_passes_every_gate():
    assert failures(evaluate_entry(_passing())) == []


def test_conviction_shortfall_is_the_distance_to_the_bar():
    rej = first_rejection(evaluate_entry(_passing(conviction=1, min_conviction=4)))
    assert rej.reason_code == "CONVICTION_BELOW_GATE"
    assert rej.shortfall == 3.0


def test_signal_shortfall_is_the_distance_to_the_threshold():
    rej = first_rejection(evaluate_entry(
        _passing(pm_degraded=True, composite=0.1, strategy_buy=True)))
    assert rej.reason_code == "NO_SIGNAL"
    assert rej.shortfall == config.SIGNAL_THRESHOLD - 0.1


def test_degraded_pm_skips_the_conviction_bar_but_hardens_the_signal_bar():
    """A degraded result carries research convictions — scoring them as PM approval
    is exactly the veto the fallback failed to apply."""
    degraded = _passing(pm_degraded=True, conviction=1, composite=0.9, strategy_buy=True)
    assert failures(evaluate_entry(degraded)) == []

    # regime_buy alone no longer suffices once nothing vetted the name.
    weak = _passing(pm_degraded=True, conviction=5, strategy_buy=False, regime_buy=True)
    assert first_rejection(evaluate_entry(weak)).reason_code == "NO_SIGNAL"


def test_macro_stress_is_named_rather_than_hidden_behind_no_signal():
    rej = first_rejection(evaluate_entry(
        _passing(strategy_buy=False, regime_buy=False, macro_stressed=True, credit_z=-2.1)))
    assert rej.reason_code == "MACRO_CREDIT_STRESS"
    assert "-2.10" in rej.detail


def test_a_sell_verdict_is_not_reported_as_a_shrug():
    rej = first_rejection(evaluate_entry(_passing(pm_action="SELL")))
    assert rej.reason_code == "PM_SELL"


def test_a_recent_stop_out_is_in_cooldown():
    """Design C: 10 trading days after a stop before the name can be re-bought.
    HON was re-entered 5 times and JPM 6 times within days of a stop-out."""
    rej = first_rejection(evaluate_entry(_passing(in_cooldown=True)))
    assert rej.reason_code == "COOLDOWN"
    assert rej.severity == "soft"
    assert failures(evaluate_entry(_passing(in_cooldown=False))) == []


def test_cooldown_sits_right_after_already_held():
    """A held name reports ALREADY_HELD, not COOLDOWN; a flat one in cooldown
    reports COOLDOWN before the weight/PM-sell/exited flags."""
    both = _passing(held=True, in_cooldown=True)
    assert first_rejection(evaluate_entry(both)).reason_code == "ALREADY_HELD"
    later = _passing(in_cooldown=True, weight_capped=True, pm_sell=True, exited_today=True)
    assert first_rejection(evaluate_entry(later)).reason_code == "COOLDOWN"


def test_every_entry_gate_is_soft():
    """A human at the terminal may overrule an opinion about a name. The gates
    that are never negotiable live in chat/pretrade.py, not here."""
    for facts in (_passing(pm_action="HOLD"), _passing(conviction=0),
                  _passing(held=True), _passing(weight_capped=True)):
        for gate in evaluate_entry(facts):
            assert gate.severity == "soft"


def test_failures_reports_all_of_them_not_just_the_first():
    """The chat shows the whole picture; the funnel records only the first."""
    facts = _passing(pm_action="HOLD", conviction=0, strategy_buy=False,
                     sentiment="NEGATIVE", held=True)
    codes = {g.reason_code for g in failures(evaluate_entry(facts))}
    assert {"PM_NOT_BUY", "CONVICTION_BELOW_GATE", "NO_SIGNAL",
            "SENTIMENT_NEGATIVE", "ALREADY_HELD"} <= codes
    assert first_rejection(evaluate_entry(facts)).reason_code == "PM_NOT_BUY"
