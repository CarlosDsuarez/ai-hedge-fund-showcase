"""The entry predicate, shared by the automated pipeline and the terminal chat.

This used to be an if/elif chain inside main.py's run_entries. Copying it into
the chat would have guaranteed drift: two ladders of gates, both claiming to be
"the rules", diverging the first time one of them was tuned. So the chain moved
here and main.py calls it.

The reason codes are unchanged on purpose — utils.funnel.REASON_LABELS renders
them, db.funnel_log stores them, and tests/test_funnel.py pins them.

Severity is "soft" for every gate in this module. These are opinions about
whether a name deserves capital, and a human at the terminal is allowed to
disagree with them in writing. The gates that are never negotiable — no paper
account, unknown order state, no price — live in chat/pretrade.py, because the
pipeline reaches them through different code.
"""
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    severity: str = "soft"          # "hard" gates cannot be overridden
    reason_code: str | None = None
    detail: str = ""
    shortfall: float | None = None  # how far short, in the gate's own unit


@dataclass(frozen=True)
class EntryFacts:
    """Everything the entry gates need about ONE candidate.

    Deliberately plain data: the caller gathers it from wherever it has it (a
    live pipeline run, or a cached run blob in the chat), and the predicate
    stays identical for both.
    """
    ticker: str
    pm_action: str | None = None
    conviction: int = 0
    composite: float | None = None
    pm_degraded: bool = False
    min_conviction: int = config.MIN_PM_CONVICTION
    regime: str = "UNKNOWN"
    strategy_buy: bool = False
    regime_buy: bool = False
    sentiment: str = "NEUTRAL"
    held: bool = False
    in_cooldown: bool = False       # stopped out within REENTRY_COOLDOWN_DAYS trading days
    weight_capped: bool = False
    pm_sell: bool = False
    exited_today: bool = False
    macro_stressed: bool = False
    credit_z: float = 0.0
    signal_threshold: float = config.SIGNAL_THRESHOLD
    max_position_pct: float = config.MAX_POSITION_PCT


def _conviction_ok(f: EntryFacts) -> bool:
    """Conviction bar, skipped entirely on a degraded PM result.

    A degraded result carries *research* convictions. Scoring those against the
    PM bar would launder a research opinion into a PM approval, which is
    exactly the veto the fallback failed to apply.
    """
    if f.pm_degraded:
        return True
    return (f.conviction or 0) >= f.min_conviction


def _signal_ok(f: EntryFacts) -> bool:
    """Signal evidence required to enter.

    Healthy cycle: the PM vetted the name, so either the strategy composite or
    the regime-specific ranking suffices. Degraded cycle: nothing vetted it, so
    the quant composite must stand alone. strategy_agent emits BUY only when
    score >= SIGNAL_THRESHOLD with the credit-stress macro filter passing; the
    threshold is re-asserted here so this gate keeps holding even if that
    internal rule changes.
    """
    if f.pm_degraded:
        return f.strategy_buy and (f.composite or 0.0) >= f.signal_threshold
    return f.strategy_buy or f.regime_buy


def evaluate_entry(f: EntryFacts) -> list[GateResult]:
    """Every entry gate, in the pipeline's original order.

    Order is load-bearing: the PM action check comes first so it also absorbs
    every name flagged SELL — split the reason later and a SELL verdict reads
    as a shrug.
    """
    results: list[GateResult] = []

    def add(name, passed, reason_code=None, detail="", shortfall=None):
        results.append(GateResult(name=name, passed=passed, reason_code=reason_code,
                                  detail=detail, shortfall=shortfall))

    if f.pm_action != "BUY":
        add("pm_verdict", False,
            "PM_SELL" if f.pm_action == "SELL" else "PM_NOT_BUY",
            f"acción={f.pm_action}")
    else:
        add("pm_verdict", True)

    conv_ok = _conviction_ok(f)
    add("conviction", conv_ok,
        None if conv_ok else "CONVICTION_BELOW_GATE",
        "" if conv_ok else
        f"convicción {f.conviction} < {f.min_conviction} ({f.regime})",
        None if conv_ok else float(f.min_conviction - (f.conviction or 0)))

    sig_ok = _signal_ok(f)
    if sig_ok:
        add("signal", True)
    elif f.macro_stressed:
        # A suppressed BUY under credit stress is not the same finding as a
        # weak score, and reporting both as "no signal" hides a universe-wide
        # macro veto behind per-ticker noise.
        add("signal", False, "MACRO_CREDIT_STRESS",
            f"z-score HYG/LQD={f.credit_z:+.2f}")
    else:
        _score = f.composite if f.composite is not None else 0.0
        add("signal", False, "NO_SIGNAL",
            f"composite={_score:+.2f} < {f.signal_threshold}",
            float(f.signal_threshold - _score))

    negative = f.sentiment == "NEGATIVE"
    add("sentiment", not negative,
        "SENTIMENT_NEGATIVE" if negative else None,
        "titulares negativos" if negative else "")

    add("not_held", not f.held, "ALREADY_HELD" if f.held else None, "")

    # Design C: a name stopped out in the last REENTRY_COOLDOWN_DAYS trading
    # days is not a candidate. Buying HON back 5 times and JPM 6 times within
    # days of each stop was the churn the trailing stop is meant to end.
    add("cooldown", not f.in_cooldown,
        "COOLDOWN" if f.in_cooldown else None,
        f"stop en los últimos {config.REENTRY_COOLDOWN_DAYS} días de bolsa" if f.in_cooldown else "")

    add("weight_headroom", not f.weight_capped,
        "WEIGHT_CAPPED" if f.weight_capped else None,
        f"> {f.max_position_pct:.0%} del equity" if f.weight_capped else "")

    add("not_pm_sell", not f.pm_sell, "PM_SELL" if f.pm_sell else None, "")

    add("not_exited_today", not f.exited_today,
        "EXITED_TODAY" if f.exited_today else None, "")

    return results


def first_rejection(results: Iterable[GateResult]) -> GateResult | None:
    """The first failing gate, which is the one the funnel records."""
    return next((r for r in results if not r.passed), None)


def failures(results: Iterable[GateResult]) -> list[GateResult]:
    """Every failing gate — what the chat shows so the user sees the whole picture."""
    return [r for r in results if not r.passed]
