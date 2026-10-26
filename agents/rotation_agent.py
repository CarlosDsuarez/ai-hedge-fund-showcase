"""
Capital-Recycling Agent
=======================
Pure decision helpers for the pending-signal capital-recycling loop. No I/O
and no broker calls — main.py owns execution and persistence so these
functions stay deterministic and unit-testable.

  1. prioritize_deployments() — order freed cash: PENDING first, then new, by score.
  2. classify_buy_outcome()   — EXECUTE / QUEUE / REJECT for an approved BUY.

The weekly Monday rotation (is_rotation_day / select_rotation_exits) that used
to live here is retired: the exit engine's rank decay (design C) sells a
laggard after RANK_DECAY_DAYS consecutive weak runs instead of on a calendar.
"""

BUDGET_SKIP_REASON = "insufficient_budget"


def prioritize_deployments(pending_signals: list, new_candidates: list, score_map: dict) -> list:
    """Order freed cash: PENDING queue first (by score), then fresh BUYs (by score).

    Returns [{"ticker","source": "pending"|"new","score","conviction"}, ...].
    Pending always outranks new; ties broken by score desc. A name already in the
    pending set is not re-listed as a new candidate.
    """
    pend = sorted(
        ({"ticker": p["ticker"], "source": "pending",
          "score": p.get("composite_score") or 0.0,
          "conviction": p.get("conviction")} for p in pending_signals),
        key=lambda x: x["score"], reverse=True,
    )
    pend_tickers = {p["ticker"] for p in pend}
    new = sorted(
        ({"ticker": t, "source": "new",
          "score": score_map.get(t, 0.0), "conviction": None}
         for t in new_candidates if t not in pend_tickers),
        key=lambda x: x["score"], reverse=True,
    )
    return pend + new


def classify_buy_outcome(gates_passed: bool, fundable: bool) -> str:
    """Decide what to do with an approved BUY at the funding step.

    Only signals that cleared EVERY policy gate (PM=BUY, conviction, technical,
    sentiment, caps, not-held, not in cooldown) are eligible to queue. Anything blocked
    by a policy gate is REJECTed and must never be queued.

      gates_passed=True,  fundable=True  -> "EXECUTE"
      gates_passed=True,  fundable=False -> "QUEUE"
      gates_passed=False, *              -> "REJECT"
    """
    if not gates_passed:
        return "REJECT"
    return "EXECUTE" if fundable else "QUEUE"
