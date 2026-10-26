"""Why each asset was discarded — the record the pipeline never kept.

A run used to decide the fate of every ticker in the universe and leave almost
no trace of it. The densest filter in the system, the `_buy_candidates`
comprehension in main.py, tested eight conditions at once, and a name failing
any of them simply vanished: no print, no log line, no row. The only after-the-
fact explanation was a best-effort guess in `_collect_health_issues`, which ran
only when a cycle produced zero orders and could name exactly one of the ~30
places a candidate can die.

This module keeps the record instead of reconstructing it. One row per ticker
per run: the stage it reached, whether it survived, and the reason code that
stopped it. Later stages overwrite earlier ones on purpose — the funnel is
sequential, so the last thing recorded for a ticker is what actually stopped it.

Nothing here talks to the network or the database; `db.database.save_funnel_log`
persists `rows()` and `alerts.telegram_alert.send_funnel_report` renders
`summary()`.
"""

from collections import Counter

# ── Outcomes ────────────────────────────────────────────────────────────
EXECUTED = "EXECUTED"
QUEUED   = "QUEUED"
REJECTED = "REJECTED"

# ── Stages, in funnel order ─────────────────────────────────────────────
STAGE_DATA     = "data"
STAGE_SIGNAL   = "signal"
STAGE_REGIME   = "regime"
STAGE_PM       = "pm"
STAGE_GATES    = "gates"
STAGE_BUDGET   = "budget"
STAGE_ORDER    = "order"

# ── Reason codes → what to show a human ─────────────────────────────────
# The label is what lands in Telegram, so it is written for the person reading
# it on a phone, not for the code path that produced it.
REASON_LABELS = {
    "NO_PRICE_DATA":         "Sin datos de precio",
    "MACRO_CREDIT_STRESS":   "Filtro macro: estrés de crédito",
    "REGIME_BLOCKED":        "Régimen bloquea entradas",
    "PM_NOT_BUY":            "El PM no recomendó comprar",
    "PM_SELL":               "El PM recomendó vender",
    "CONVICTION_BELOW_GATE": "Convicción bajo la compuerta",
    "NO_SIGNAL":             "Sin señal BUY de estrategia",
    "SENTIMENT_NEGATIVE":    "Sentimiento negativo",
    "ALREADY_HELD":          "Ya está en cartera",
    "COOLDOWN":              "En enfriamiento tras stop",
    "WEIGHT_CAPPED":         "Supera el tope por posición",
    "EXITED_TODAY":          "Vendido o rotado hoy",
    "NO_BUDGET":             "Sin presupuesto disponible",
    "GROSS_TARGET_REACHED":  "Exposición bruta objetivo alcanzada",
    "BELOW_MIN_TICKET":      "Importe bajo el mínimo",
    "QTY_LT_1":              "Menos de 1 acción al precio actual",
    "ORDER_REJECTED":        "El bróker rechazó la orden",
    "ORDER_ERROR":           "Error al enviar la orden",
    "EXECUTED":              "Comprado",
}


def label(reason_code: str) -> str:
    return REASON_LABELS.get(reason_code, reason_code)


class FunnelRecorder:
    """Per-run record of what happened to every ticker considered."""

    def __init__(self, run_timestamp: str, universe: "list[str] | None" = None):
        self.run_timestamp = run_timestamp
        self.universe = list(universe or [])
        self._records: dict[str, dict] = {}
        # Stage headcounts, set by main.py as the funnel narrows. Kept separate
        # from the per-ticker records because a stage can be skipped wholesale
        # (a regime block) without any single ticker being evaluated.
        self.stages: dict[str, int] = {}

    # ── writing ─────────────────────────────────────────────────────────
    def record(self, ticker: str, outcome: str, reason_code: str, *,
               stage: str = STAGE_GATES, detail: str = "",
               score: "float | None" = None,
               conviction: "int | None" = None) -> None:
        self._records[ticker] = {
            "ticker": ticker,
            "outcome": outcome,
            "reason_code": reason_code,
            "stage": stage,
            "detail": detail,
            "composite_score": score,
            "conviction": conviction,
        }

    def record_many(self, tickers, outcome: str, reason_code: str, **kwargs) -> None:
        for t in tickers:
            self.record(t, outcome, reason_code, **kwargs)

    def stage_count(self, stage: str, count: int) -> None:
        self.stages[stage] = count

    # ── reading ─────────────────────────────────────────────────────────
    def rows(self) -> list[dict]:
        return list(self._records.values())

    def by_outcome(self, outcome: str) -> list[dict]:
        return [r for r in self._records.values() if r["outcome"] == outcome]

    @staticmethod
    def _reason_breakdown(records: list) -> list:
        """Counts by reason_code, most-frequent first, for one outcome bucket."""
        counts = Counter(r["reason_code"] for r in records)
        return [
            {
                "code": code,
                "label": label(code),
                "count": n,
                "tickers": sorted(r["ticker"] for r in records if r["reason_code"] == code),
            }
            for code, n in counts.most_common()
        ]

    def summary(self) -> dict:
        """Counts by reason plus the single biggest blocker.

        `reasons` is ordered most-frequent first because that ordering IS the
        answer to "why did nothing trade today". `queued_reasons` is the same
        shape but built from QUEUED outcomes (e.g. NO_BUDGET) instead of
        REJECTED ones — kept separate because a queued ticker was deferred,
        not turned away, and folding it into `reasons`/`bottleneck` would
        make a name still eligible for tomorrow read as a rejection.
        """
        rejected = self.by_outcome(REJECTED)
        queued = self.by_outcome(QUEUED)
        reasons = self._reason_breakdown(rejected)
        queued_reasons = self._reason_breakdown(queued)
        return {
            "run_timestamp": self.run_timestamp,
            "universe": len(self.universe),
            "stages": dict(self.stages),
            "executed": sorted(r["ticker"] for r in self.by_outcome(EXECUTED)),
            "queued": sorted(r["ticker"] for r in queued),
            "rejected_total": len(rejected),
            "reasons": reasons,
            "queued_reasons": queued_reasons,
            "bottleneck": reasons[0] if reasons else None,
        }
