"""reports/conviction_backfill.py — fills in the forward returns
conviction_log has always logged but never scored.

Every research run appends a (date, ticker, conviction, entry_price) row via
db.append_conviction_log; return_30d/return_60d exist to answer "was
conviction 5 actually worth more than conviction 3", and have been NULL for
the project's whole life because nothing ever wrote them back in.

backfill_conviction_returns() walks the rows old enough to score (>=30
calendar days since the logged date for return_30d, >=60 for return_60d —
db.load_conviction_rows_needing_backfill owns that cutoff), fetches historical
closes for the tickers involved in one batched call, and derives each return
as (first close on/after date+Nd) / entry_price - 1. "First on/after" rather
than an exact-date lookup because date+30d routinely lands on a weekend or
holiday with no bar at all.
"""
from datetime import date, timedelta

import pandas as pd

from db import database as db
from data.fetcher import get_closes


def _first_close_on_or_after(closes: "pd.DataFrame", ticker: str, target: date) -> "float | None":
    if closes is None or closes.empty or ticker not in closes.columns:
        return None
    col = closes[ticker].dropna()
    after = col[col.index >= pd.Timestamp(target)]
    if after.empty:
        return None
    return float(after.iloc[0])


def backfill_conviction_returns(today: "date | None" = None) -> int:
    """Backfill every conviction_log row due for scoring. Returns the count
    of rows actually updated (at least one of return_30d/return_60d written).

    Called from main.run_reports() in a try/except, after signals are saved,
    and skipped entirely under DRY_RUN — this writes to conviction_log same
    as any other persisted state.
    """
    today = today or date.today()
    rows = db.load_conviction_rows_needing_backfill(min_age_days=30, today=today)
    if not rows:
        return 0

    tickers = sorted({r["ticker"] for r in rows if r.get("ticker")})
    closes = get_closes(tickers, period="6mo")

    updated = 0
    for row in rows:
        ticker = row.get("ticker")
        entry_price = row.get("entry_price")
        row_date = row.get("date")
        if not ticker or not entry_price or not row_date:
            continue
        try:
            entry_date = date.fromisoformat(row_date[:10])
        except ValueError:
            continue

        return_30d = return_60d = None
        if row.get("return_30d") is None and today >= entry_date + timedelta(days=30):
            close_30 = _first_close_on_or_after(closes, ticker, entry_date + timedelta(days=30))
            if close_30 is not None:
                return_30d = close_30 / entry_price - 1
        if row.get("return_60d") is None and today >= entry_date + timedelta(days=60):
            close_60 = _first_close_on_or_after(closes, ticker, entry_date + timedelta(days=60))
            if close_60 is not None:
                return_60d = close_60 / entry_price - 1

        if return_30d is not None or return_60d is not None:
            db.update_conviction_returns(ticker, row_date,
                                         return_30d=return_30d, return_60d=return_60d)
            updated += 1
    return updated
