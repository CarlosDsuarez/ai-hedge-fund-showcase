"""
db/database.py — SQLite persistence layer for the AI Hedge Fund.

Single source of truth for all mutable state. Replaces fragile JSON files.
Uses sqlite3 (Python stdlib — no new dependencies).

Tables:
  positions       — open position metadata (entry price, conviction, holding_days, etc.)
  position_state  — exit-engine record per position (stops, chandelier, decay counter, broker stop)
  regime_state    — latest market regime snapshot (single-row cache)
  runs            — full run history (one row per daily execution)
  signals         — per-ticker signals per run (for analysis)
  conviction_log  — ticker + conviction + entry_price + forward returns
  chat_decisions  — trade decisions taken from the terminal chat (audit journal)
  fills           — Alpaca's own fill history (execution/ledger.py, keyed on Alpaca's activity id)
  trades          — FIFO-rebuilt round trips derived from fills (realized P&L)
  exits           — append-only log of why a position disappeared (broker or software)

All writes are wrapped in transactions. Readers return dicts matching the
exact format of the old JSON files so callers need no changes.

Resilience: every public read returns an empty default on failure (callers fall
back to JSON), and every public write logs a warning and returns instead of
raising. The system must never crash because of a DB problem.
"""

import json
import os
import sqlite3
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "hedge_fund.db")


@contextmanager
def _conn():
    """Context manager: yields a connection with WAL mode and foreign keys."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")   # allows concurrent reads during writes
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Columns added to position_state after its original CREATE TABLE. The CREATE
# above is left as it was on purpose: a fresh DB and a migrated one must end up
# with the identical column set, and one list feeding both paths is how.
_POSITION_STATE_MIGRATION_COLUMNS = (
    ("initial_stop",         "REAL"),
    ("trail_stop",           "REAL"),
    ("highest_close",        "REAL"),
    ("decay_count",          "INTEGER DEFAULT 0"),
    ("broker_stop_order_id", "TEXT"),
    ("broker_stop_price",    "REAL"),
    ("broker_stop_qty",      "INTEGER"),
)

# What save/upsert write. Legacy ladder columns are deliberately absent: a
# record migrated from logs/position_state.json carries its old sl_price as
# initial_stop/trail_stop (execution/order_manager), and nothing downstream
# reads tp*_price from the DB.
_POSITION_STATE_COLUMNS = (
    "entry_price", "entry_time", "atr_14", "initial_stop", "trail_stop",
    "highest_close", "qty_total", "qty_remaining", "decay_count",
    "broker_stop_order_id", "broker_stop_price", "broker_stop_qty",
)

# The TP-ladder columns, still physically in the table (SQLite cannot drop a
# column cheaply) but never read again. Hidden from load_position_state so a
# record round-trips as exactly what was saved — their DEFAULT 0 would
# otherwise resurface as tp1_filled=0 on every record forever.
_POSITION_STATE_RETIRED_COLUMNS = frozenset((
    "sl_price", "sl_is_breakeven", "tp1_price", "tp1_filled",
    "tp2_price", "tp2_filled", "tp3_price", "tp3_filled",
))


def init_db():
    """
    Create all tables if they don't exist.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS.
    Never raises: a DB failure prints a warning and falls back to JSON-only mode.
    """
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with _conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                ticker          TEXT PRIMARY KEY,
                entry_price     REAL NOT NULL,
                entry_date      TEXT NOT NULL,
                holding_days    INTEGER DEFAULT 0,
                conviction      INTEGER DEFAULT 3,
                size_pct        REAL DEFAULT 5.0,
                updated_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS position_state (
                ticker          TEXT PRIMARY KEY,
                entry_price     REAL,
                atr_14          REAL,
                sl_price        REAL,
                sl_is_breakeven INTEGER DEFAULT 0,
                tp1_price       REAL,
                tp1_filled      INTEGER DEFAULT 0,
                tp2_price       REAL,
                tp2_filled      INTEGER DEFAULT 0,
                tp3_price       REAL,
                tp3_filled      INTEGER DEFAULT 0,
                qty_total       INTEGER DEFAULT 0,
                qty_remaining   INTEGER DEFAULT 0,
                entry_time      TEXT,
                updated_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS regime_state (
                id              INTEGER PRIMARY KEY DEFAULT 1,
                regime          TEXT DEFAULT 'BULL',
                spy_price       REAL,
                spy_ema200      REAL,
                spy_sma20       REAL,
                buffer_pct      REAL,
                confirmed       INTEGER DEFAULT 1,
                days_in_regime  INTEGER DEFAULT 0,
                switched_today  INTEGER DEFAULT 0,
                vix             REAL,
                vol_regime      TEXT DEFAULT 'MED',
                size_multiplier REAL DEFAULT 1.0,
                credit_signal   TEXT DEFAULT 'NORMAL',
                switch_count_ytd INTEGER DEFAULT 0,
                switch_year     INTEGER,
                updated_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp   TEXT NOT NULL,
                paper_trading   INTEGER DEFAULT 1,
                yc_allocation   REAL,
                yc_regime       TEXT,
                vol_regime      TEXT,
                vix             REAL,
                macro_stressed  INTEGER DEFAULT 0,
                orders_count    INTEGER DEFAULT 0,
                exits_count     INTEGER DEFAULT 0,
                run_json        TEXT,       -- full run_data as JSON blob (for backward compat)
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(run_timestamp);

            CREATE TABLE IF NOT EXISTS signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp   TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                signal          TEXT,
                composite_score REAL,
                mom_z           REAL,
                val_z           REAL,
                sent_z          REAL,
                pm_action       TEXT,
                conviction      INTEGER,
                regime_signal   TEXT,
                alpha_filtered  INTEGER DEFAULT 0,
                filter_reason   TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker, run_timestamp);

            CREATE TABLE IF NOT EXISTS conviction_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                conviction      INTEGER NOT NULL,
                entry_price     REAL NOT NULL,
                return_30d      REAL,
                return_60d      REAL,
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_conviction_ticker ON conviction_log(ticker, date);

            CREATE TABLE IF NOT EXISTS pending_signals (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp    TEXT NOT NULL,
                ticker           TEXT NOT NULL,
                composite_score  REAL,
                conviction       INTEGER,
                pm_action        TEXT,
                size_pct         REAL,
                reason_deferred  TEXT,
                status           TEXT DEFAULT 'PENDING',
                created_at       TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_signals(status, composite_score);

            -- Why each ticker was discarded, one row per ticker per run. The
            -- run used to decide this for the whole universe and keep no
            -- record of it, so "nothing traded today" had no auditable answer.
            CREATE TABLE IF NOT EXISTS funnel_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp    TEXT NOT NULL,
                ticker           TEXT NOT NULL,
                stage            TEXT,
                outcome          TEXT,
                reason_code      TEXT,
                detail           TEXT,
                composite_score  REAL,
                conviction       INTEGER,
                created_at       TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_funnel_run ON funnel_log(run_timestamp, reason_code);
            CREATE INDEX IF NOT EXISTS idx_funnel_ticker ON funnel_log(ticker, run_timestamp);

            -- Every trade decision taken from the terminal chat, accepted AND
            -- rejected, with the gate verdicts that were on screen when the
            -- user answered. funnel_log cannot serve this: save_funnel_log
            -- deletes the run's rows before inserting, so proposing the same
            -- ticker twice would erase the first decision.
            CREATE TABLE IF NOT EXISTS chat_decisions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id      TEXT NOT NULL UNIQUE,
                created_at       TEXT DEFAULT (datetime('now')),
                session_id       TEXT,
                question         TEXT,
                kind             TEXT NOT NULL,
                ticker           TEXT NOT NULL,
                paper_trading    INTEGER NOT NULL,
                source           TEXT,
                proposed_qty     INTEGER,
                ref_price        REAL,
                size_pct         REAL,
                sl_price         REAL,
                tp_price         REAL,
                atr_14           REAL,
                conviction       INTEGER,
                composite_score  REAL,
                regime           TEXT,
                sizing_json      TEXT,
                gates_json       TEXT,
                gates_failed     TEXT,
                user_choice      TEXT NOT NULL,
                overridden       INTEGER DEFAULT 0,
                outcome          TEXT,
                filled_qty       INTEGER,
                fill_price       REAL,
                order_id         TEXT,
                oco_order_id     TEXT,
                protected        INTEGER,
                error            TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_chat_dec_ticker ON chat_decisions(ticker, created_at);
            CREATE INDEX IF NOT EXISTS idx_chat_dec_outcome ON chat_decisions(outcome, created_at);

            -- D. Ledger: the source of truth for realized P&L. Without it,
            -- broker-side exits were never reconciled — the only local record of a position was
            -- logs/position_state.json, which a broker-driven close never
            -- touched. fills is Alpaca's own activity feed (immutable, keyed
            -- on Alpaca's own id so re-fetching an overlapping window is
            -- idempotent); trades is FIFO-rebuilt from fills, never edited by
            -- hand; exits is the append-only log of "why did this position
            -- disappear" (execution/ledger.py).
            CREATE TABLE IF NOT EXISTS fills (
                id                TEXT PRIMARY KEY,
                symbol            TEXT NOT NULL,
                side              TEXT NOT NULL,
                qty               REAL NOT NULL,
                price             REAL NOT NULL,
                transaction_time  TEXT NOT NULL,
                order_id          TEXT,
                client_order_id   TEXT,
                tag               TEXT,
                created_at        TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol, transaction_time);

            CREATE TABLE IF NOT EXISTS trades (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol         TEXT NOT NULL,
                entry_time     TEXT,
                exit_time      TEXT NOT NULL,
                qty            REAL NOT NULL,
                entry_price    REAL,
                exit_price     REAL,
                pnl_usd        REAL,
                pnl_pct        REAL,
                holding_days   INTEGER,
                exit_reason    TEXT,
                entry_fill_id  TEXT,
                exit_fill_id   TEXT,
                created_at     TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol, exit_time);
            CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);

            CREATE TABLE IF NOT EXISTS exits (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker         TEXT NOT NULL,
                exit_time      TEXT NOT NULL,
                reason         TEXT NOT NULL,
                qty            REAL,
                price          REAL,
                run_timestamp  TEXT,
                created_at     TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_exits_ticker ON exits(ticker, exit_time);
            CREATE INDEX IF NOT EXISTS idx_exits_reason ON exits(reason, exit_time);
            """)
            # C. Exit engine (Sprint 1): position_state grows the trailing-stop
            # columns. The TP-ladder columns (sl_price, tp*_price, *_filled)
            # stay — SQLite cannot drop them cheaply and nothing writes them any
            # more. Same try/except ALTER pattern as save_signals: on an existing
            # DB the first call adds each column, every later call hits
            # "duplicate column name" and moves on.
            for col, dtype in _POSITION_STATE_MIGRATION_COLUMNS:
                try:
                    conn.execute(f"ALTER TABLE position_state ADD COLUMN {col} {dtype}")
                except Exception:
                    pass
    except Exception as e:
        print(f"  [DB] init failed, falling back to JSON-only mode: {e}")




# ── POSITIONS ────────────────────────────────────────────────────────────────

def load_position_metadata() -> dict:
    """Returns {ticker: {entry_price, entry_date, holding_days, conviction, size_pct}}"""
    try:
        with _conn() as conn:
            rows = conn.execute("SELECT * FROM positions").fetchall()
            return {
                row["ticker"]: {
                    "entry_price":  row["entry_price"],
                    "entry_date":   row["entry_date"],
                    "holding_days": row["holding_days"],
                    "conviction":   row["conviction"],
                    "size_pct":     row["size_pct"],
                }
                for row in rows
            }
    except Exception as e:
        print(f"  [DB] load_position_metadata failed, returning empty: {e}")
        return {}


def save_position_metadata(metadata: dict):
    """
    Full replace: deletes all positions then re-inserts from metadata dict.
    Called by main.py after each run cycle.
    """
    try:
        with _conn() as conn:
            conn.execute("DELETE FROM positions")
            for ticker, data in metadata.items():
                conn.execute("""
                    INSERT INTO positions
                        (ticker, entry_price, entry_date, holding_days, conviction, size_pct, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """, (
                    ticker,
                    data.get("entry_price"),
                    data.get("entry_date"),
                    data.get("holding_days", 0),
                    data.get("conviction", 3),
                    data.get("size_pct", 5.0),
                ))
    except Exception as e:
        print(f"  [DB] save_position_metadata failed (JSON backup still written): {e}")


def upsert_position(ticker: str, data: dict):
    """Insert or update a single position (used during live execution)."""
    try:
        with _conn() as conn:
            conn.execute("""
                INSERT INTO positions
                    (ticker, entry_price, entry_date, holding_days, conviction, size_pct, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(ticker) DO UPDATE SET
                    entry_price  = excluded.entry_price,
                    entry_date   = excluded.entry_date,
                    holding_days = excluded.holding_days,
                    conviction   = excluded.conviction,
                    size_pct     = excluded.size_pct,
                    updated_at   = datetime('now')
            """, (
                ticker,
                data.get("entry_price"),
                data.get("entry_date"),
                data.get("holding_days", 0),
                data.get("conviction", 3),
                data.get("size_pct", 5.0),
            ))
    except Exception as e:
        print(f"  [DB] upsert_position failed: {e}")


def delete_position(ticker: str):
    try:
        with _conn() as conn:
            conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
    except Exception as e:
        print(f"  [DB] delete_position failed: {e}")


# ── POSITION STATE (exit engine) ─────────────────────────────────────────────
# Source of truth for execution/order_manager since Sprint 1 (design C). Until
# then these functions were dead code — order_manager read and wrote
# logs/position_state.json only — and the decay counter never persisted.

def _row_to_state_record(row) -> dict:
    """One position_state row as the record order_manager works with.

    Every live column flows through so a column added by migration reaches
    the engine without a matching edit here. NULLs are dropped, not surfaced
    as None, so a record round-trips as exactly the keys it was saved with.
    """
    return {
        k: v for k, v in dict(row).items()
        if k not in ("ticker", "updated_at")
        and k not in _POSITION_STATE_RETIRED_COLUMNS
        and v is not None
    }


def load_position_state() -> dict:
    """Returns {ticker: record}. Empty dict on any failure (never raises)."""
    try:
        with _conn() as conn:
            rows = conn.execute("SELECT * FROM position_state").fetchall()
            return {row["ticker"]: _row_to_state_record(row) for row in rows}
    except Exception as e:
        print(f"  [DB] load_position_state failed, returning empty: {e}")
        return {}


def _state_values(data: dict) -> tuple:
    return tuple(data.get(col) for col in _POSITION_STATE_COLUMNS)


_STATE_INSERT_SQL = (
    f"INSERT INTO position_state (ticker, {', '.join(_POSITION_STATE_COLUMNS)}, updated_at) "
    f"VALUES (?, {', '.join('?' * len(_POSITION_STATE_COLUMNS))}, datetime('now'))"
)


def save_position_state(state: dict):
    """Full replace of position_state: {ticker: record} in, table out."""
    try:
        with _conn() as conn:
            conn.execute("DELETE FROM position_state")
            for ticker, data in state.items():
                conn.execute(_STATE_INSERT_SQL, (ticker, *_state_values(data)))
    except Exception as e:
        print(f"  [DB] save_position_state failed (JSON mirror still written): {e}")


# ── REGIME STATE ─────────────────────────────────────────────────────────────

def load_regime_state() -> dict:
    """Returns the single-row regime state as a dict (matches regime_state.json format)."""
    try:
        with _conn() as conn:
            row = conn.execute("SELECT * FROM regime_state WHERE id = 1").fetchone()
            if row is None:
                return {}
            result = dict(row)
            result.pop("id", None)
            result.pop("updated_at", None)
            return result
    except Exception as e:
        print(f"  [DB] load_regime_state failed, returning empty: {e}")
        return {}


def save_regime_state(state: dict):
    """Upsert the single regime state row."""
    try:
        with _conn() as conn:
            conn.execute("""
                INSERT INTO regime_state
                    (id, regime, spy_price, spy_ema200, spy_sma20, buffer_pct,
                     confirmed, days_in_regime, switched_today, vix, vol_regime,
                     size_multiplier, credit_signal, switch_count_ytd, switch_year,
                     updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                    regime           = excluded.regime,
                    spy_price        = excluded.spy_price,
                    spy_ema200       = excluded.spy_ema200,
                    spy_sma20        = excluded.spy_sma20,
                    buffer_pct       = excluded.buffer_pct,
                    confirmed        = excluded.confirmed,
                    days_in_regime   = excluded.days_in_regime,
                    switched_today   = excluded.switched_today,
                    vix              = excluded.vix,
                    vol_regime       = excluded.vol_regime,
                    size_multiplier  = excluded.size_multiplier,
                    credit_signal    = excluded.credit_signal,
                    switch_count_ytd = excluded.switch_count_ytd,
                    switch_year      = excluded.switch_year,
                    updated_at       = datetime('now')
            """, (
                state.get("regime", "BULL"),
                state.get("spy_price"),
                state.get("spy_ema200"),
                state.get("spy_sma20"),
                state.get("buffer_pct"),
                int(state.get("confirmed", True)),
                state.get("days_in_regime", 0),
                int(state.get("switched_today", False)),
                state.get("vix"),
                state.get("vol_regime", "MED"),
                state.get("size_multiplier", 1.0),
                state.get("credit_signal", "NORMAL"),
                state.get("switch_count_ytd", 0),
                state.get("switch_year", datetime.now().year),
            ))
    except Exception as e:
        print(f"  [DB] save_regime_state failed (JSON backup still written): {e}")


# ── RUNS ─────────────────────────────────────────────────────────────────────

def save_run(run_data: dict):
    """Persist a full daily run. Stores key metrics as columns + full JSON blob."""
    try:
        regime = run_data.get("regime") if isinstance(run_data.get("regime"), dict) else {}
        with _conn() as conn:
            conn.execute("""
                INSERT INTO runs
                    (run_timestamp, paper_trading, yc_allocation, yc_regime,
                     vol_regime, vix, macro_stressed, orders_count,
                     exits_count, run_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                run_data.get("timestamp"),
                int(run_data.get("paper_trading", True)),
                run_data.get("yc_allocation"),
                regime.get("regime"),
                regime.get("vol_regime"),
                regime.get("vix"),
                int(bool(run_data.get("macro_stressed", False))),
                len(run_data.get("orders_submitted", [])),
                len(run_data.get("closed_positions", [])),
                json.dumps(run_data, default=str),
            ))
    except Exception as e:
        print(f"  [DB] save_run failed (JSON backup still written): {e}")


def load_recent_runs(n: int = 30) -> list[dict]:
    """Returns the last N runs as dicts (from JSON blob)."""
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT run_json FROM runs ORDER BY created_at DESC LIMIT ?", (n,)
            ).fetchall()
            results = []
            for row in rows:
                try:
                    results.append(json.loads(row["run_json"]))
                except Exception:
                    pass
            return results
    except Exception as e:
        print(f"  [DB] load_recent_runs failed, returning empty: {e}")
        return []


# ── CONVICTION LOG ───────────────────────────────────────────────────────────

def append_conviction_log(date: str, ticker: str, conviction: int, entry_price: float):
    """Append one entry. return_30d / return_60d are filled later by reports/conviction_backfill.py"""
    try:
        with _conn() as conn:
            conn.execute("""
                INSERT INTO conviction_log (date, ticker, conviction, entry_price)
                VALUES (?, ?, ?, ?)
            """, (date, ticker, conviction, entry_price))
    except Exception as e:
        print(f"  [DB] append_conviction_log failed (CSV backup still written): {e}")


def update_conviction_returns(ticker: str, date: str, return_30d: float = None, return_60d: float = None):
    """Backfill forward returns for a conviction log entry."""
    try:
        with _conn() as conn:
            if return_30d is not None:
                conn.execute(
                    "UPDATE conviction_log SET return_30d = ? WHERE ticker = ? AND date = ?",
                    (return_30d, ticker, date)
                )
            if return_60d is not None:
                conn.execute(
                    "UPDATE conviction_log SET return_60d = ? WHERE ticker = ? AND date = ?",
                    (return_60d, ticker, date)
                )
    except Exception as e:
        print(f"  [DB] update_conviction_returns failed: {e}")


def load_conviction_log() -> list[dict]:
    try:
        with _conn() as conn:
            rows = conn.execute("SELECT * FROM conviction_log ORDER BY date").fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"  [DB] load_conviction_log failed, returning empty: {e}")
        return []


# ── SIGNALS ──────────────────────────────────────────────────────────────────

def save_signals(run_date: str, research_results: list):
    """Upsert per-ticker research signals for a given run date into the signals table.
    Adds ATR and thesis columns if they don't exist yet (non-destructive migration)."""
    try:
        with _conn() as conn:
            for col, dtype in [
                ("sl_price",  "REAL"),
                ("tp1_price", "REAL"),
                ("tp2_price", "REAL"),
                ("tp3_price", "REAL"),
                ("thesis",    "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {dtype}")
                except Exception:
                    pass  # column already exists — sqlite3 raises "duplicate column name"

            for r in research_results:
                ticker = r.get("ticker")
                if not ticker:
                    continue
                conn.execute("""
                    INSERT INTO signals
                        (run_timestamp, ticker, signal, conviction, thesis,
                         sl_price, tp1_price, tp2_price, tp3_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_date,
                    ticker,
                    r.get("recommendation"),
                    r.get("conviction"),
                    r.get("thesis"),
                    r.get("sl_price"),
                    r.get("tp1_price"),
                    r.get("tp2_price"),
                    r.get("tp3_price"),
                ))
    except Exception as e:
        print(f"  [DB] save_signals failed: {e}")


# ── PENDING SIGNALS (capital-recycling queue) ────────────────────────────────

def save_pending_signals(run_timestamp: str, signals: list):
    """Queue approved BUYs deferred for lack of free budget.

    Each item: {ticker, composite_score, conviction, pm_action, size_pct, reason_deferred}.
    De-dupes on ticker: an existing PENDING row for the same name is left in place."""
    try:
        with _conn() as conn:
            existing = {
                row["ticker"] for row in conn.execute(
                    "SELECT ticker FROM pending_signals WHERE status = 'PENDING'"
                ).fetchall()
            }
            for s in signals:
                ticker = s.get("ticker")
                if not ticker or ticker in existing:
                    continue
                conn.execute("""
                    INSERT INTO pending_signals
                        (run_timestamp, ticker, composite_score, conviction,
                         pm_action, size_pct, reason_deferred, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', datetime('now'))
                """, (
                    run_timestamp,
                    ticker,
                    s.get("composite_score"),
                    s.get("conviction"),
                    s.get("pm_action", "BUY"),
                    s.get("size_pct"),
                    s.get("reason_deferred", "insufficient_budget"),
                ))
                existing.add(ticker)
    except Exception as e:
        print(f"  [DB] save_pending_signals failed (JSON backup still written): {e}")


def save_funnel_log(run_timestamp: str, rows: list):
    """Persist one decision row per ticker for this run.

    Each row: {ticker, stage, outcome, reason_code, detail, composite_score,
    conviction} — the shape utils.funnel.FunnelRecorder.rows() emits. Replaces
    any rows already stored for the same run_timestamp so a re-run of the same
    cycle does not stack duplicate explanations."""
    try:
        with _conn() as conn:
            conn.execute("DELETE FROM funnel_log WHERE run_timestamp = ?", (run_timestamp,))
            for r in rows:
                ticker = r.get("ticker")
                if not ticker:
                    continue
                conn.execute("""
                    INSERT INTO funnel_log
                        (run_timestamp, ticker, stage, outcome, reason_code,
                         detail, composite_score, conviction, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (
                    run_timestamp, ticker, r.get("stage"), r.get("outcome"),
                    r.get("reason_code"), r.get("detail"),
                    r.get("composite_score"), r.get("conviction"),
                ))
    except Exception as e:
        print(f"  [DB] save_funnel_log failed: {e}")


def load_funnel_log(run_timestamp: str = None, limit: int = 500) -> list[dict]:
    """Decision rows, newest run first. Pass run_timestamp for a single run."""
    try:
        with _conn() as conn:
            if run_timestamp:
                rows = conn.execute(
                    "SELECT * FROM funnel_log WHERE run_timestamp = ? ORDER BY ticker",
                    (run_timestamp,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM funnel_log ORDER BY run_timestamp DESC, ticker LIMIT ?",
                    (limit,)).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"  [DB] load_funnel_log failed: {e}")
        return []


def funnel_reason_history(days: int = 30) -> list[dict]:
    """How often each reason stopped a candidate over the recent past.

    The point of storing reasons is trend, not a single snapshot: "the
    conviction gate killed 40 signals this month" is the kind of statement that
    should come from data instead of an argument."""
    try:
        with _conn() as conn:
            rows = conn.execute("""
                SELECT reason_code, COUNT(*) AS n, COUNT(DISTINCT run_timestamp) AS runs
                FROM funnel_log
                WHERE outcome = 'REJECTED'
                  AND created_at >= datetime('now', ?)
                GROUP BY reason_code
                ORDER BY n DESC
            """, (f"-{int(days)} days",)).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"  [DB] funnel_reason_history failed: {e}")
        return []


def load_pending_signals() -> list[dict]:
    """Return all PENDING queued signals, highest composite score first."""
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT * FROM pending_signals WHERE status = 'PENDING' "
                "ORDER BY composite_score DESC"
            ).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"  [DB] load_pending_signals failed, returning empty: {e}")
        return []


def mark_pending_executed(ticker: str):
    """Flip the PENDING row for `ticker` to EXECUTED once it has been deployed."""
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE pending_signals SET status = 'EXECUTED' "
                "WHERE ticker = ? AND status = 'PENDING'",
                (ticker,),
            )
    except Exception as e:
        print(f"  [DB] mark_pending_executed failed: {e}")


def expire_pending_signals(max_age_days: int) -> int:
    """Mark PENDING signals older than max_age_days (calendar) as EXPIRED.
    Returns the count expired. Stale theses must never execute."""
    try:
        with _conn() as conn:
            cur = conn.execute(
                "UPDATE pending_signals SET status = 'EXPIRED' "
                "WHERE status = 'PENDING' "
                "AND julianday('now') - julianday(created_at) > ?",
                (max_age_days,),
            )
            return cur.rowcount
    except Exception as e:
        print(f"  [DB] expire_pending_signals failed: {e}")
        return 0


def save_llm_usage(agent: str, model: str, tokens_in: int, tokens_out: int, cache_read: int = 0):
    """Persist one Anthropic API call's token usage. Never raises (DB failure = warning)."""
    try:
        with _conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tokens_in INTEGER NOT NULL,
                    tokens_out INTEGER NOT NULL,
                    cache_read INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                INSERT INTO llm_usage (ts, agent, model, tokens_in, tokens_out, cache_read)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(timespec="seconds"),
                agent, model, int(tokens_in), int(tokens_out), int(cache_read),
            ))
    except Exception as e:
        print(f"  [DB] save_llm_usage failed: {e}")


# ── CHAT DECISIONS ───────────────────────────────────────────────────────────

_CHAT_DECISION_FIELDS = (
    "session_id", "question", "kind", "ticker", "paper_trading", "source",
    "proposed_qty", "ref_price", "size_pct", "sl_price", "tp_price", "atr_14",
    "conviction", "composite_score", "regime", "sizing_json", "gates_json",
    "gates_failed", "user_choice", "overridden",
)

_CHAT_OUTCOME_FIELDS = (
    "outcome", "filled_qty", "fill_price", "order_id", "oco_order_id",
    "protected", "error",
)


def record_chat_decision(decision_id: str, data: dict) -> bool:
    """Persist a chat trade decision the moment the user answers.

    Written for confirmed, rejected AND blocked decisions alike: a crash
    between the answer and the broker call must still leave the intent and the
    gate verdicts on record. The outcome arrives later via
    update_chat_decision_outcome().

    `data` carries the _CHAT_DECISION_FIELDS keys; `gates_json` and
    `sizing_json` are expected pre-serialised. Returns False on failure —
    never raises, like every other writer here.
    """
    try:
        cols = ", ".join(("decision_id",) + _CHAT_DECISION_FIELDS)
        placeholders = ", ".join("?" * (1 + len(_CHAT_DECISION_FIELDS)))
        values = [decision_id] + [data.get(f) for f in _CHAT_DECISION_FIELDS]
        with _conn() as conn:
            conn.execute(
                f"INSERT INTO chat_decisions ({cols}) VALUES ({placeholders})",
                values,
            )
        return True
    except Exception as e:
        print(f"  [DB] record_chat_decision failed: {e}")
        return False


def update_chat_decision_outcome(decision_id: str, **fields) -> bool:
    """Fill in what the broker actually did for an already-recorded decision.

    Accepts any of _CHAT_OUTCOME_FIELDS; unknown keys are ignored rather than
    raising, so a caller adding context cannot break the journal.
    """
    updates = {k: v for k, v in fields.items() if k in _CHAT_OUTCOME_FIELDS}
    if not updates:
        return False
    try:
        assignments = ", ".join(f"{k} = ?" for k in updates)
        with _conn() as conn:
            cur = conn.execute(
                f"UPDATE chat_decisions SET {assignments} WHERE decision_id = ?",
                list(updates.values()) + [decision_id],
            )
            return cur.rowcount > 0
    except Exception as e:
        print(f"  [DB] update_chat_decision_outcome failed: {e}")
        return False


def load_chat_decisions(limit: int = 50, ticker: str | None = None) -> list[dict]:
    """Most recent chat decisions first. Returns [] on failure."""
    try:
        with _conn() as conn:
            if ticker:
                rows = conn.execute(
                    "SELECT * FROM chat_decisions WHERE ticker = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (ticker, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM chat_decisions ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"  [DB] load_chat_decisions failed: {e}")
        return []


# ── LEDGER: fills, trades, exits (audit D, 2026-09-15) ──────────────────────

def insert_fills(fills: list[dict]) -> list[dict]:
    """INSERT OR IGNORE keyed on Alpaca's own activity id.

    execution.ledger.reconcile_fills fetches with `after=last_fill_time()`,
    and Alpaca's `after` is inclusive on that boundary row, so re-running the
    same window must not double-insert it. Returns the fills actually
    inserted (the NEW subset, in input order), not len(fills): the ledger's
    broker-stop signal must only ever look at fills it has not seen before,
    or the boundary row re-detects the same exit on every run.
    """
    if not fills:
        return []
    try:
        with _conn() as conn:
            inserted = []
            for f in fills:
                cur = conn.execute("""
                    INSERT OR IGNORE INTO fills
                        (id, symbol, side, qty, price, transaction_time,
                         order_id, client_order_id, tag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f.get("id"), f.get("symbol"), f.get("side"),
                    f.get("qty"), f.get("price"), f.get("transaction_time"),
                    f.get("order_id"), f.get("client_order_id"), f.get("tag"),
                ))
                if cur.rowcount:
                    inserted.append(f)
            return inserted
    except Exception as e:
        print(f"  [DB] insert_fills failed: {e}")
        return []


def last_fill_time() -> "str | None":
    """The transaction_time of the most recent fill on record, or None.

    execution.ledger.reconcile_fills uses this as the `after` cursor, falling
    back to config.LEDGER_BACKFILL_FROM on an empty table (first run).
    """
    try:
        with _conn() as conn:
            row = conn.execute("SELECT MAX(transaction_time) AS t FROM fills").fetchone()
            return row["t"] if row and row["t"] else None
    except Exception as e:
        print(f"  [DB] last_fill_time failed: {e}")
        return None


def load_fills(symbol: "str | None" = None) -> list[dict]:
    try:
        with _conn() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM fills WHERE symbol = ? ORDER BY transaction_time",
                    (symbol,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM fills ORDER BY transaction_time"
                ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"  [DB] load_fills failed: {e}")
        return []


def _parse_fill_dt(ts: str) -> datetime:
    """Alpaca timestamps end in 'Z'; fromisoformat wants '+00:00' on < 3.11."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def rebuild_trades() -> int:
    """Rebuild the trades table FIFO from fills. Idempotent — always starts
    from a clean slate, so repeating a fills backfill never double-counts.

    Per symbol, ordered by transaction_time: a buy opens a lot; a sell
    consumes the oldest lots first, emitting one trades row per (sell, lot)
    pair it draws from (design decision: a sell spanning two buy lots is two
    trades rows, never a single blended-average row). A sell that runs out of
    lots — a position opened before LEDGER_BACKFILL_FROM, whose buy fill was
    never recorded — still gets a row with entry_price/pnl_usd NULL, so it is
    visible in load_trades() without corrupting P&L stats.

    exit_reason resolves the sell fill's tag through
    execution.ledger.EXIT_REASON_BY_TAG, imported lazily to avoid a
    module-load cycle (execution.ledger imports this module at top level).

    Returns the number of trades rows written.
    """
    from execution.ledger import EXIT_REASON_BY_TAG

    try:
        with _conn() as conn:
            conn.execute("DELETE FROM trades")
            fills = conn.execute(
                "SELECT * FROM fills ORDER BY symbol, transaction_time, id"
            ).fetchall()

            lots: dict[str, list[dict]] = {}
            count = 0
            for f in fills:
                symbol = f["symbol"]
                side = f["side"]
                qty = float(f["qty"])
                if side == "buy":
                    lots.setdefault(symbol, []).append({
                        "fill_id": f["id"], "qty": qty,
                        "price": float(f["price"]), "time": f["transaction_time"],
                    })
                    continue
                if side != "sell":
                    continue

                symbol_lots = lots.setdefault(symbol, [])
                exit_reason = EXIT_REASON_BY_TAG.get(f["tag"], "UNKNOWN")
                remaining = qty
                while remaining > 1e-9 and symbol_lots:
                    lot = symbol_lots[0]
                    matched = min(remaining, lot["qty"])
                    entry_price = lot["price"]
                    exit_price = float(f["price"])
                    pnl_usd = matched * (exit_price - entry_price)
                    pnl_pct = (exit_price / entry_price - 1) if entry_price else None
                    holding_days = (_parse_fill_dt(f["transaction_time"])
                                    - _parse_fill_dt(lot["time"])).days
                    conn.execute("""
                        INSERT INTO trades
                            (symbol, entry_time, exit_time, qty, entry_price,
                             exit_price, pnl_usd, pnl_pct, holding_days,
                             exit_reason, entry_fill_id, exit_fill_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (symbol, lot["time"], f["transaction_time"], matched,
                          entry_price, exit_price, pnl_usd, pnl_pct,
                          holding_days, exit_reason, lot["fill_id"], f["id"]))
                    count += 1
                    lot["qty"] -= matched
                    remaining -= matched
                    if lot["qty"] <= 1e-9:
                        symbol_lots.pop(0)

                if remaining > 1e-9:
                    # No lot left to match — a pre-ledger position. Still
                    # recorded, but unscored (NULL entry/pnl).
                    conn.execute("""
                        INSERT INTO trades
                            (symbol, entry_time, exit_time, qty, entry_price,
                             exit_price, pnl_usd, pnl_pct, holding_days,
                             exit_reason, entry_fill_id, exit_fill_id)
                        VALUES (?, NULL, ?, ?, NULL, ?, NULL, NULL, NULL, ?, NULL, ?)
                    """, (symbol, f["transaction_time"], remaining,
                          float(f["price"]), exit_reason, f["id"]))
                    count += 1
            return count
    except Exception as e:
        print(f"  [DB] rebuild_trades failed: {e}")
        return 0


def load_trades(since: "str | None" = None) -> list[dict]:
    try:
        with _conn() as conn:
            if since:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE exit_time >= ? ORDER BY exit_time",
                    (since,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY exit_time"
                ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"  [DB] load_trades failed: {e}")
        return []


def record_exit(ticker: str, exit_time: str, reason: str, qty=None, price=None,
                run_timestamp: "str | None" = None) -> None:
    """Append-only log of why a position disappeared. Never updates or
    dedupes — a position can legitimately be entered and stopped out more
    than once, and each occurrence is its own row."""
    try:
        with _conn() as conn:
            conn.execute("""
                INSERT INTO exits (ticker, exit_time, reason, qty, price, run_timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ticker, exit_time, reason, qty, price, run_timestamp))
    except Exception as e:
        print(f"  [DB] record_exit failed: {e}")


def recent_stop_exits(days: int, today: "object | None" = None) -> dict:
    """{ticker: exit_time} for stop-driven exits within `days` TRADING days.

    Consumed by the exit engine's cooldown (design C: 10 trading days after a
    stop-out before re-entry is allowed) — a re-entry gate must count trading
    days, not calendar days, or a stop on a Friday only blocks Monday+Tuesday
    instead of two full trading days.
    """
    from datetime import date as _date
    import numpy as np

    today = today or _date.today()
    reasons = ("STOP_INITIAL", "STOP_TRAIL", "BROKER_STOP")
    try:
        with _conn() as conn:
            placeholders = ",".join("?" * len(reasons))
            rows = conn.execute(f"""
                SELECT ticker, exit_time FROM exits
                WHERE reason IN ({placeholders})
                ORDER BY exit_time
            """, reasons).fetchall()
        result: dict = {}
        for row in rows:
            exit_time = row["exit_time"]
            if not exit_time:
                continue
            exit_date = _date.fromisoformat(exit_time[:10])
            if np.busday_count(exit_date, today) < days:
                result[row["ticker"]] = exit_time
        return result
    except Exception as e:
        print(f"  [DB] recent_stop_exits failed: {e}")
        return {}


def load_conviction_rows_needing_backfill(min_age_days: int = 30,
                                          today: "object | None" = None) -> list[dict]:
    """conviction_log rows whose 30d or 60d forward return is old enough to
    score. The 60-day cutoff is always 2×min_age_days — return_30d/return_60d
    were designed as a matched pair, never independently configurable ages.
    """
    from datetime import date as _date, timedelta

    today = today or _date.today()
    cutoff_30 = (today - timedelta(days=min_age_days)).isoformat()
    cutoff_60 = (today - timedelta(days=min_age_days * 2)).isoformat()
    try:
        with _conn() as conn:
            rows = conn.execute("""
                SELECT * FROM conviction_log
                WHERE (return_30d IS NULL AND date <= ?)
                   OR (return_60d IS NULL AND date <= ?)
                ORDER BY date
            """, (cutoff_30, cutoff_60)).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"  [DB] load_conviction_rows_needing_backfill failed: {e}")
        return []
