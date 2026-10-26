"""Position state and the exit engine.

One engine, one stop per position: an initial stop STOP_ATR_MULT × ATR14 below
entry, a chandelier trail TRAIL_ATR_MULT × ATR14 below the highest close that
only ever rises, a rank-decay exit, a PM SELL exit, a hard time cap — and no
take-profit partials (it replaced an earlier take-profit ladder).

State lives in db.database.position_state (source of truth). The JSON file
logs/position_state.json is written as a mirror after every save because
dashboard/app.py, chat/context.py and reports/weekly_report.py still read it,
and it is the one-time migration source on the first run after deploy.

execution.alpaca is imported inside the functions that need it so the suite
can patch `execution.alpaca.<name>` by attribute (see trade_executor.py).
"""
import json
import os
from datetime import date, datetime

import config
from db import database as db

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "position_state.json")


# ── persistence ──────────────────────────────────────────────────────────────

def _legacy_to_engine_record(record: dict) -> dict:
    """A TP-ladder record from logs/position_state.json as an engine record.

    Controller ruling (task 3, fix round 1): initial_stop is recomputed as
    entry - STOP_ATR_MULT × ATR14 from the stored ATR, so an untouched
    position gets the designed STOP_ATR_MULT×ATR room instead of the old ladder's
    tighter stop (sl_price is the fallback when the ATR is missing). A position whose
    ladder already raised its stop (breakeven after TP1, or a TP2 fill) keeps
    that raised level as trail_stop — it was earned, and the engine labels
    such an exit STOP_TRAIL. highest_close starts at entry; the ratchet is a
    max(), so the first engine pass can only lift it. Ladder fields dropped.
    """
    if "initial_stop" in record and "trail_stop" in record:
        return dict(record)
    entry = record.get("entry_price")
    atr = float(record.get("atr_14") or 0.0)
    sl_price = record.get("sl_price")
    if entry is not None and atr > 0:
        initial = round(float(entry) - config.STOP_ATR_MULT * atr, 2)
    else:
        initial = sl_price
    raised = bool(record.get("sl_is_breakeven") or record.get("tp1_filled")
                  or record.get("tp2_filled"))
    trail = sl_price if (raised and sl_price is not None) else initial
    return {
        "entry_price":   entry,
        "entry_time":    record.get("entry_time"),
        "atr_14":        record.get("atr_14"),
        "initial_stop":  initial,
        "trail_stop":    trail,
        "highest_close": entry,
        "qty_total":     record.get("qty_total"),
        "qty_remaining": record.get("qty_remaining"),
        "decay_count":   0,
    }


def load_position_state() -> dict:
    """{ticker: record}. DB first; an empty DB with a JSON file present is the
    first run after deploy, and the JSON ladder is migrated into the DB once."""
    state = db.load_position_state()
    if state:
        return state
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE) as f:
            legacy = json.load(f) or {}
    except Exception as e:
        print(f"  [State] could not read {STATE_FILE}: {e}")
        return {}
    if not legacy:
        return {}
    migrated = {t: _legacy_to_engine_record(r) for t, r in legacy.items()}
    db.save_position_state(migrated)
    print(f"  [State] migrated {len(migrated)} position record(s) from "
          f"position_state.json into the DB (one-time)")
    return migrated


def save_position_state(state: dict) -> None:
    """Write the DB, then mirror to JSON.

    The mirror carries an extra "sl_price" = trail_stop per record: it is the
    key the dashboard, the chat context and the weekly report read for "the
    stop", and rewriting three readers for a rename buys nothing.
    """
    db.save_position_state(state)
    mirror = {}
    for ticker, rec in state.items():
        row = dict(rec)
        if row.get("trail_stop") is not None:
            row["sl_price"] = row["trail_stop"]
        mirror[ticker] = row
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(mirror, f, indent=2, default=str)


# ── records ──────────────────────────────────────────────────────────────────

def _build_record(entry_price: float, qty: int, atr_14: float, entry_time: str) -> dict:
    initial_stop = round(entry_price - config.STOP_ATR_MULT * atr_14, 2)
    return {
        "entry_price":          round(entry_price, 2),
        "entry_time":           entry_time,
        "atr_14":               round(atr_14, 4),
        "initial_stop":         initial_stop,
        "trail_stop":           initial_stop,
        "highest_close":        round(entry_price, 2),
        "qty_total":            int(qty),
        "qty_remaining":        int(qty),
        "decay_count":          0,
        "broker_stop_order_id": None,
        "broker_stop_price":    None,
        "broker_stop_qty":      None,
    }


def register_new_position(ticker: str, entry_price: float, qty: int,
                          atr_14: float, *, entry_time: "str | None" = None) -> dict:
    """Write the engine record for a fresh fill. The stop multiplier is
    config.STOP_ATR_MULT and nothing else — the ATR_MULTIPLIER_SL env var that
    used to override it is gone, so the live stop cannot drift from the value
    the backtest validated."""
    if not atr_14 or atr_14 <= 0:
        raise ValueError(f"ATR inválido para {ticker}: {atr_14}")
    record = _build_record(entry_price, qty, atr_14, entry_time or datetime.now().isoformat())
    state = load_position_state()
    state[ticker] = record
    save_position_state(state)
    return record


def register_on_the_fly(ticker: str, pos: dict, atr_14: float, entry_time: str) -> dict:
    """Engine record for a broker position the pipeline never registered.

    Replaces the legacy exit branch (a simple-mean ATR stop + flat percentage
    rules, both since deleted) that used to govern such positions for their
    whole life. The record is built off avg_entry_price exactly as a fresh
    fill would be; the caller runs update_trail next, which is what lifts the
    chandelier to the actual highest close.
    """
    avg_entry = float(pos.get("avg_entry_price") or pos.get("current_price") or 0)
    qty = int(float(pos.get("qty") or 0))
    if avg_entry <= 0 or qty <= 0:
        raise ValueError(f"{ticker}: cannot register without entry price and qty")
    if not atr_14 or atr_14 <= 0:
        raise ValueError(f"ATR inválido para {ticker}: {atr_14}")
    return _build_record(avg_entry, qty, atr_14, entry_time)


def reduce_position_qty(ticker: str, sold_qty: int) -> "dict | None":
    """Shrink a tracked position after a sell placed outside the engine (a chat
    trim). Both counters move so qty_remaining keeps matching the broker: it is
    what the ledger records for an untagged exit and what a stateless remainder
    would otherwise be re-registered at.

    Returns the updated record, or None when the position is untracked or the
    sell closed it out (the record is then deleted, so the next run treats the
    name as flat rather than as a phantom holding).
    """
    state = load_position_state()
    record = state.get(ticker)
    if not record:
        return None

    sold = max(int(sold_qty), 0)
    remaining = int(record.get("qty_remaining", 0)) - sold
    if remaining <= 0:
        del state[ticker]
        save_position_state(state)
        return None

    record["qty_remaining"] = remaining
    record["qty_total"] = max(int(record.get("qty_total", remaining)) - sold, remaining)
    state[ticker] = record
    save_position_state(state)
    return record


# ── pure helpers (shared semantics with backtest/reference_backtest.py) ──────────

def update_trail(rec: dict, close: float) -> dict:
    """Chandelier ratchet. Returns a new record; never lowers a stop.

    ATR is the value stored at entry: a stop that widens because volatility
    rose after entry is a stop that moved against the trade.
    """
    out = dict(rec)
    highest = max(float(rec.get("highest_close") or 0.0), float(close))
    trail = float(rec.get("trail_stop") or 0.0)
    atr = float(rec.get("atr_14") or 0.0)
    if atr > 0:
        # Without an ATR the chandelier is undefined; ratcheting to
        # `highest - 0` would park the stop AT the market and sell on arrival.
        trail = max(trail, highest - config.TRAIL_ATR_MULT * atr)
    out["highest_close"] = round(highest, 4)
    out["trail_stop"] = round(trail, 4)
    return out


def rank_percentile(ticker: str, score_map: dict) -> "float | None":
    """Cross-sectional percentile of `ticker` among the finite scores, as
    pandas rank(pct=True) computes it (average rank of ties / n) — the same
    number the backtest's `pctl` panel holds. None when it cannot be ranked."""
    scores = {t: float(v) for t, v in (score_map or {}).items()
              if v is not None and v == v}          # drops None and NaN
    if ticker not in scores or not scores:
        return None
    s = scores[ticker]
    below = sum(1 for v in scores.values() if v < s)
    ties = sum(1 for v in scores.values() if v == s)
    return (below + (ties + 1) / 2) / len(scores)


def _days_held(entry_time: "str | None", today: date) -> int:
    if not entry_time:
        return 0
    try:
        return (today - datetime.fromisoformat(str(entry_time)).date()).days
    except ValueError:
        return 0


def _scores_usable(score_map: "dict | None") -> bool:
    """A strategy run that failed scores every name 0.0 (or nothing at all).
    Counting that as a weak day would sell the book on a data outage."""
    if not score_map:
        return False
    return any(v is not None and v == v and float(v) != 0.0 for v in score_map.values())


# ── the engine ───────────────────────────────────────────────────────────────

def _evaluate(ticker: str, rec: dict, price: float, *, today: date,
              score_map: dict, pm_sells: dict) -> "tuple[str | None, dict]":
    """One position's verdict, in priority order, plus the (possibly
    updated) record. Only the rank-decay branch mutates the record — and only
    when no higher-priority exit fired, so a stopped position's counter is
    never touched on its way out.

    STOP → TIME_CAP → PM_SELL → RANK_DECAY. The stop goes first because it is
    the only exit whose level was promised to the broker; the others are
    opinions about whether the name still deserves its slot.
    """
    trail = float(rec.get("trail_stop") or 0.0)
    initial = float(rec.get("initial_stop") or 0.0)
    if trail > 0 and price <= trail:
        return ("STOP_INITIAL" if trail == initial else "STOP_TRAIL"), rec
    if _days_held(rec.get("entry_time"), today) >= config.MAX_HOLDING_DAYS:
        return "TIME_CAP", rec
    if ticker in pm_sells and int(pm_sells[ticker] or 0) >= config.PM_SELL_MIN_CONVICTION:
        return "PM_SELL", rec
    if _scores_usable(score_map):
        pctl = rank_percentile(ticker, score_map)
        if pctl is not None:
            count = int(rec.get("decay_count") or 0)
            new_count = count + 1 if pctl < config.RANK_DECAY_PCTL else 0
            if new_count != count:
                rec = {**rec, "decay_count": new_count}
            if new_count >= config.RANK_DECAY_DAYS:
                return "RANK_DECAY", rec
    return None, rec


def run_exit_engine(positions: list, *, score_map: "dict | None", pm_sells: dict,
                    metadata: dict, today: "date | None" = None,
                    dry_run: bool = False) -> tuple[set, list]:
    """Evaluate every long broker position and execute at most one exit each.

    positions : Alpaca GET /v2/positions rows (symbol uses "BRK.B" form).
    score_map : {ticker: composite} from this run's strategy agent; empty or
                all-zero means the rank-decay counters are left untouched.
    pm_sells  : {ticker: conviction} — PM SELL verdicts on HELD names only,
                and only from a non-degraded PM result (main.run_exits builds it).
    metadata  : {ticker: {"entry_date": ...}} — the entry time for a position
                that has to be registered on the fly.

    Returns (exited_tickers, closed) where each closed row is
    {"ticker", "reason", "qty", "price", "pnl_pct"}. State is persisted
    after EVERY mutation (register, trail ratchet, decay tick, exit) so a
    crash mid-loop cannot leave the DB claiming a position the broker has
    already sold. dry_run evaluates and prints; it neither sells nor writes.
    """
    from data.fetcher import get_atr
    from execution.trade_executor import sell_and_confirm

    today = today or date.today()
    state = load_position_state()
    exited: set = set()
    closed: list = []
    run_ts = datetime.now().isoformat()

    def _persist():
        if not dry_run:
            save_position_state(state)

    def _process_position(pos: dict) -> None:
        """Everything for ONE broker row. Raises propagate to the loop below,
        which logs and moves on — never past the fills already made."""
        ticker = (pos.get("symbol") or "").replace(".", "-")
        if not ticker or float(pos.get("market_value") or 0) <= 0:
            return                                     # long-only: shorts are not ours
        qty = int(float(pos.get("qty") or 0))
        price = float(pos.get("current_price") or 0)
        if qty <= 0 or price <= 0:
            return

        rec = state.get(ticker)
        if rec is None:
            atr = get_atr(ticker, 14, "3mo")
            if not atr or atr <= 0:
                print(f"  [Exit] {ticker}: no state record and no ATR — left unmanaged this run")
                return
            entry_time = (metadata.get(ticker) or {}).get("entry_date") or datetime.now().isoformat()
            try:
                rec = register_on_the_fly(ticker, pos, atr, entry_time)
            except ValueError as e:
                print(f"  [Exit] {ticker}: {e}")
                return
            state[ticker] = rec
            _persist()
            print(f"  [Exit] {ticker}: registered on the fly — entry ${rec['entry_price']:.2f} "
                  f"stop ${rec['initial_stop']:.2f} (ATR {atr:.2f})")

        ratcheted = update_trail(rec, price)
        if ratcheted != rec:
            state[ticker] = rec = ratcheted
            _persist()

        reason, updated = _evaluate(ticker, rec, price, today=today,
                                    score_map=score_map or {}, pm_sells=pm_sells or {})
        if updated is not rec:
            state[ticker] = rec = updated
            _persist()
        if reason is None:
            return

        entry_price = float(rec.get("entry_price") or 0.0)
        if dry_run:
            print(f"  [DryRun] Would SELL {ticker} qty={qty} @ ~${price:.2f} — {reason} "
                  f"(stop ${float(rec.get('trail_stop') or 0):.2f}, "
                  f"decay {int(rec.get('decay_count') or 0)})")
            return

        filled, fill_price = sell_and_confirm(ticker, qty, tag=reason.lower())
        if filled == 0:
            print(f"  [Exit] {ticker}: {reason} sell did not fill — position and state kept")
            return

        db.record_exit(ticker, exit_time=datetime.now().isoformat(), reason=reason,
                       qty=filled, price=fill_price, run_timestamp=run_ts)
        pnl_pct = (fill_price / entry_price - 1.0) if entry_price > 0 else 0.0
        closed.append({"ticker": ticker, "reason": reason, "qty": filled,
                       "price": fill_price, "pnl_pct": round(pnl_pct, 4)})
        if filled < qty:
            # The remainder is still owned. Keep the record (shrunk) so the
            # stop audit protects it and the next run finishes the exit.
            rec = dict(rec)
            rec["qty_remaining"] = qty - filled
            rec["qty_total"] = max(int(rec.get("qty_total") or qty) - filled, qty - filled)
            state[ticker] = rec
            _persist()
            print(f"  [Exit] {ticker}: {reason} PARTIAL {filled}/{qty} @ ${fill_price:.2f} — "
                  f"{qty - filled} share(s) still held")
            return
        del state[ticker]
        _persist()
        exited.add(ticker)
        print(f"  [Exit] {ticker}: {reason} — sold {filled} @ ${fill_price:.2f} ({pnl_pct:+.1%})")

    for pos in positions:
        try:
            _process_position(pos)
        except Exception as e:
            # Fault isolation per position: a malformed row, a data-feed
            # error or a DB hiccup on one name must not abort the pass and
            # discard the sells that already executed at the broker.
            print(f"  [Exit] {(pos.get('symbol') or '?')}: skipped this run — "
                  f"{type(e).__name__}: {e}")

    return exited, closed
