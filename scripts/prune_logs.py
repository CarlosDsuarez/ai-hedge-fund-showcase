#!/usr/bin/env python3
"""prune_logs.py — retention policy for logs/ and data/cache/.

Dry-run by default: prints what it would delete and exits 0. Pass --apply to
delete. Run it from the repo root, e.g. monthly:

    ./venv/bin/python scripts/prune_logs.py            # list
    ./venv/bin/python scripts/prune_logs.py --apply    # delete

What it considers (and nothing else):

  regime_signals_YYYY-MM-DD[_REGIME].json   date from the name, --older-than
  weekly_report_YYYYMMDD.md                 date from the name, --older-than
  data/cache/*.json                         file mtime, --cache-older-than

--older-than defaults to 90d. regime_strategy_agent re-reads the CURRENT
month's regime_signals files as a cache, so never set it below ~45d for that
category. The cache has a 1h/6h TTL, so --cache-older-than 1d (the default)
only ever removes entries the fetcher would refuse to serve anyway.

What it never touches, whatever the flags: logs/hedge_fund.db, the state
JSONs (position_state, position_metadata, regime_state, last_run),
backtest_v2_results.json, everything under logs/runs/ (the only record of
pre-ledger runs), *.log (live LaunchAgent / scheduler output) and *.lock.
Those are not in the candidate patterns at all, so the guard is structural,
not a deny-list that could drift.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGS_DIR = REPO_ROOT / "logs"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"

CATEGORIES = ("regime_signals", "weekly_report", "cache")

_REGIME_RE = re.compile(r"^regime_signals_(\d{4}-\d{2}-\d{2})(?:_[A-Z]+)?\.json$")
_WEEKLY_RE = re.compile(r"^weekly_report_(\d{8})\.md$")
_AGE_RE = re.compile(r"^(\d+)d$")


def parse_age(text: str) -> timedelta:
    """'90d' -> timedelta(days=90). Days only; anything else is a ValueError."""
    m = _AGE_RE.match(text)
    if not m:
        raise ValueError(f"age must look like '90d', got {text!r}")
    return timedelta(days=int(m.group(1)))


def _dated(path: Path) -> tuple[str, date] | None:
    """(category, date) for a file whose retention date is in its name, else None."""
    m = _REGIME_RE.match(path.name)
    if m:
        return "regime_signals", datetime.strptime(m.group(1), "%Y-%m-%d").date()
    m = _WEEKLY_RE.match(path.name)
    if m:
        return "weekly_report", datetime.strptime(m.group(1), "%Y%m%d").date()
    return None


def candidates(logs_dir: Path, cache_dir: Path, *, older_than: timedelta,
               cache_older_than: timedelta, only: set[str] | None,
               today: date | None = None, now: datetime | None = None) -> list[Path]:
    """Files the policy would delete. Pure: reads names and mtimes, deletes nothing."""
    today = today or date.today()
    now = now or datetime.now()
    wanted = set(CATEGORIES) if only is None else set(only)
    out: list[Path] = []

    if logs_dir.is_dir():
        for path in sorted(logs_dir.iterdir()):
            if not path.is_file():
                continue
            hit = _dated(path)
            if hit is None:
                continue
            category, stamped = hit
            if category in wanted and today - stamped > older_than:
                out.append(path)

    if "cache" in wanted and cache_dir.is_dir():
        for path in sorted(cache_dir.glob("*.json")):
            if path.is_file() and now - datetime.fromtimestamp(path.stat().st_mtime) > cache_older_than:
                out.append(path)

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--older-than", type=parse_age, default=timedelta(days=90),
                    help="age for regime_signals_* / weekly_report_* (from the filename date); default 90d")
    ap.add_argument("--cache-older-than", type=parse_age, default=timedelta(days=1),
                    help="age for data/cache/*.json (from mtime); default 1d")
    ap.add_argument("--only", default=None,
                    help=f"comma-separated subset of {','.join(CATEGORIES)}; default all")
    ap.add_argument("--apply", action="store_true", help="delete instead of listing")
    ap.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = ap.parse_args(argv)

    only = None
    if args.only:
        only = {c.strip() for c in args.only.split(",") if c.strip()}
        unknown = only - set(CATEGORIES)
        if unknown:
            ap.error(f"unknown category {sorted(unknown)}; choose from {CATEGORIES}")

    victims = candidates(args.logs_dir, args.cache_dir, older_than=args.older_than,
                         cache_older_than=args.cache_older_than, only=only)
    verb = "deleting" if args.apply else "would delete"
    for path in victims:
        print(f"{verb}: {path}")
        if args.apply:
            path.unlink()
    print(f"{len(victims)} file(s) {'deleted' if args.apply else 'listed (dry run; pass --apply to delete)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
