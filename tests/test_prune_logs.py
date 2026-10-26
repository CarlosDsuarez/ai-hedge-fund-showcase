"""scripts/prune_logs.py — retention policy for the dated artefacts in logs/ and
the TTL price cache in data/cache/.

Everything the pipeline needs to resume (the SQLite DB, the state JSONs,
logs/runs/, live .log files, the run lock) must be untouchable no matter what
flags are passed; only files whose date is in their NAME (regime_signals_*,
weekly_report_*) or whose mtime says they are stale (data/cache/*.json) are
ever candidates.
"""
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.prune_logs import candidates, main, parse_age  # noqa: E402

TODAY = date(2026, 9, 20)
NOW = datetime(2026, 9, 20, 12, 0, 0)


def _touch(path: Path, age: timedelta | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    if age is not None:
        stamp = (NOW - age).timestamp()
        os.utime(path, (stamp, stamp))
    return path


def _names(paths):
    return sorted(p.name for p in paths)


def test_regime_signals_older_than_the_cutoff_are_candidates(tmp_path):
    logs = tmp_path / "logs"
    _touch(logs / "regime_signals_2026-05-17.json")
    _touch(logs / "regime_signals_2026-06-04_BULL.json")
    _touch(logs / "regime_signals_2026-09-10_BULL.json")   # current month: the agent re-reads it

    out = candidates(logs, tmp_path / "cache", older_than=timedelta(days=45),
                     cache_older_than=timedelta(days=1), only=None, today=TODAY, now=NOW)

    assert _names(out) == ["regime_signals_2026-05-17.json", "regime_signals_2026-06-04_BULL.json"]


def test_weekly_reports_use_the_yyyymmdd_date_in_their_name(tmp_path):
    logs = tmp_path / "logs"
    _touch(logs / "weekly_report_20260523.md")
    _touch(logs / "weekly_report_20260911.md")

    kept_recent = candidates(logs, tmp_path / "cache", older_than=timedelta(days=90),
                             cache_older_than=timedelta(days=1), only=None, today=TODAY, now=NOW)
    everything = candidates(logs, tmp_path / "cache", older_than=timedelta(days=0),
                            cache_older_than=timedelta(days=1), only=None, today=TODAY, now=NOW)

    assert _names(kept_recent) == ["weekly_report_20260523.md"]
    assert _names(everything) == ["weekly_report_20260523.md", "weekly_report_20260911.md"]


def test_files_without_a_date_in_the_name_are_ignored(tmp_path):
    logs = tmp_path / "logs"
    _touch(logs / "regime_signals_latest.json", age=timedelta(days=400))
    _touch(logs / "weekly_report.md", age=timedelta(days=400))
    _touch(logs / "backtest_daily_results.json", age=timedelta(days=400))

    out = candidates(logs, tmp_path / "cache", older_than=timedelta(days=0),
                     cache_older_than=timedelta(days=1), only=None, today=TODAY, now=NOW)

    assert out == []


def test_protected_state_files_are_never_candidates(tmp_path):
    logs = tmp_path / "logs"
    old = timedelta(days=400)
    for name in ("hedge_fund.db", "position_state.json", "position_metadata.json",
                 "regime_state.json", "last_run.json", "backtest_v2_results.json",
                 "launchagent.log", "scheduler.log", "main.lock"):
        _touch(logs / name, age=old)
    _touch(logs / "runs" / "2026-05-25_08-31.json", age=old)

    out = candidates(logs, tmp_path / "cache", older_than=timedelta(days=0),
                     cache_older_than=timedelta(days=0), only=None, today=TODAY, now=NOW)

    assert out == []


def test_cache_files_are_selected_by_mtime(tmp_path):
    cache = tmp_path / "cache"
    _touch(cache / "get_stock_price_AAPL_9f61f6e8.json", age=timedelta(days=3))
    _touch(cache / "get_stock_info_MSFT_ad6060e1.json", age=timedelta(hours=2))

    out = candidates(tmp_path / "logs", cache, older_than=timedelta(days=90),
                     cache_older_than=timedelta(days=1), only=None, today=TODAY, now=NOW)

    assert _names(out) == ["get_stock_price_AAPL_9f61f6e8.json"]


def test_only_restricts_the_categories_considered(tmp_path):
    logs = tmp_path / "logs"
    _touch(logs / "regime_signals_2026-05-17.json")
    _touch(logs / "weekly_report_20260523.md")
    _touch(tmp_path / "cache" / "get_stock_price_AAPL_9f61f6e8.json", age=timedelta(days=3))

    out = candidates(logs, tmp_path / "cache", older_than=timedelta(days=0),
                     cache_older_than=timedelta(days=1), only={"weekly_report"},
                     today=TODAY, now=NOW)

    assert _names(out) == ["weekly_report_20260523.md"]


def test_dry_run_lists_without_deleting_and_apply_deletes(tmp_path, capsys):
    logs = tmp_path / "logs"
    cache = tmp_path / "cache"
    stale = _touch(logs / "regime_signals_2020-01-06.json")
    keep = _touch(logs / "position_state.json", age=timedelta(days=400))
    argv = ["--logs-dir", str(logs), "--cache-dir", str(cache), "--older-than", "90d"]

    assert main(argv) == 0
    assert stale.exists() and keep.exists()
    assert "regime_signals_2020-01-06.json" in capsys.readouterr().out

    assert main(argv + ["--apply"]) == 0
    assert not stale.exists()
    assert keep.exists()


def test_parse_age_accepts_days_only():
    assert parse_age("90d") == timedelta(days=90)
    assert parse_age("0d") == timedelta(days=0)
    for bad in ("90", "1w", "-3d", "d"):
        try:
            parse_age(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should be rejected")


def test_missing_directories_are_not_an_error(tmp_path):
    logs = tmp_path / "logs"
    _touch(logs / "regime_signals_2020-01-06.json")

    out = candidates(logs, tmp_path / "does-not-exist", older_than=timedelta(days=90),
                     cache_older_than=timedelta(days=1), only=None, today=TODAY, now=NOW)

    assert _names(out) == ["regime_signals_2020-01-06.json"]
