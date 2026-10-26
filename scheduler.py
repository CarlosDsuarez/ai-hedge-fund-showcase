import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

from utils.env import load_project_env

load_project_env()

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "scheduler.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

MIDDAY_RUN = os.getenv("MIDDAY_RUN", "false").lower() == "true"
PYTHON = sys.executable
MAIN_PY = os.path.join(os.path.dirname(__file__), "main.py")
REPORT_PY = os.path.join(os.path.dirname(__file__), "reports", "weekly_report.py")

# All times below are ET (America/New_York). The morning run sits 15 minutes
# after the 09:30 ET open so quotes are live instead of pre-market.
MORNING_HOUR = 9
MORNING_MINUTE = 45
MIDDAY_HOUR = 13
MIDDAY_MINUTE = 0
REPORT_HOUR = 17
REPORT_MINUTE = 0

CATCHUP_WINDOW_MINUTES = 30


def _et_now():
    import zoneinfo
    et = zoneinfo.ZoneInfo("America/New_York")
    return datetime.now(et)


def _next_weekday_at(hour: int, minute: int) -> datetime:
    now = _et_now()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _run_main():
    logger.info("Running main.py...")
    result = subprocess.run([PYTHON, MAIN_PY], capture_output=False)
    if result.returncode == 0:
        logger.info("main.py exited with code 0 (success or skipped due to lock)")
    else:
        logger.warning(f"main.py exited with code {result.returncode} — check output above")


def _run_report():
    logger.info("Running weekly report...")
    result = subprocess.run([PYTHON, REPORT_PY], capture_output=False)
    logger.info(f"weekly_report.py exited with code {result.returncode}")


def _show_schedule():
    morning = _next_weekday_at(MORNING_HOUR, MORNING_MINUTE)
    print(f"Next morning run:  {morning.strftime('%Y-%m-%d %H:%M ET')}")
    if MIDDAY_RUN:
        midday = _next_weekday_at(MIDDAY_HOUR, MIDDAY_MINUTE)
        print(f"Next midday run:   {midday.strftime('%Y-%m-%d %H:%M ET')}")
    next_friday = _next_weekday_at(REPORT_HOUR, REPORT_MINUTE)
    while next_friday.weekday() != 4:
        next_friday += timedelta(days=1)
    print(f"Next weekly report:{next_friday.strftime('%Y-%m-%d %H:%M ET')}")


def _check_catchup():
    now = _et_now()
    if now.weekday() >= 5:
        return

    morning = now.replace(hour=MORNING_HOUR, minute=MORNING_MINUTE, second=0, microsecond=0)
    diff = (now - morning).total_seconds() / 60
    if 0 <= diff <= CATCHUP_WINDOW_MINUTES:
        logger.info(f"Catchup: within {int(diff)}m of morning run, running now.")
        _run_main()
        return

    if MIDDAY_RUN:
        midday = now.replace(hour=MIDDAY_HOUR, minute=MIDDAY_MINUTE, second=0, microsecond=0)
        diff_mid = (now - midday).total_seconds() / 60
        if 0 <= diff_mid <= CATCHUP_WINDOW_MINUTES:
            logger.info(f"Catchup: within {int(diff_mid)}m of midday run, running now.")
            _run_main()


def main():
    parser = argparse.ArgumentParser(description="AI Hedge Fund Scheduler")
    parser.add_argument("--dry-run", action="store_true", help="Show next scheduled times and exit")
    parser.add_argument("--catchup", action="store_true", help="Run immediately if within 30 min of scheduled time")
    parser.add_argument("--report", action="store_true", help="Generate weekly report now")
    args = parser.parse_args()

    if args.dry_run:
        _show_schedule()
        return

    if args.report:
        _run_report()
        return

    if args.catchup:
        _check_catchup()

    logger.info("Scheduler started. Monitoring for scheduled runs...")

    _morning_ran_date = None
    _midday_ran_date = None
    _report_ran_date = None

    while True:
        now = _et_now()
        today = now.date()
        is_weekday = now.weekday() < 5

        if is_weekday and _morning_ran_date != today:
            morning = now.replace(hour=MORNING_HOUR, minute=MORNING_MINUTE, second=0, microsecond=0)
            diff = (now - morning).total_seconds() / 60
            if 0 <= diff <= CATCHUP_WINDOW_MINUTES:
                late = f" ({int(diff)}m late)" if diff >= 1 else ""
                logger.info(f"Triggering morning run{late}.")
                _run_main()
                _morning_ran_date = today
                time.sleep(61)

        if MIDDAY_RUN and is_weekday and _midday_ran_date != today:
            midday = now.replace(hour=MIDDAY_HOUR, minute=MIDDAY_MINUTE, second=0, microsecond=0)
            diff = (now - midday).total_seconds() / 60
            if 0 <= diff <= CATCHUP_WINDOW_MINUTES:
                late = f" ({int(diff)}m late)" if diff >= 1 else ""
                logger.info(f"Triggering midday run{late}.")
                _run_main()
                _midday_ran_date = today
                time.sleep(61)

        if is_weekday and now.weekday() == 4 and _report_ran_date != today:
            report_time = now.replace(hour=REPORT_HOUR, minute=REPORT_MINUTE, second=0, microsecond=0)
            diff = (now - report_time).total_seconds() / 60
            if 0 <= diff <= CATCHUP_WINDOW_MINUTES:
                late = f" ({int(diff)}m late)" if diff >= 1 else ""
                logger.info(f"Triggering Friday weekly report{late}.")
                _run_report()
                _report_ran_date = today
                time.sleep(61)

        time.sleep(30)


if __name__ == "__main__":
    main()
