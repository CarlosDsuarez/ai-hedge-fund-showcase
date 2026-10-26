"""Shared retry helper (audit Fase B, prompt 12).

Single implementation reused by data/fetcher.py (yfinance) and
execution/alpaca.py (broker REST). Fixed delays, optional retriable
predicate so callers can restrict retries to transient failures.
"""

import logging
import time
from typing import TypeVar
from collections.abc import Callable

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


def with_retry(
    fn: Callable[[], _T],
    label: str,
    *,
    max_retries: int = 3,
    delays: tuple = (2, 5),
    retriable: Callable[[Exception], bool] | None = None,
) -> _T:
    """Call fn() up to max_retries times; raise the last exception on final failure.

    retriable: predicate deciding whether an exception is worth retrying.
    None (default) retries any Exception — the historical fetcher behavior.
    Non-retriable exceptions propagate immediately.
    """
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if retriable is not None and not retriable(exc):
                raise
            last_exc = exc
            if attempt < max_retries:
                delay = delays[min(attempt - 1, len(delays) - 1)]
                logger.warning("[Retry] %s attempt %d/%d failed (%s), retrying in %ds",
                               label, attempt, max_retries, exc, delay)
                time.sleep(delay)
            else:
                logger.error("[Retry] %s failed after %d attempts: %s",
                             label, max_retries, exc)
    raise last_exc
