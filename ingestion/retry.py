"""Retry utilities for ingestion operations."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def execute_with_retry(
    operation: Callable[[], T],
    max_retries: int,
    backoff_factor: float,
    logger: logging.Logger,
    operation_name: str,
    retry_if: Callable[[Exception], bool] | None = None,
) -> T:
    """Execute an operation with exponential backoff."""
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if retry_if is not None and not retry_if(exc):
                logger.error("%s failed without retry: %s", operation_name, exc)
                raise
            attempt += 1
            if attempt > max_retries:
                logger.error("%s failed after %s retries: %s", operation_name, max_retries, exc)
                raise
            sleep_seconds = backoff_factor ** attempt
            logger.warning(
                "%s failed on attempt %s/%s. Retrying in %.2f seconds. Error: %s",
                operation_name,
                attempt,
                max_retries,
                sleep_seconds,
                exc,
            )
            time.sleep(sleep_seconds)
