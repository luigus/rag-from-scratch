"""Retry with exponential backoff for transient failures.

Transient means "trying again later will probably work":
- rate limits (429): free API tiers allow few requests per minute, and the
  evaluation sends many requests in a short time;
- server errors (5xx);
- network errors: a connection reset by the server, a dropped connection;
- calls that hang: every attempt has a timeout, so one stuck request can't
  block the whole run.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# SDK errors that wrap network problems without being an OSError
RETRYABLE_ERROR_NAMES = {"APIConnectionError", "APITimeoutError", "ServerDisconnectedError"}


def _status_code(error: Exception) -> int | None:
    # Gemini errors expose `.code`, OpenAI errors expose `.status_code`
    for attr in ("code", "status_code"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_transient(error: Exception) -> bool:
    if "PerDay" in str(error):
        # a daily quota is used up: retrying in a few seconds can't work, so fail fast
        return False
    if _status_code(error) in RETRYABLE_STATUS:
        return True
    # ConnectionResetError, aiohttp.ClientOSError and TimeoutError are all OSErrors
    if isinstance(error, OSError | TimeoutError):
        return True
    return type(error).__name__ in RETRYABLE_ERROR_NAMES


def _describe(error: Exception) -> str:
    status = _status_code(error)
    return f"status {status}" if status else type(error).__name__


async def with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    attempts: int = 6,
    base_delay: float = 2.0,
    timeout: float = 90.0,
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            async with asyncio.timeout(timeout):
                return await call()
        except Exception as e:
            if not is_transient(e) or attempt == attempts:
                raise
            # 2s, 4s, 8s, ... plus jitter, so parallel requests don't retry in sync
            delay = base_delay * 2 ** (attempt - 1) + random.uniform(0, 1)
            logger.warning(
                "API call failed (%s), retrying in %.0fs (attempt %d/%d)",
                _describe(e),
                delay,
                attempt,
                attempts,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")
