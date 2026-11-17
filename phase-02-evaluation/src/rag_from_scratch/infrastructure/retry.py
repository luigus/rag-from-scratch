"""Retry with exponential backoff for rate limits and temporary server errors.

Free API tiers have low requests-per-minute limits, and the evaluation sends
many requests in a short time, so a 429 is expected, not exceptional.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _status_code(error: Exception) -> int | None:
    # Gemini errors expose `.code`, OpenAI errors expose `.status_code`
    for attr in ("code", "status_code"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    return None


async def with_retry(
    call: Callable[[], Awaitable[T]], *, attempts: int = 6, base_delay: float = 2.0
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except Exception as e:
            if _status_code(e) not in RETRYABLE_STATUS or attempt == attempts:
                raise
            # 2s, 4s, 8s, ... plus jitter, so parallel requests don't retry in sync
            delay = base_delay * 2 ** (attempt - 1) + random.uniform(0, 1)
            logger.warning(
                "API returned %s, retrying in %.0fs (attempt %d/%d)",
                _status_code(e), delay, attempt, attempts,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")
