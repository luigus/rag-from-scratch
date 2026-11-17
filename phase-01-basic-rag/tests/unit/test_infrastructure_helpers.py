import asyncio
import math

import pytest

from rag_from_scratch.infrastructure import retry
from rag_from_scratch.infrastructure.vectors import l2_normalize


def test_l2_normalize_gives_length_one():
    v = l2_normalize([3.0, 4.0])
    assert v == pytest.approx([0.6, 0.8])
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0)


def test_l2_normalize_keeps_zero_vector():
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


class FakeAPIError(Exception):
    def __init__(self, code: int) -> None:
        self.code = code


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(retry.asyncio, "sleep", instant)


def test_retries_rate_limit_then_succeeds():
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeAPIError(429)
        return "ok"

    assert asyncio.run(retry.with_retry(flaky)) == "ok"
    assert calls["n"] == 3


def test_does_not_retry_client_errors():
    calls = {"n": 0}

    async def bad_request() -> str:
        calls["n"] += 1
        raise FakeAPIError(400)

    with pytest.raises(FakeAPIError):
        asyncio.run(retry.with_retry(bad_request))
    assert calls["n"] == 1


def test_gives_up_after_the_last_attempt():
    async def always_busy() -> str:
        raise FakeAPIError(503)

    with pytest.raises(FakeAPIError):
        asyncio.run(retry.with_retry(always_busy, attempts=3))
