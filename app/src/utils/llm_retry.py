"""Shared async retry + timeout helper for LLM calls."""

import asyncio
from typing import Awaitable, Callable, Optional

from openai import APIConnectionError, RateLimitError, APIError


async def call_with_retry(
    coro_factory: Callable[[], Awaitable],
    *,
    retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    timeout: float = 90.0,
    progress: bool = False,
    label: str = "LLM",
    logger: Optional[Callable[[str], None]] = None,
):
    """Execute an async call with timeout + exponential backoff retries.

    Args:
        coro_factory: Zero-arg callable returning the awaited coroutine (e.g., lambda: llm.ainvoke(...)).
        retries: Number of retry attempts (total attempts = retries + 1).
        initial_delay: Seconds to wait before first retry.
        max_delay: Max backoff seconds between retries.
        timeout: Per-attempt timeout in seconds.
        progress: Whether to print progress logs when logger is not provided.
        label: Component label for logs.
        logger: Optional logging callable; defaults to print when progress is True.

    Raises:
        APIConnectionError, RateLimitError, APIError, asyncio.TimeoutError, or the last caught exception after retries.
    """

    delay = initial_delay
    last_exception = None

    def _log(msg: str):
        if logger:
            logger(msg)
        elif progress:
            print(msg)

    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(coro_factory(), timeout=timeout)
        except (APIConnectionError, RateLimitError, APIError, asyncio.TimeoutError) as e:
            last_exception = e

            if attempt == retries:
                raise

            wait_time = min(delay, max_delay)
            _log(f"[{label} Retry {attempt + 1}/{retries}] {type(e).__name__}. Retrying in {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            delay *= 2

    if last_exception:
        raise last_exception
