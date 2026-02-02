"""Shared async retry + timeout helper for LLM calls."""

import asyncio
import csv
import os
from datetime import datetime
from typing import Awaitable, Callable, Optional

from openai import APIConnectionError, RateLimitError, APIError


async def call_with_retry(
    coro_factory: Callable[[], Awaitable],
    *,
    retries: int = 5,
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

    def _extract_usage(response) -> Optional[dict]:
        # LangChain messages typically store usage in usage_metadata or response_metadata.
        usage = getattr(response, "usage_metadata", None)
        if isinstance(usage, dict):
            return usage

        meta = getattr(response, "response_metadata", None)
        if isinstance(meta, dict):
            usage = meta.get("usage") or meta.get("token_usage")
            if isinstance(usage, dict):
                return usage
            if any(k in meta for k in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")):
                return meta

        usage = getattr(response, "usage", None)
        if isinstance(usage, dict):
            return usage

        return None

    def _log_usage(usage: dict) -> None:
        log_dir = os.path.join("log", "tokens")
        os.makedirs(log_dir, exist_ok=True)
        date_stamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"tokens_{date_stamp}.log")
        timestamp = datetime.now().isoformat(timespec="seconds")
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", ""))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", ""))
        total_tokens = usage.get("total_tokens", "")
        try:
            file_exists = os.path.exists(log_file)
            with open(log_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "label", "input_tokens", "output_tokens", "total_tokens"])
                writer.writerow([timestamp, label, input_tokens, output_tokens, total_tokens])
        except Exception as e:
            _log(f"[{label} UsageLogError] {type(e).__name__}: {e}")

    delay = initial_delay
    last_exception = None

    def _log(msg: str):
        if logger:
            logger(msg)
        elif progress:
            print(msg)

    for attempt in range(retries + 1):
        try:
            response = await asyncio.wait_for(coro_factory(), timeout=timeout)
            usage = _extract_usage(response)
            if usage:
                _log_usage(usage)
            return response
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
