"""Shared async retry + timeout helper for LLM calls."""

import asyncio
import csv
import os
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from langchain_core.messages import HumanMessage
from openai import APIConnectionError, RateLimitError, APIError

from src.utils.logger import Logger

_error_logger = Logger(os.path.join("log", "errors"))


def resolve_openai_seed(seed: Optional[int] = None) -> Optional[int]:
    """Resolve seed from explicit argument or OPENAI_SEED env var."""
    if seed is not None:
        return seed

    env_seed = os.getenv("OPENAI_SEED")
    if env_seed is None:
        return None

    try:
        return int(env_seed)
    except (TypeError, ValueError):
        return None


def _message_seed(message: HumanMessage) -> Optional[int]:
    seed = getattr(message, "seed", None)
    if seed is not None:
        return seed

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        seed = additional_kwargs.get("seed")
        if seed is not None:
            return seed

    return None


def _with_seed(message: HumanMessage, seed: int) -> HumanMessage:
    if _message_seed(message) is not None:
        return message

    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    additional_kwargs["seed"] = seed

    if hasattr(message, "model_copy"):
        return message.model_copy(update={"additional_kwargs": additional_kwargs}, deep=True)
    if hasattr(message, "copy"):
        return message.copy(update={"additional_kwargs": additional_kwargs}, deep=True)

    return HumanMessage(content=message.content, additional_kwargs=additional_kwargs)


async def ainvoke_with_seed(llm: Any, payload: Any, seed: Optional[int] = None):
    """Invoke llm.ainvoke with optional OPENAI_SEED-based message seeding."""
    resolved_seed = resolve_openai_seed(seed)
    if resolved_seed is None:
        return await llm.ainvoke(payload)

    if isinstance(payload, str):
        return await llm.ainvoke([HumanMessage(content=payload, seed=resolved_seed)])

    if isinstance(payload, HumanMessage):
        return await llm.ainvoke([_with_seed(payload, resolved_seed)])

    if isinstance(payload, list):
        seeded_payload = [
            _with_seed(message, resolved_seed) if isinstance(message, HumanMessage) else message
            for message in payload
        ]
        return await llm.ainvoke(seeded_payload)

    return await llm.ainvoke(payload)


async def call_with_retry(
    coro_factory: Callable[[], Awaitable],
    *,
    retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    timeout: float = 90.0,
    label: str = "LLM",
):
    """Execute an async call with timeout + exponential backoff retries.

    Args:
        coro_factory: Zero-arg callable returning the awaited coroutine (e.g., lambda: llm.ainvoke(...)).
        retries: Number of retry attempts (total attempts = retries + 1).
        initial_delay: Seconds to wait before first retry.
        max_delay: Max backoff seconds between retries.
        timeout: Per-attempt timeout in seconds.
        label: Component label for logs.

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
            _error_logger.error_log(label, attempt + 1, type(e).__name__, str(e))

            if attempt == retries:
                raise

            if isinstance(e, asyncio.TimeoutError):
                timeout += 30
                _log(f"[{label} Retry {attempt + 1}/{retries}] {type(e).__name__}. Extending timeout to {timeout:.0f}s. Retrying in {min(delay, max_delay):.1f}s...")
            else:
                _log(f"[{label} Retry {attempt + 1}/{retries}] {type(e).__name__}. Retrying in {min(delay, max_delay):.1f}s...")

            wait_time = min(delay, max_delay)
            await asyncio.sleep(wait_time)
            delay *= 2

    if last_exception:
        raise last_exception
