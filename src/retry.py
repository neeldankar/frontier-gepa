"""Retry decorators for transient API failures.

Wrap each task-LM and reflection-LM call site in `retryable(...)` so one
flaky timeout does not kill a multi-hour run. Persistent errors raise after
`max_attempts` with a clear chained traceback rather than a silent hang.

Transient error classification
------------------------------
We treat the following as transient:
  - ``requests.exceptions.Timeout`` / ``ConnectionError``
  - ``httpx.TimeoutException`` / ``ConnectError`` / ``ReadError``
  - ``openai.APITimeoutError`` / ``APIConnectionError`` / ``RateLimitError``
  - ``openai.InternalServerError`` (5xx)
  - ``litellm.Timeout`` / ``APIConnectionError`` / ``RateLimitError``
  - ``litellm.InternalServerError`` (5xx)

We do NOT retry:
  - ``BadRequestError`` / ``AuthenticationError`` / ``PermissionDeniedError``
    (4xx that are not 408 / 429): these are configuration bugs, retry is
    pointless and burns budget.
  - ``litellm.BudgetExceededError``: respect the user's cost ceiling.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _build_transient_types() -> tuple[type, ...]:
    """Build the tuple of exception types to retry on. Resolved lazily so
    imports do not require every provider SDK to be installed."""
    types: list[type] = []

    # stdlib network errors
    types.append(TimeoutError)

    try:
        import requests
        types.extend([
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ])
    except ImportError:
        pass

    try:
        import httpx
        types.extend([
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
        ])
    except ImportError:
        pass

    try:
        import openai
        types.extend([
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ])
    except ImportError:
        pass

    try:
        import litellm
        litellm_types = []
        for name in ("Timeout", "APIConnectionError", "RateLimitError",
                     "ServiceUnavailableError", "InternalServerError"):
            cls = getattr(litellm, name, None)
            if isinstance(cls, type):
                litellm_types.append(cls)
        types.extend(litellm_types)
    except ImportError:
        pass

    return tuple(types)


_TRANSIENT_TYPES_CACHE: tuple[type, ...] | None = None


def _transient_types() -> tuple[type, ...]:
    global _TRANSIENT_TYPES_CACHE
    if _TRANSIENT_TYPES_CACHE is None:
        _TRANSIENT_TYPES_CACHE = _build_transient_types()
    return _TRANSIENT_TYPES_CACHE


def _is_transient(exc: BaseException, extra_transient: tuple[type, ...] = ()) -> bool:
    transient = _transient_types() + extra_transient
    return isinstance(exc, transient)


def retryable(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff: float = 2.0,
    jitter: float = 0.25,
    extra_transient: tuple[type, ...] = (),
    label: str | None = None,
) -> Callable[[F], F]:
    """Decorator: retry the wrapped callable on transient errors.

    Parameters
    ----------
    max_attempts:
        Including the first attempt. Default 5 (the first try plus 4 retries).
    base_delay:
        Seconds before the first retry. Subsequent delays are
        `base_delay * backoff ** (attempt-1)`, capped at `max_delay`.
    backoff:
        Multiplier for exponential backoff.
    jitter:
        Multiplicative jitter range; each sleep is multiplied by a uniform
        sample from `[1 - jitter, 1 + jitter]`. Avoids thundering-herd retries
        when many concurrent calls hit the same rate limit.
    extra_transient:
        Additional exception types to treat as transient. Useful if the
        provider raises a custom timeout subclass we haven't enumerated.
    label:
        Optional name to use in log messages. Defaults to the wrapped
        function's name.
    """
    import random as _random  # local: don't share state with band sampler's rng

    def decorator(fn: F) -> F:
        name = label or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while True:
                attempt += 1
                try:
                    return fn(*args, **kwargs)
                except BaseException as exc:
                    if not _is_transient(exc, extra_transient):
                        raise
                    if attempt >= max_attempts:
                        # Persistent transient -- raise with context so the
                        # operator sees something actionable.
                        raise RuntimeError(
                            f"{name}: gave up after {attempt} attempts on "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    delay = min(base_delay * (backoff ** (attempt - 1)), max_delay)
                    delay *= (1.0 - jitter) + 2 * jitter * _random.random()
                    logger.warning(
                        "%s: attempt %d/%d failed with %s: %s. Sleeping %.2fs.",
                        name, attempt, max_attempts,
                        type(exc).__name__, exc, delay,
                    )
                    time.sleep(delay)

        return wrapper  # type: ignore[return-value]

    return decorator
