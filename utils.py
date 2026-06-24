import asyncio
import functools
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_with_backoff(max_retries: int = 2, base_delay: float = 0.5):
    """Retry decorator with exponential backoff for async and sync callables."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                last_exc: Exception | None = None
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        last_exc = exc
                        if attempt < max_retries:
                            await asyncio.sleep(base_delay * (2**attempt))
                raise last_exc  # type: ignore[misc]

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        import time

                        time.sleep(base_delay * (2**attempt))
            raise last_exc  # type: ignore[misc]

        return sync_wrapper  # type: ignore[return-value]

    return decorator
