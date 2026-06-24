import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def process_items_in_batches(
    items: list[T],
    batch_size: int,
    process_func: Callable[[T], Awaitable[str]],
) -> list[str]:
    """Process items in manual async batches using Semaphore and gather."""
    semaphore = asyncio.Semaphore(batch_size)
    results: list[str] = []

    async def _process_one(item: T) -> str:
        async with semaphore:
            return await process_func(item)

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = await asyncio.gather(*[_process_one(item) for item in batch])
        results.extend(batch_results)

    return results
