import asyncio
import time
from unittest.mock import AsyncMock, patch

from batching import process_items_in_batches
from graph import build_graph, route_after_decompose
from utils import retry_with_backoff


def test_process_items_in_batches_runs_concurrently():
    """Verify batching processes items concurrently within each batch."""

    async def _run() -> None:
        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def slow_process(item: str) -> str:
            nonlocal active, peak
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.05)
            async with lock:
                active -= 1
            return f"done:{item}"

        items = ["a", "b", "c", "d"]
        start = time.perf_counter()
        results = await process_items_in_batches(items, batch_size=2, process_func=slow_process)
        elapsed = time.perf_counter() - start

        assert results == ["done:a", "done:b", "done:c", "done:d"]
        assert peak == 2
        assert elapsed < 0.25

    asyncio.run(_run())


def test_retry_decorator_fails_twice_then_succeeds():
    """Verify retry decorator retries up to max_retries before succeeding."""

    async def _run() -> None:
        attempts = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("temporary failure")
            return "success"

        result = await flaky()
        assert result == "success"
        assert attempts == 3

    asyncio.run(_run())


def test_route_after_decompose_to_error_on_fatal():
    """Ensure catastrophic decomposition failure routes to handle_error."""
    state = {
        "original_task": "test",
        "subtasks": [],
        "completed_steps": [],
        "accumulated_data": [],
        "errors": ["FATAL: Decomposition failed: parse error"],
    }
    assert route_after_decompose(state) == "handle_error"


def test_graph_routes_to_error_node_on_catastrophic_failure():
    """Simulate a catastrophic failure and verify handle_error is reached."""

    async def _run() -> None:
        graph = build_graph()

        with patch("graph.decompose", new_callable=AsyncMock) as mock_decompose:
            mock_decompose.side_effect = ValueError("unparseable junk")

            final_state = await graph.ainvoke(
                {
                    "original_task": "Explain quantum computing",
                    "subtasks": [],
                    "completed_steps": [],
                    "accumulated_data": [],
                    "errors": [],
                }
            )

        assert any("FATAL:" in err for err in final_state["errors"])
        assert any("Error handled" in step for step in final_state["completed_steps"])
        assert final_state["accumulated_data"]
        assert "could not be completed" in final_state["accumulated_data"][-1].lower()

    asyncio.run(_run())
