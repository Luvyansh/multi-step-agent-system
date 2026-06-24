import asyncio
import time
from unittest.mock import AsyncMock, patch

from agents import DecomposeResult, SummaryResult, parse_subtasks, retrieve_data
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
        "thinking_log": {},
        "node_elapsed": {},
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
                    "thinking_log": {},
                    "node_elapsed": {},
                }
            )

        assert any("FATAL:" in err for err in final_state["errors"])
        assert any("Error handled" in step for step in final_state["completed_steps"])
        assert final_state["accumulated_data"]
        assert "could not be completed" in final_state["accumulated_data"][-1].lower()

    asyncio.run(_run())


def test_parse_subtasks_accepts_valid_questions():
    raw = (
        "What is zero trust security?\n"
        "How does zero trust differ from perimeter-based models?\n"
        "What are the implementation challenges of zero trust?\n"
    )
    parsed = parse_subtasks(raw)
    assert len(parsed) == 3
    assert all("?" in item for item in parsed)


def test_parse_subtasks_rejects_non_question_lines():
    mixed = (
        "This is plain prose and should be discarded.\n"
        "What is edge computing?\n"
        "Another declarative sentence with no punctuation\n"
        "How is latency reduced with edge computing?\n"
    )
    parsed = parse_subtasks(mixed)
    assert parsed == [
        "What is edge computing?",
        "How is latency reduced with edge computing?",
    ]

    with_no_questions = "This output has no question marks at all."
    try:
        parse_subtasks(with_no_questions)
        assert False, "Expected ValueError when no question lines are present"
    except ValueError:
        pass


def test_parse_subtasks_raises_on_garbage_input():
    garbage = "Sorry, I cannot help with that request right now"
    try:
        parse_subtasks(garbage)
        assert False, "Expected ValueError for non-question garbage input"
    except ValueError:
        pass


def test_retrieve_data_is_labeled_as_mock():
    async def _run() -> None:
        subtask = "What are the benefits of test-driven development?"
        output = await retrieve_data(subtask)
        assert "[MOCK FINDING]" in output
        assert subtask in output

    asyncio.run(_run())


def test_graph_happy_path_with_mocked_llm_calls():
    async def _run() -> None:
        graph = build_graph()
        mocked_questions = [
            "What is Kubernetes?",
            "How does Kubernetes scheduling work?",
            "What are common Kubernetes operational risks?",
        ]
        mocked_summary = "Kubernetes orchestrates containers across clusters with scheduling, scaling, and resilience features."

        with (
            patch("graph.decompose", new_callable=AsyncMock) as mock_decompose,
            patch("graph.write_summary", new_callable=AsyncMock) as mock_write_summary,
        ):
            mock_decompose.return_value = DecomposeResult(subtasks=mocked_questions, thinking="")
            mock_write_summary.return_value = SummaryResult(summary=mocked_summary, thinking="")

            final_state = await graph.ainvoke(
                {
                    "original_task": "Explain Kubernetes for platform engineering",
                    "subtasks": [],
                    "completed_steps": [],
                    "accumulated_data": [],
                    "errors": [],
                    "thinking_log": {},
                    "node_elapsed": {},
                }
            )

        assert final_state["errors"] == []
        assert final_state["accumulated_data"][-1] == mocked_summary
        assert any("Task decomposed into subtasks" in step for step in final_state["completed_steps"])
        assert any("Running" in step and "batch(es)" in step for step in final_state["completed_steps"])
        assert any("Retrieved data for 3 subtasks" in step for step in final_state["completed_steps"])
        assert any("Final briefing synthesized" in step for step in final_state["completed_steps"])

    asyncio.run(_run())
