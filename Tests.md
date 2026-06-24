# Test Suite

## Overview

This document describes the pytest suite for the Multi-Step Technical Briefing Engine.

## How to Run

```bash
pip install -r requirements.txt
pytest test_system.py -v
```

## Test Cases

### 1. `test_process_items_in_batches_runs_concurrently`

Validates the manual asyncio batching helper in `batching.py`.

- Processes four items with `batch_size=2`
- Tracks peak concurrent executions
- Asserts all results are returned in order
- Asserts peak concurrency equals the batch limit
- Asserts total runtime is faster than fully sequential execution

### 2. `test_retry_decorator_fails_twice_then_succeeds`

Validates the exponential backoff retry decorator in `utils.py`.

- Mocks an async function that fails on the first two attempts
- Asserts the third attempt succeeds
- Asserts exactly three attempts were made

### 3. `test_route_after_decompose_to_error_on_fatal`

Validates conditional routing logic in `graph.py`.

- Supplies graph state containing a `FATAL:` error
- Asserts routing selects `handle_error`

### 4. `test_graph_routes_to_error_node_on_catastrophic_failure`

Validates end-to-end error routing through the compiled LangGraph.

- Mocks decomposition to raise after retries would fail
- Invokes the graph asynchronously
- Asserts fatal errors are recorded
- Asserts the error handler step ran
- Asserts a graceful failure message is stored in `accumulated_data`

## Notes

- Tests avoid calling the local Ollama model directly
- Graph and routing tests use mocks for deterministic behavior
- Ensure Ollama is running with `gemma4:e2b` before exercising the live API
