from langgraph.graph import END, StateGraph
import time

from agents import decompose, retrieve_data, write_summary
from batching import process_items_in_batches
from state import GraphState

BATCH_SIZE = 2
_FATAL_PREFIX = "FATAL:"


def _initial_state(task: str) -> GraphState:
    return {
        "original_task": task,
        "subtasks": [],
        "completed_steps": [],
        "accumulated_data": [],
        "errors": [],
        "thinking_log": {},
        "node_elapsed": {},
    }


async def decompose_task(state: GraphState) -> dict:
    """Break the original task into subtasks."""
    start = time.perf_counter()
    try:
        result = await decompose(state["original_task"])
        elapsed = time.perf_counter() - start
        return {
            "subtasks": result.subtasks,
            "completed_steps": ["Task decomposed into subtasks"],
            "thinking_log": {"decompose_task": result.thinking},
            "node_elapsed": {"decompose_task": elapsed},
        }
    except Exception as exc:
        return {
            "errors": [f"{_FATAL_PREFIX} Decomposition failed: {exc}"],
            "completed_steps": ["Decomposition failed"],
        }


async def execute_batch_retrieval(state: GraphState) -> dict:
    """Retrieve data for each subtask using manual async batching."""
    subtasks = state.get("subtasks", [])
    if not subtasks:
        return {
            "errors": [f"{_FATAL_PREFIX} No subtasks available for retrieval"],
            "completed_steps": ["Batch retrieval skipped"],
        }

    try:
        num_subtasks = len(subtasks)
        num_batches = (num_subtasks + BATCH_SIZE - 1) // BATCH_SIZE
        results = await process_items_in_batches(subtasks, BATCH_SIZE, retrieve_data)
        return {
            "accumulated_data": results,
            "completed_steps": [
                (
                    f"Running {num_subtasks} subtask(s) in {num_batches} batch(es) "
                    f"of up to {BATCH_SIZE} concurrent task(s)"
                ),
                f"Retrieved data for {len(results)} subtasks",
            ],
        }
    except Exception as exc:
        return {
            "errors": [f"{_FATAL_PREFIX} Batch retrieval failed: {exc}"],
            "completed_steps": ["Batch retrieval failed"],
        }


async def synthesize_output(state: GraphState) -> dict:
    """Synthesize retrieved data into a final briefing paragraph."""
    start = time.perf_counter()
    try:
        result = await write_summary(state["original_task"], state["accumulated_data"])
        elapsed = time.perf_counter() - start
        return {
            "accumulated_data": [result.summary],
            "completed_steps": ["Final briefing synthesized"],
            "thinking_log": {"synthesize_output": result.thinking},
            "node_elapsed": {"synthesize_output": elapsed},
        }
    except Exception as exc:
        return {
            "errors": [f"{_FATAL_PREFIX} Synthesis failed: {exc}"],
            "completed_steps": ["Synthesis failed"],
        }


async def handle_error(state: GraphState) -> dict:
    """Record graceful failure when a pipeline step exhausts retries."""
    existing = state.get("errors", [])
    if existing:
        message = existing[-1]
    else:
        message = "Unknown pipeline error"
    return {
        "completed_steps": [f"Error handled: {message}"],
        "accumulated_data": [f"Briefing could not be completed. {message}"],
    }


def _has_fatal_error(state: GraphState) -> bool:
    return any(err.startswith(_FATAL_PREFIX) for err in state.get("errors", []))


def route_after_decompose(state: GraphState) -> str:
    if _has_fatal_error(state):
        return "handle_error"
    return "execute_batch_retrieval"


def route_after_retrieval(state: GraphState) -> str:
    if _has_fatal_error(state):
        return "handle_error"
    return "synthesize_output"


def route_after_synthesis(state: GraphState) -> str:
    if _has_fatal_error(state):
        return "handle_error"
    return END


def build_graph():
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(GraphState)

    graph.add_node("decompose_task", decompose_task)
    graph.add_node("execute_batch_retrieval", execute_batch_retrieval)
    graph.add_node("synthesize_output", synthesize_output)
    graph.add_node("handle_error", handle_error)

    graph.set_entry_point("decompose_task")

    graph.add_conditional_edges("decompose_task", route_after_decompose)
    graph.add_conditional_edges("execute_batch_retrieval", route_after_retrieval)
    graph.add_conditional_edges("synthesize_output", route_after_synthesis)

    graph.add_edge("handle_error", END)

    return graph.compile()


compiled_graph = build_graph()


def make_initial_state(task: str) -> GraphState:
    return _initial_state(task)
