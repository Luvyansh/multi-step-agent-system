from typing import Annotated, TypedDict
import operator


def _merge_thinking_logs(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    return {**left, **right}


class GraphState(TypedDict):
    """LangGraph state for the multi-step briefing engine."""

    original_task: str
    subtasks: list[str]
    completed_steps: Annotated[list[str], operator.add]
    accumulated_data: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    thinking_log: Annotated[dict[str, str], _merge_thinking_logs]
