from typing import Annotated, TypedDict
import operator


class GraphState(TypedDict):
    """LangGraph state for the multi-step briefing engine."""

    original_task: str
    subtasks: list[str]
    completed_steps: Annotated[list[str], operator.add]
    accumulated_data: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
