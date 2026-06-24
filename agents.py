import asyncio
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma4:e2b", temperature=0)

DECOMPOSE_SYSTEM = "Break tasks into exactly 3 subtasks. One subtask per line. No numbers or bullets."
DECOMPOSE_HUMAN = "Task: {task}"

WRITER_SYSTEM = "Synthesize the research notes into one clear paragraph."
WRITER_HUMAN = "Original task: {task}\nNotes:\n{notes}"


def parse_subtasks(raw: str) -> list[str]:
    """Parse LLM output into a list of subtasks using line-based parsing."""
    try:
        lines = []
        for line in raw.strip().splitlines():
            cleaned = re.sub(r"^[\d\.\-\*\|]+\s*", "", line.strip())
            if cleaned:
                lines.append(cleaned)
        if len(lines) >= 3:
            return lines[:3]
        if lines:
            return lines
        raise ValueError("No subtasks found in LLM output")
    except Exception as exc:
        raise ValueError(f"Failed to parse subtasks: {exc}") from exc


async def decompose(task: str) -> list[str]:
    """Analyzer/Decomposer: break a complex task into 3 subtasks."""
    messages = [
        SystemMessage(content=DECOMPOSE_SYSTEM),
        HumanMessage(content=DECOMPOSE_HUMAN.format(task=task)),
    ]
    response = await llm.ainvoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    return parse_subtasks(content)


async def retrieve_data(subtask: str) -> str:
    """Mock retriever that simulates async data fetching for a subtask."""
    await asyncio.sleep(0.5)
    return f"Research finding for '{subtask}': key facts and context gathered."


async def write_summary(task: str, accumulated_data: list[str]) -> str:
    """Writer: synthesize collected data into a cohesive final paragraph."""
    notes = "\n".join(f"- {item}" for item in accumulated_data)
    messages = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=WRITER_HUMAN.format(task=task, notes=notes)),
    ]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)
