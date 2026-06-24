import asyncio
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from utils import retry_with_backoff

llm = ChatOllama(model="gemma4:e2b", temperature=0)

DECOMPOSE_SYSTEM = (
    "Break the task into exactly 3 specific research QUESTIONS needed to answer it. "
    "Each line must be a question ending in '?'. No numbers, no bullets, no answers."
)
DECOMPOSE_HUMAN = (
    "Task: {task}\n\n"
    "Example for task 'Explain how vaccines work':\n"
    "What is the biological mechanism by which vaccines trigger immunity?\n"
    "What are the main types of vaccines and how do they differ?\n"
    "What risks or side effects are associated with vaccines?\n\n"
    "Now write 3 questions for the task above:"
)

WRITER_SYSTEM = "Synthesize the research notes into one clear paragraph."
WRITER_HUMAN = "Original task: {task}\nNotes:\n{notes}"


def parse_subtasks(raw: str) -> list[str]:
    """Parse LLM output into a list of subtasks using line-based parsing."""
    try:
        lines = []
        for line in raw.strip().splitlines():
            cleaned = re.sub(r"^[\d\.\-\*\|]+\s*", "", line.strip())
            if cleaned and "?" in cleaned:
                lines.append(cleaned)
        if len(lines) >= 3:
            return lines[:3]
        if lines:
            return lines
        raise ValueError("No subtasks found in LLM output")
    except Exception as exc:
        raise ValueError(f"Failed to parse subtasks: {exc}") from exc


@retry_with_backoff(max_retries=2)
async def decompose(task: str) -> list[str]:
    """Analyzer/Decomposer: break a complex task into 3 subtasks."""
    messages = [
        SystemMessage(content=DECOMPOSE_SYSTEM),
        HumanMessage(content=DECOMPOSE_HUMAN.format(task=task)),
    ]
    response = await llm.ainvoke(messages)
    content = response.content if hasattr(response, "content") else str(response)
    return parse_subtasks(content)


@retry_with_backoff(max_retries=2)
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
