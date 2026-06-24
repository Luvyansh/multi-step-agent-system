import asyncio
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from utils import retry_with_backoff

llm = ChatOllama(model="gemma4:e2b", temperature=0)
writer_llm = ChatOllama(model="gemma4:e2b", temperature=0, num_predict=1024)


@dataclass
class DecomposeResult:
    subtasks: list[str]
    thinking: str


@dataclass
class SummaryResult:
    summary: str
    thinking: str


def extract_thinking(response) -> tuple[str, str]:
    """Returns (final_answer, thinking_text). Reads reasoning_content from
    additional_kwargs directly; returns empty thinking_text if absent. No
    regex parsing of content is needed — confirmed via live testing that
    no <|channel>thought markers ever appear embedded in content for this
    model/library combination."""
    final_answer = response.content if hasattr(response, "content") else str(response)
    thinking_text = ""
    if hasattr(response, "additional_kwargs"):
        thinking_text = response.additional_kwargs.get("reasoning_content", "")
    return final_answer, thinking_text

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

WRITER_SYSTEM = (
    "You are a technical writer producing a detailed briefing. Do NOT write a single short paragraph. "
    "Structure your response with: "
    "1) A brief 1-2 sentence introduction. "
    "2) A clearly labeled section for each research question provided, using a '## ' markdown header per section, with 2-4 sentences of substantive explanation per section. "
    "3) A short comparison or summary table in markdown if the topic involves comparing two or more things. "
    "4) A 1-2 sentence closing synthesis. "
    "Use markdown formatting (headers, bullet points, bold) throughout. "
    "If a mathematical formula is relevant, express it in LaTeX using \\(...\\) for inline or \\[...\\] for block math."
)
WRITER_HUMAN = (
    "Original task: {task}\n\n"
    "Research questions:\n{questions}\n\n"
    "Retrieved notes:\n{notes}"
)


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
async def decompose(task: str) -> DecomposeResult:
    """Analyzer/Decomposer: break a complex task into 3 subtasks."""
    messages = [
        SystemMessage(content=DECOMPOSE_SYSTEM),
        HumanMessage(content=DECOMPOSE_HUMAN.format(task=task)),
    ]
    response = await llm.ainvoke(messages, reasoning=True)
    content, thinking = extract_thinking(response)
    return DecomposeResult(subtasks=parse_subtasks(content), thinking=thinking)


@retry_with_backoff(max_retries=2)
async def retrieve_data(subtask: str) -> str:
    """Mock retriever that simulates async data fetching for a subtask.

    MOCK IMPLEMENTATION: this function only simulates retrieval latency.
    In production, this would call a real data source such as a web search API,
    internal document index, or vector database.
    """
    await asyncio.sleep(0.5)
    return f"[MOCK FINDING] Simulated retrieval result for '{subtask}'."


async def write_summary(task: str, subtasks: list[str], accumulated_data: list[str]) -> SummaryResult:
    """Writer: synthesize collected data into a detailed structured briefing."""
    questions = "\n".join(f"- {q}" for q in subtasks)
    notes = "\n".join(f"- {item}" for item in accumulated_data)
    messages = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=WRITER_HUMAN.format(task=task, questions=questions, notes=notes)),
    ]
    response = await writer_llm.ainvoke(messages, reasoning=True)
    summary, thinking = extract_thinking(response)
    return SummaryResult(summary=summary, thinking=thinking)
