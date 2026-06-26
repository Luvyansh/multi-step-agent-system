"""Throwaway diagnostic: how gemma4:e2b thinking output arrives via langchain-ollama."""

import asyncio
import json
import pprint

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

MODEL = "gemma4:e2b"
PROMPT_SYSTEM = "Break the task into exactly 3 specific research QUESTIONS. One per line, each ending in '?'."
PROMPT_HUMAN = "Task: Explain the difference between SQL and NoSQL databases."


def _dump_response(label: str, response) -> None:
    print("=" * 80)
    print(label)
    print("=" * 80)
    print("\n--- type ---")
    print(type(response))
    print("\n--- repr(response) ---")
    print(repr(response))
    print("\n--- response.content (COMPLETE, no truncation) ---")
    print(response.content)
    print("\n--- response.additional_kwargs ---")
    pprint.pp(response.additional_kwargs)
    print("\n--- response.response_metadata ---")
    pprint.pp(getattr(response, "response_metadata", {}))
    print("\n--- dir(response) notable attrs ---")
    for attr in ("content", "additional_kwargs", "response_metadata", "usage_metadata", "id"):
        if hasattr(response, attr):
            val = getattr(response, attr)
            print(f"{attr}: {val!r}")
    print("\n--- json-serializable snapshot ---")
    try:
        snapshot = {
            "content": response.content,
            "additional_kwargs": response.additional_kwargs,
            "response_metadata": getattr(response, "response_metadata", None),
            "usage_metadata": getattr(response, "usage_metadata", None),
        }
        print(json.dumps(snapshot, indent=2, default=str))
    except Exception as exc:
        print(f"(snapshot failed: {exc})")
    print()


async def test_a_reasoning_kwarg() -> None:
    """Approach (a): reasoning=True on ChatOllama (maps to Ollama think API param)."""
    llm = ChatOllama(model=MODEL, temperature=0, reasoning=True)
    messages = [
        SystemMessage(content=PROMPT_SYSTEM),
        HumanMessage(content=PROMPT_HUMAN),
    ]
    response = await llm.ainvoke(messages)
    _dump_response("(a) ChatOllama(reasoning=True) -> ainvoke", response)


async def test_a_think_ainvoke_kwarg() -> None:
    """Approach (a) variant: reasoning=True passed to ainvoke (not constructor)."""
    llm = ChatOllama(model=MODEL, temperature=0)
    messages = [
        SystemMessage(content=PROMPT_SYSTEM),
        HumanMessage(content=PROMPT_HUMAN),
    ]
    response = await llm.ainvoke(messages, reasoning=True)
    _dump_response("(a2) ainvoke(reasoning=True) kwarg", response)


async def test_b_think_token_in_system() -> None:
    """Approach (b): <|think|> prefix in system prompt, no reasoning kwarg."""
    llm = ChatOllama(model=MODEL, temperature=0)
    messages = [
        SystemMessage(content=f"<|think|>{PROMPT_SYSTEM}"),
        HumanMessage(content=PROMPT_HUMAN),
    ]
    response = await llm.ainvoke(messages)
    _dump_response("(b) SystemMessage starts with <|think|>, no reasoning kwarg", response)


async def main() -> None:
    print("langchain-ollama thinking diagnostic for gemma4:e2b")
    print(f"Model: {MODEL}\n")

    for name, coro in [
        ("a", test_a_reasoning_kwarg()),
        ("a2", test_a_think_ainvoke_kwarg()),
        ("b", test_b_think_token_in_system()),
    ]:
        try:
            await coro
        except Exception as exc:
            print("=" * 80)
            print(f"ERROR in test {name}: {type(exc).__name__}: {exc}")
            print("=" * 80)
            print()


if __name__ == "__main__":
    asyncio.run(main())
