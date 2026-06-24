# Post-Mortem — Multi-Step Technical Briefing Engine

## Summary

This project implements a multi-step agentic pipeline (Decompose → Retrieve → Synthesize) orchestrated by a hand-built LangGraph state machine, running entirely on local hardware via `ChatOllama` (`gemma4:e2b`). It includes manual async batching, retry-with-backoff failure handling, and real-time SSE streaming of pipeline progress. The system was validated against both a successful run and a live induced failure (Ollama taken offline mid-request).

---

## Scaling Issue

**The Retriever agent is currently mocked.** It simulates network latency (`asyncio.sleep(0.5)`) and returns a templated, clearly-labeled placeholder string per subtask rather than querying a real data source.

This was a deliberate choice given the project's hardware constraints (4GB VRAM, limited system RAM) and time box — the goal was to prove the orchestration, batching, and failure-handling architecture works correctly, not to build a production retrieval backend. But it is the part of the system that would break first under real scaling pressure. A real implementation would need to account for:

- **Rate limits** on whatever external API or search service backs retrieval (most have per-minute caps that would directly conflict with the batch concurrency settings).
- **Latency variance** — `asyncio.sleep(0.5)` is constant; real retrieval latency is not, and the batch logic would need timeout handling per-item, not just per-batch.
- **Failure modes specific to the new dependency** — a real retriever introduces a new external point of failure (network errors, malformed responses, empty result sets) that the current retry decorator is generically built to handle, but has never been tested against.

If this system needed to scale to real retrieval today, the batch size (currently 2, chosen for memory safety on a 2B local model) would also need to be reconsidered — the constraint that justified `batch_size=2` was protecting local VRAM/RAM during concurrent LLM calls, not protecting a remote API's rate limit, and those two constraints don't necessarily call for the same number.

---

## Design Change I'd Make in Hindsight

**The original Decomposer prompt asked for "3 subtasks" with no examples and no constraint on what a subtask should look like.** In practice, the 2B model interpreted this as "write 3 sentences of the final answer" rather than "ask 3 independent research questions" — so the first working version of the pipeline technically ran end-to-end, but the Retriever and Writer steps were doing almost no real work; the "decomposition" was really just chunking a pre-formed answer and gluing it back together.

This only became visible by actually reading the model's raw output against a real prompt — the system *looked* correct from the outside (it ran, it streamed, it produced a coherent paragraph) while the core agentic behavior underneath wasn't real.

In hindsight, I would have written the prompt with a one-shot example from the start, and explicitly required each decomposed line to be a question (validated structurally, not just by length or line count). Both of those came later as a fix; if I rebuilt this from scratch, the first version of the Decomposer would have had:
- A worked example in the prompt
- A structural validation rule (e.g., "must contain `?`") in the parser, not just "must be non-empty"

This would have caught the false-decomposition failure mode in step 2 of the original build instead of step 7.

---

## Trade-off #1: Mocked Retrieval vs. Real Retrieval

**Decision:** Use a mocked, clearly-labeled Retriever instead of integrating a real data source.

**Reasoning:** The assignment's evaluation criteria emphasize architecture, orchestration, and failure handling under explicit hardware constraints — not the quality of retrieved content. Building a real retrieval integration (API keys, rate limiting, response parsing, new failure modes) would have spent disproportionate time on a part of the system that isn't what's being assessed, at the cost of testing and hardening the orchestration logic that is. The trade-off was made explicit rather than disguised: the mock is labeled `[MOCK FINDING]` in its actual output (not just in a comment), so anyone running the system sees the limitation directly rather than discovering it later.

**Cost of this choice:** The system cannot currently demonstrate that retrieval failures (timeouts, malformed API responses, empty results) are handled correctly — only that *retry logic in general* is handled correctly, validated against an LLM call failure. If retrieval were real, that would need its own dedicated failure test.

---

## Trade-off #2: Line-Based Text Parsing vs. Strict JSON Schema

**Decision:** Have the LLM return plain text (one item per line) and parse it with regex/string logic wrapped in retry-on-failure, rather than enforcing a strict JSON schema on every LLM response.

**Reasoning:** `gemma4:e2b` is a small, efficient on-device model, and small models are measurably less reliable at producing valid nested JSON under constraint than they are at producing simple delimited text. Strict JSON schema enforcement would likely have increased parse failure rate and required more aggressive retry/repair logic for no real benefit, since the data being exchanged (a short list of questions, a list of findings, a paragraph) doesn't actually need nested structure.

**Cost of this choice:** Parsing is more fragile than schema validation would be in the abstract — it depends on the model loosely following formatting instructions (one item per line, lines containing `?`) rather than a parser that mechanically rejects anything that doesn't match a schema. This was a real risk, not a theoretical one: the first version of the Decomposer prompt produced output that parsed *successfully* but was semantically wrong (sentences instead of questions), which a JSON schema wouldn't have caught either, but which a more constrained text format combined with a stricter validation rule eventually did.

---

## What Was Verified Live (Not Just Unit-Tested)

- **Happy path:** Submitted a real task through the running system; the Decomposer produced 3 distinct, non-overlapping research questions from the actual model; the Writer produced a coherent final briefing with no mock-label leakage into the output.
- **Failure path:** Took Ollama offline mid-request; observed the retry decorator exhaust its attempts against a real connection failure, the graph correctly route to the error-handling node, and the SSE stream surface a graceful failure message instead of hanging or crashing.
