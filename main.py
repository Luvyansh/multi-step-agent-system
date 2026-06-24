import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph import compiled_graph, make_initial_state

app = FastAPI(title="Multi-Step Technical Briefing Engine")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ExecuteRequest(BaseModel):
    task: str


NODE_LABELS = {
    "decompose_task": "Step 1: Task decomposed",
    "execute_batch_retrieval": "Step 2: Processing batch retrieval",
    "synthesize_output": "Step 3: Synthesizing final briefing",
    "handle_error": "Error: Pipeline failure handled",
}


def _format_event(node_name: str, update: dict) -> str:
    label = NODE_LABELS.get(node_name, node_name)
    steps = update.get("completed_steps", [])
    detail = steps[-1] if steps else "update received"
    payload = {"step": label, "detail": detail, "node": node_name}

    if update.get("subtasks"):
        payload["subtasks"] = update["subtasks"]
    if update.get("accumulated_data"):
        payload["output"] = update["accumulated_data"][-1]

    return f"data: {json.dumps(payload)}\n\n"


@app.get("/")
async def root():
    """Serve the interactive frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/execute")
async def execute(request: ExecuteRequest):
    """Stream partial pipeline outputs via Server-Sent Events."""

    async def event_stream():
        initial = make_initial_state(request.task)
        yield f"data: {json.dumps({'step': 'Step 0: Pipeline started', 'detail': request.task})}\n\n"

        async for chunk in compiled_graph.astream(initial):
            for node_name, update in chunk.items():
                yield _format_event(node_name, update)

        yield f"data: {json.dumps({'step': 'Complete', 'detail': 'Pipeline finished'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
