# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Privacy Game Environment.

This module creates an HTTP server that exposes the PrivacyGameEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import DisclosureAction, DisclosureObservation
    from .privacy_game_environment import PrivacyGameEnvironment
except (ModuleNotFoundError, ImportError):
    from models import DisclosureAction, DisclosureObservation  # type: ignore[no-redef]
    from server.privacy_game_environment import PrivacyGameEnvironment  # type: ignore[no-redef]


# Create the app with web interface and README integration.
# max_concurrent_envs > 1 lets each TRL rollout group share one Space — important
# for parallel rollouts during GRPO training.
app = create_app(
    PrivacyGameEnvironment,
    DisclosureAction,
    DisclosureObservation,
    env_name="privacy_game",
    max_concurrent_envs=8,
)


# ──────────────────────────────────────────────────────────────────────────────
# Pixel UI — a richer human-play UI than the default Gradio Playground.
# Mounted at /play on the same FastAPI app so judges/visitors of the HF Space
# get a polished demo (live profile + reward gauge + segment-fill animation +
# adversary recovery readout) without leaving the Space URL.
#
# The default Gradio /web is preserved for raw-API testing; /play is the
# storytelling surface for human users.
# ──────────────────────────────────────────────────────────────────────────────

import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from .tasks import ALL_TASKS_BY_ID
except (ModuleNotFoundError, ImportError):
    from server.tasks import ALL_TASKS_BY_ID  # type: ignore[no-redef]

# voice/static/index.html lives next to the server package
_VOICE_STATIC = Path(__file__).resolve().parent.parent / "voice" / "static"
_PIXEL_INDEX = _VOICE_STATIC / "index.html"

# Per-session env cache for the pixel UI (separate from OpenEnv's own session
# pool so we don't fight over the OpenEnv API contract).
_PIXEL_SESSIONS: dict[str, PrivacyGameEnvironment] = {}
_VALID_REWARD_MODES = {"additive", "pareto_it"}


class _PixelStartReq(BaseModel):
    task_id: Optional[str] = None
    reward_mode: str = "pareto_it"


class _PixelStepReq(BaseModel):
    message: str


def _serialize_obs(obs) -> dict:
    return {
        "task_id": obs.task_id,
        "phase": obs.phase,
        "task_description": obs.task_description,
        "profile": obs.profile,
        "required_fields": obs.required_fields,
        "protected_fields": obs.protected_fields,
        "relying_party_message": obs.relying_party_message,
        "turn_number": obs.turn_number,
        "max_turns": obs.max_turns,
        "history": obs.history,
        "terminated": obs.terminated,
        "terminated_reason": obs.terminated_reason,
        "reward": obs.reward,
        "metadata": obs.metadata or {},
    }


if _PIXEL_INDEX.exists():
    # Static assets (in case the HTML ever references local files; today it's
    # all inline + Google Fonts so this is mostly a future-proof hook).
    app.mount("/static", StaticFiles(directory=str(_VOICE_STATIC)), name="static")

    @app.get("/play", response_class=HTMLResponse, include_in_schema=False)
    def pixel_ui():
        """Serve the pixel-themed human-play UI (text mode only on the Space —
        voice mode requires local ffmpeg + Whisper + a TTS engine)."""
        return HTMLResponse(_PIXEL_INDEX.read_text())

    @app.get("/api/tasks", include_in_schema=False)
    def pixel_list_tasks():
        return {
            "tasks": [
                {"task_id": t.task_id, "phase": t.phase, "description": t.description}
                for t in ALL_TASKS_BY_ID.values()
            ],
            "reward_modes": sorted(_VALID_REWARD_MODES),
        }

    @app.post("/api/start", include_in_schema=False)
    def pixel_start(req: _PixelStartReq):
        if req.reward_mode not in _VALID_REWARD_MODES:
            raise HTTPException(400, f"reward_mode must be one of {sorted(_VALID_REWARD_MODES)}")
        if req.task_id is not None and req.task_id not in ALL_TASKS_BY_ID:
            raise HTTPException(400, f"unknown task_id {req.task_id!r}")
        env = PrivacyGameEnvironment(reward_mode=req.reward_mode, force_task_id=req.task_id)
        obs = env.reset()
        sid = str(uuid.uuid4())
        _PIXEL_SESSIONS[sid] = env
        return {"session_id": sid, "observation": _serialize_obs(obs)}

    @app.post("/api/step/{sid}", include_in_schema=False)
    def pixel_step(sid: str, req: _PixelStepReq):
        env = _PIXEL_SESSIONS.get(sid)
        if env is None:
            raise HTTPException(404, "session not found")
        obs = env.step(DisclosureAction(message=req.message))
        return {"observation": _serialize_obs(obs)}


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m privacy_game.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn privacy_game.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if len(sys.argv) > 1:
        main(port=args.port)
    else:
        main()
