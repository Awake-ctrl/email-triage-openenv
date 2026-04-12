"""
server/app.py
-------------
FastAPI environment server — canonical location required by openenv validate.
Entry point registered in pyproject.toml as:
  [project.scripts]
  serve = "server.app:main"

Endpoints:
  GET  /health   → {"status": "ok"}
  GET  /tasks    → task metadata
  POST /reset    → Observation
  POST /step     → {observation, reward, done, info}
  GET  /state    → current env state dict
  POST /close    → {"status": "closed", "final_score": float}
"""

from __future__ import annotations
import os
import sys
from typing import Optional

# Ensure project root is on sys.path so `env.*` imports work whether
# run as `python server/app.py` or via the installed `serve` script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import EmailTriageEnv
from env.models import Action, Observation

app = FastAPI(
    title="Email Triage OpenEnv",
    description="Real-world email triage environment for RL agent evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server process
_env: Optional[EmailTriageEnv] = None


# ── Request bodies ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str]=None
    task:Optional[str]=None


class StepRequest(BaseModel):
    action: Action


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env_initialized": _env is not None}


@app.get("/tasks")
def list_tasks():
    from env.tasks import TASKS
    return {
        name: {
            "difficulty":  t.difficulty,
            "max_steps":   t.max_steps,
            "num_emails":  len(t.emails),
            "description": t.description[:120] + "...",
        }
        for name, t in TASKS.items()
    }


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest]=None):
    global _env
    # handle cases where no body is sent
    task_name = "email-triage-easy"

    if req:
        task_name = req.task_name or req.task or task_name
    try:
        _env = EmailTriageEnv(task_name=task_name)
        obs  = _env.reset()
        return obs
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(req: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )
    try:
        obs, reward, done, info = _env.step(req.action)
        return {
            "observation": obs.model_dump(),
            "reward":      round(reward, 2),
            "done":        done,
            "info":        {k: v for k, v in info.items() if k != "actions_log"},
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )
    return _env.state()


@app.post("/close")
def close():
    global _env
    if _env is None:
        return {"status": "already_closed"}
    _env.close()
    score = _env.final_score
    _env  = None
    return {"status": "closed", "final_score": round(score, 4)}


# ── Entry point registered in pyproject.toml [project.scripts] ───────────────

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
