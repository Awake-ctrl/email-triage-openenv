"""
server.py
---------
FastAPI server exposing the EmailTriageEnv over HTTP.
This is the OpenEnv environment server — the inference script calls it.

Endpoints:
  POST /reset          → Observation
  POST /step           → StepResult
  GET  /state          → dict
  POST /close          → {"status": "closed"}
  GET  /tasks          → list of available task names
  GET  /health         → {"status": "ok"}
"""

from __future__ import annotations
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import EmailTriageEnv
from env.models import Action, Observation, StepResult

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

# Global environment instance (one per server process)
_env: Optional[EmailTriageEnv] = None


# ── Request bodies ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "email-triage-easy"


class StepRequest(BaseModel):
    action: Action

@app.get("/")
def root():
    return {"message": "Email Triage OpenEnv Environment Running"}
# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env_initialized": _env is not None}


@app.get("/tasks")
def list_tasks():
    from env.tasks import TASKS
    return {
        name: {
            "difficulty": t.difficulty,
            "max_steps":  t.max_steps,
            "num_emails": len(t.emails),
            "description": t.description[:120] + "...",
        }
        for name, t in TASKS.items()
    }


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    global _env
    try:
        _env = EmailTriageEnv(task_name=req.task_name)
        obs  = _env.reset()
        return obs
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(req: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400,
                            detail="Environment not initialized. Call /reset first.")
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
        raise HTTPException(status_code=400,
                            detail="Environment not initialized. Call /reset first.")
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
