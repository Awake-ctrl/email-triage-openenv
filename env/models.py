"""
models.py
---------
Pydantic models for the Email Triage OpenEnv environment.

Observation  → what the agent sees each step
Action       → what the agent can do
StepResult   → full return value of env.step()
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
#  Email data model
# ─────────────────────────────────────────────────────────────

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str                          # ISO-8601
    has_attachment: bool = False
    thread_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────
#  Observation
# ─────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """Everything the agent can see at a given step."""
    inbox: List[Email]                      # remaining unprocessed emails
    current_email: Optional[Email]          # email currently being examined
    processed_count: int = 0               # how many emails handled so far
    total_emails: int = 0
    step_number: int = 0
    last_action_result: str = "none"        # feedback string from last action
    last_action_error: Optional[str] = None
    task_description: str = ""
    done: bool = False


# ─────────────────────────────────────────────────────────────
#  Action
# ─────────────────────────────────────────────────────────────

Priority   = Literal["urgent", "high", "normal", "low"]
Category   = Literal["bug_report", "feature_request", "billing",
                      "general_inquiry", "spam", "internal"]
ActionType = Literal[
    "classify",     # classify(email_id, category, priority)
    "reply",        # reply(email_id, body)
    "archive",      # archive(email_id)
    "escalate",     # escalate(email_id, reason)
    "skip",         # skip(email_id) — move to next without acting
    "done",         # signal task completion
]


class Action(BaseModel):
    """A single agent action."""
    action_type: ActionType
    email_id: Optional[str]    = None
    category:  Optional[Category]  = None
    priority:  Optional[Priority]  = None
    body:      Optional[str]       = None   # for reply
    reason:    Optional[str]       = None   # for escalate

    def to_str(self) -> str:
        """Compact string representation for [STEP] output."""
        if self.action_type == "classify":
            return (f"classify('{self.email_id}', "
                    f"category='{self.category}', priority='{self.priority}')")
        if self.action_type == "reply":
            snippet = (self.body or "")[:40].replace("\n", " ")
            return f"reply('{self.email_id}', '{snippet}...')"
        if self.action_type == "escalate":
            return f"escalate('{self.email_id}', reason='{self.reason}')"
        if self.action_type in ("archive", "skip"):
            return f"{self.action_type}('{self.email_id}')"
        return self.action_type


# ─────────────────────────────────────────────────────────────
#  Step result
# ─────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)
