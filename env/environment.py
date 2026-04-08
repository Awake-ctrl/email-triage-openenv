"""
environment.py
--------------
EmailTriageEnv: implements the full OpenEnv interface.

  reset()       → Observation
  step(action)  → (Observation, reward, done, info)
  state()       → dict
  close()       → None
"""

from __future__ import annotations
import copy
from typing import Dict, Any, List, Optional, Tuple

from env.models import Action, Observation, Email, StepResult
from env.tasks import TASKS, TaskDefinition


class EmailTriageEnv:
    """
    OpenEnv-compatible email triage environment.

    The agent sees an inbox of emails and must classify, reply to,
    escalate, archive, or skip each one. Rewards are shaped to encourage
    correct classification and appropriate responses.
    """

    VALID_CATEGORIES = {"bug_report", "feature_request", "billing",
                        "general_inquiry", "spam", "internal"}
    VALID_PRIORITIES = {"urgent", "high", "normal", "low"}
    VALID_ACTIONS    = {"classify", "reply", "archive", "escalate", "skip", "done"}

    # Shaped reward constants
    REWARD_CLASSIFY_CORRECT    =  0.15
    REWARD_CLASSIFY_WRONG      = -0.10
    REWARD_REPLY_RELEVANT      =  0.20
    REWARD_REPLY_IRRELEVANT    = -0.05   # reply to spam
    REWARD_ESCALATE_CORRECT    =  0.15
    REWARD_ESCALATE_WRONG      = -0.10
    REWARD_ARCHIVE_SPAM        =  0.10
    REWARD_ARCHIVE_NON_SPAM    = -0.10
    REWARD_COMPLETION_BONUS    =  0.20
    REWARD_STEP_PENALTY        = -0.01   # small cost per step → efficiency

    def __init__(self, task_name: str = "email-triage-easy"):
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {list(TASKS.keys())}"
            )
        self.task_name: str             = task_name
        self._task: TaskDefinition      = TASKS[task_name]
        self._inbox: List[Email]        = []
        self._processed: List[Email]    = []
        self._actions_log: List[Dict]   = []
        self._step_count: int           = 0
        self._done: bool                = False
        self._current_email: Optional[Email] = None
        self._last_result: str          = "none"
        self._last_error: Optional[str] = None
        self._rewards: List[float]      = []
        self._obs: Optional[Observation] = None

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment to initial state and return first observation."""
        self._inbox         = copy.deepcopy(self._task.emails)
        self._processed     = []
        self._actions_log   = []
        self._step_count    = 0
        self._done          = False
        self._current_email = self._inbox[0] if self._inbox else None
        self._last_result   = "Environment reset. Ready to process emails."
        self._last_error    = None
        self._rewards       = []

        self._obs = self._build_observation()
        return self._obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Execute one action.

        Returns:
            observation  : Observation
            reward       : float
            done         : bool
            info         : dict
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() first.")

        self._step_count += 1
        self._last_error = None
        reward = self.REWARD_STEP_PENALTY   # efficiency cost per step

        try:
            reward += self._dispatch(action)
        except ValueError as exc:
            self._last_error = str(exc)
            reward = -0.05   # invalid action penalty

        # Check step limit
        if self._step_count >= self._task.max_steps:
            self._done = True
            self._last_result = f"Step limit ({self._task.max_steps}) reached."

        self._rewards.append(round(reward, 2))
        self._obs = self._build_observation()

        info = {
            "actions_log": self._actions_log,
            "step":        self._step_count,
            "final_score": self._task.grader(self._actions_log) if self._done else None,
        }

        return self._obs, round(reward, 2), self._done, info

    def state(self) -> dict:
        """Return current environment state as a plain dict."""
        return {
            "task":            self.task_name,
            "step":            self._step_count,
            "done":            self._done,
            "processed_count": len(self._processed),
            "total_emails":    len(self._task.emails),
            "inbox_remaining": len(self._inbox),
            "rewards":         self._rewards,
            "actions_log":     self._actions_log,
        }

    def close(self) -> None:
        """Clean up. Compute final score."""
        self._done = True

    @property
    def final_score(self) -> float:
        return self._task.grader(self._actions_log)

    # ── Action dispatcher ────────────────────────────────────────────────────

    def _dispatch(self, action: Action) -> float:
        """Route action to handler, return shaped reward delta."""
        atype = action.action_type

        if atype not in self.VALID_ACTIONS:
            raise ValueError(f"Unknown action type: {atype}")

        if atype == "done":
            return self._handle_done()

        # All other actions require an email_id
        if not action.email_id:
            raise ValueError("email_id is required for this action.")

        email = self._find_email(action.email_id)
        if email is None:
            raise ValueError(f"Email '{action.email_id}' not found in inbox.")

        if atype == "classify":
            return self._handle_classify(action, email)
        if atype == "reply":
            return self._handle_reply(action, email)
        if atype == "escalate":
            return self._handle_escalate(action, email)
        if atype == "archive":
            return self._handle_archive(action, email)
        if atype == "skip":
            return self._handle_skip(action, email)

        raise ValueError(f"Unhandled action: {atype}")

    # ── Action handlers ──────────────────────────────────────────────────────

    def _handle_classify(self, action: Action, email: Email) -> float:
        if action.category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{action.category}'. "
                f"Valid: {self.VALID_CATEGORIES}"
            )
        if action.priority not in self.VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{action.priority}'. "
                f"Valid: {self.VALID_PRIORITIES}"
            )

        self._log_action(action)
        self._last_result = (
            f"Classified '{email.id}' as {action.category}/{action.priority}."
        )

        # Shaped reward: we do a heuristic check
        # (full grading happens at episode end via grader)
        reward = self._heuristic_classify_reward(email, action)
        return reward

    def _handle_reply(self, action: Action, email: Email) -> float:
        body = (action.body or "").strip()
        if len(body) < 10:
            raise ValueError("Reply body is too short (minimum 10 characters).")

        self._log_action(action)
        self._last_result = f"Reply sent to '{email.id}'."

        # Penalize replying to clearly spam emails (sender signals)
        if self._looks_like_spam(email):
            return self.REWARD_REPLY_IRRELEVANT
        return self.REWARD_REPLY_RELEVANT

    def _handle_escalate(self, action: Action, email: Email) -> float:
        reason = (action.reason or "").strip()
        if len(reason) < 5:
            raise ValueError("Escalation reason is too short (minimum 5 characters).")

        self._log_action(action)
        self._last_result = f"Escalated '{email.id}': {reason}"

        if self._looks_like_spam(email):
            return self.REWARD_ESCALATE_WRONG   # shouldn't escalate spam
        return self.REWARD_ESCALATE_CORRECT

    def _handle_archive(self, action: Action, email: Email) -> float:
        self._log_action(action)
        self._inbox  = [e for e in self._inbox if e.id != email.id]
        self._processed.append(email)
        self._current_email = self._inbox[0] if self._inbox else None
        self._last_result = f"Archived '{email.id}'."

        if self._looks_like_spam(email):
            return self.REWARD_ARCHIVE_SPAM
        return self.REWARD_ARCHIVE_NON_SPAM

    def _handle_skip(self, action: Action, email: Email) -> float:
        self._log_action(action)
        # Move email to back of inbox
        self._inbox = [e for e in self._inbox if e.id != email.id] + [email]
        self._current_email = self._inbox[0] if self._inbox else None
        self._last_result = f"Skipped '{email.id}', moved to end."
        return 0.0   # no reward for skipping

    def _handle_done(self) -> float:
        self._done = True
        score = self._task.grader(self._actions_log)
        self._last_result = f"Task complete. Final score: {score:.4f}"
        return self.REWARD_COMPLETION_BONUS * score   # bonus scales with quality

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _find_email(self, email_id: str) -> Optional[Email]:
        for e in self._inbox:
            if e.id == email_id:
                return e
        return None

    def _log_action(self, action: Action):
        self._actions_log.append(action.model_dump())

    def _looks_like_spam(self, email: Email) -> bool:
        spam_signals = [
            "free", "click now", "won", "prize", "unsubscribe",
            "!!!",  "newsletter.biz", "promotions.example",
            "suspicious", "darkweb", "phishing",
        ]
        text = (email.subject + " " + email.body + " " + email.sender).lower()
        return any(sig in text for sig in spam_signals)

    def _heuristic_classify_reward(self, email: Email, action: Action) -> float:
        """
        Lightweight heuristic for shaped reward during episode.
        Full scoring happens via grader at episode end.
        """
        text = (email.subject + " " + email.body).lower()

        # Obvious spam correctly labelled as spam → always reward
        spam_indicators = ["unsubscribe", "click now", "free iphone", "!!!"]
        is_obvious_spam = any(ind in text for ind in spam_indicators)
        if is_obvious_spam:
            return (self.REWARD_CLASSIFY_CORRECT if action.category == "spam"
                    else self.REWARD_CLASSIFY_WRONG)

        urgent_indicators = ["urgent", "critical", "cvss 9", "down", "breach",
                              "production", "immediately", "chargeback"]
        is_urgent = any(ind in text for ind in urgent_indicators)
        if is_urgent and action.priority == "low":
            return self.REWARD_CLASSIFY_WRONG

        return self.REWARD_CLASSIFY_CORRECT

    def _build_observation(self) -> Observation:
        return Observation(
            inbox=copy.deepcopy(self._inbox),
            current_email=copy.deepcopy(self._current_email),
            processed_count=len(self._processed),
            total_emails=len(self._task.emails),
            step_number=self._step_count,
            last_action_result=self._last_result,
            last_action_error=self._last_error,
            task_description=self._task.description,
            done=self._done,
        )
