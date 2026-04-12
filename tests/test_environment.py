"""
tests/test_environment.py
--------------------------
Unit tests for the Email Triage environment.
Run with: python -m pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.environment import EmailTriageEnv
from env.models import Action
from env.tasks import grade_task1, grade_task2, grade_task3, TASK1_GROUND_TRUTH


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    env = EmailTriageEnv("email-triage-easy")
    env.reset()
    return env

@pytest.fixture
def medium_env():
    env = EmailTriageEnv("email-triage-medium")
    env.reset()
    return env

@pytest.fixture
def hard_env():
    env = EmailTriageEnv("email-triage-hard")
    env.reset()
    return env


# ─────────────────────────────────────────────────────────────
#  Init tests
# ─────────────────────────────────────────────────────────────

def test_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        EmailTriageEnv("does-not-exist")

def test_reset_returns_observation(easy_env):
    obs = easy_env.reset()
    assert obs.total_emails == 5
    assert obs.step_number  == 0
    assert len(obs.inbox)   == 5
    assert obs.done is False

def test_initial_observation_has_task_description(easy_env):
    obs = easy_env.reset()
    assert "classify" in obs.task_description.lower()


# ─────────────────────────────────────────────────────────────
#  Step tests
# ─────────────────────────────────────────────────────────────

def test_classify_valid(easy_env):
    obs, reward, done, info = easy_env.step(
        Action(action_type="classify", email_id="e001",
               category="bug_report", priority="urgent")
    )
    assert reward > 0
    assert done is False
    assert obs.last_action_error is None

def test_classify_wrong_category_raises(easy_env):
    obs, reward, done, info = easy_env.step(
        Action(action_type="classify", email_id="e001",
               category="invalid_cat", priority="urgent")
    )
    assert obs.last_action_error is not None

def test_classify_wrong_priority_raises(easy_env):
    obs, reward, done, info = easy_env.step(
        Action(action_type="classify", email_id="e001",
               category="bug_report", priority="SUPER_URGENT")
    )
    assert obs.last_action_error is not None

def test_step_increments_step_count(easy_env):
    easy_env.step(Action(action_type="classify", email_id="e001",
                          category="bug_report", priority="urgent"))
    assert easy_env._step_count == 1
    easy_env.step(Action(action_type="classify", email_id="e002",
                          category="spam", priority="low"))
    assert easy_env._step_count == 2

def test_done_action_ends_episode(easy_env):
    _, _, done, _ = easy_env.step(Action(action_type="done"))
    assert done is True

def test_archive_removes_email_from_inbox(easy_env):
    inbox_before = len(easy_env._inbox)
    easy_env.step(Action(action_type="archive", email_id="e002"))
    assert len(easy_env._inbox) == inbox_before - 1

def test_skip_moves_email_to_back(easy_env):
    first_id_before = easy_env._inbox[0].id
    easy_env.step(Action(action_type="skip", email_id=first_id_before))
    # The email should now be at the back
    assert easy_env._inbox[-1].id == first_id_before

def test_reply_short_body_raises(easy_env):
    obs, reward, done, info = easy_env.step(
        Action(action_type="reply", email_id="e001", body="Hi")
    )
    assert obs.last_action_error is not None

def test_reply_to_spam_gives_negative_reward(easy_env):
    # e002 is the newsletter/spam email
    _, reward, _, _ = easy_env.step(
        Action(action_type="reply", email_id="e002",
               body="Thank you for your email, we appreciate your interest.")
    )
    assert reward < 0

def test_escalate_requires_reason(easy_env):
    obs, reward, done, _ = easy_env.step(
        Action(action_type="escalate", email_id="e001", reason="Hi")
    )
    assert obs.last_action_error is not None

def test_nonexistent_email_id(easy_env):
    obs, reward, done, _ = easy_env.step(
        Action(action_type="classify", email_id="FAKE-999",
               category="spam", priority="low")
    )
    assert obs.last_action_error is not None

def test_step_after_done_raises(easy_env):
    easy_env.step(Action(action_type="done"))
    with pytest.raises(RuntimeError, match="done"):
        easy_env.step(Action(action_type="classify", email_id="e001",
                              category="spam", priority="low"))


# ─────────────────────────────────────────────────────────────
#  Reward shaping
# ─────────────────────────────────────────────────────────────

def test_correct_classify_gives_positive_reward(easy_env):
    _, reward, _, _ = easy_env.step(
        Action(action_type="classify", email_id="e001",
               category="bug_report", priority="urgent")
    )
    # reward = step_penalty + classify_correct
    assert reward > 0

def test_wrong_classification_gives_negative_reward(easy_env):
    # e002 is obvious spam, calling it urgent bug_report should penalize
    _, reward, _, _ = easy_env.step(
        Action(action_type="classify", email_id="e002",
               category="bug_report", priority="urgent")
    )
    assert reward < 0

def test_completion_reward_scales_with_score(easy_env):
    """Perfect classification should give higher done-reward than empty."""
    # Perfect env
    env_perfect = EmailTriageEnv("email-triage-easy")
    env_perfect.reset()
    env_perfect.step(Action(action_type="classify", email_id="e001", category="bug_report",      priority="urgent"))
    env_perfect.step(Action(action_type="classify", email_id="e002", category="spam",            priority="low"))
    env_perfect.step(Action(action_type="classify", email_id="e003", category="internal",        priority="normal"))
    env_perfect.step(Action(action_type="classify", email_id="e004", category="billing",         priority="high"))
    env_perfect.step(Action(action_type="classify", email_id="e005", category="feature_request", priority="low"))
    _, r_perfect, _, _ = env_perfect.step(Action(action_type="done"))

    # Empty env (no classifications)
    env_empty = EmailTriageEnv("email-triage-easy")
    env_empty.reset()
    _, r_empty, _, _ = env_empty.step(Action(action_type="done"))

    assert r_perfect > r_empty


# ─────────────────────────────────────────────────────────────
#  Grader tests
# ─────────────────────────────────────────────────────────────

def test_grade_task1_perfect_score():
    actions = [
        {"action_type": "classify", "email_id": "e001", "category": "bug_report",      "priority": "urgent"},
        {"action_type": "classify", "email_id": "e002", "category": "spam",            "priority": "low"},
        {"action_type": "classify", "email_id": "e003", "category": "internal",        "priority": "normal"},
        {"action_type": "classify", "email_id": "e004", "category": "billing",         "priority": "high"},
        {"action_type": "classify", "email_id": "e005", "category": "feature_request", "priority": "low"},
    ]
    assert grade_task1(actions) == 1.0

def test_grade_task1_zero_score():
    assert grade_task1([]) == 0.0

def test_grade_task1_partial_score():
    actions = [
        {"action_type": "classify", "email_id": "e001", "category": "bug_report", "priority": "urgent"},
        # only 1 of 5 correct
    ]
    score = grade_task1(actions)
    assert 0.0 < score < 1.0

def test_grade_task1_category_only():
    """Category right, priority wrong → partial score per email."""
    actions = [
        {"action_type": "classify", "email_id": "e001", "category": "bug_report", "priority": "low"},
    ]
    score = grade_task1(actions)
    # category match = 0.6 pts for this email, out of 5 emails total
    expected = 0.6 / 5
    assert abs(score - expected) < 0.001

def test_grade_task2_empty():
    assert grade_task2([]) == 0.0

def test_grade_task3_empty():
    assert grade_task3([]) == 0.0

def test_grade_task3_phishing_archived():
    """Archiving phishing email (e201) should contribute positively."""
    actions = [
        {"action_type": "classify", "email_id": "e201", "category": "spam",   "priority": "low"},
        {"action_type": "archive",  "email_id": "e201"},
    ]
    score = grade_task3(actions)
    assert score > 0.0

def test_grade_task3_security_escalated():
    """Escalating security email (e203 CVE) should score well."""
    actions = [
        {"action_type": "classify", "email_id": "e203", "category": "bug_report", "priority": "urgent"},
        {"action_type": "escalate", "email_id": "e203", "reason": "Critical CVE"},
    ]
    score = grade_task3(actions)
    assert score > 0.0


# ─────────────────────────────────────────────────────────────
#  State tests
# ─────────────────────────────────────────────────────────────

def test_state_returns_dict(easy_env):
    s = easy_env.state()
    assert isinstance(s, dict)
    assert "task"     in s
    assert "step"     in s
    assert "done"     in s
    assert "rewards"  in s

def test_state_updates_after_step(easy_env):
    easy_env.step(Action(action_type="classify", email_id="e001",
                          category="bug_report", priority="urgent"))
    s = easy_env.state()
    assert s["step"] == 1
    assert len(s["rewards"]) == 1


# ─────────────────────────────────────────────────────────────
#  Max step limit
# ─────────────────────────────────────────────────────────────

def test_step_limit_terminates_episode():
    """Verify the environment terminates after max_steps."""
    env = EmailTriageEnv("email-triage-easy")
    env.reset()
    done = False
    for _ in range(20):
        _, _, done, _ = env.step(
            Action(action_type="classify", email_id="e001",
                   category="spam", priority="low")
        )
        if done:
            break
    assert done is True   # must terminate within max_steps=15
