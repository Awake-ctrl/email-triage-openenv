"""
inference.py
============
Baseline inference script for the Email Triage OpenEnv environment.
Complies exactly with the OpenEnv RL Challenge submission format.

Output format:
  [START] task=<task_name> env=email-triage model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations
import os
import sys
import json
import time
import re
import requests
from typing import Optional

from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_NAME    = os.getenv("TASK_NAME",    "email-triage-easy")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI client ─────────────────────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ── Environment HTTP helpers ──────────────────────────────────────────────────

def env_reset(task_name: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_name": task_name}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_close() -> dict:
    r = requests.post(f"{ENV_BASE_URL}/close", timeout=30)
    r.raise_for_status()
    return r.json()


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage agent. Your job is to process emails
and respond with structured JSON actions.

Available actions:
1. classify  - {"action_type":"classify","email_id":"<id>","category":"<cat>","priority":"<pri>"}
2. reply     - {"action_type":"reply","email_id":"<id>","body":"<text>"}
3. escalate  - {"action_type":"escalate","email_id":"<id>","reason":"<text>"}
4. archive   - {"action_type":"archive","email_id":"<id>"}
5. skip      - {"action_type":"skip","email_id":"<id>"}
6. done      - {"action_type":"done"}

Categories: bug_report | feature_request | billing | general_inquiry | spam | internal
Priorities:  urgent | high | normal | low

Rules:
- Always classify an email before replying/escalating/archiving
- Archive spam and phishing emails — do NOT reply to them
- Escalate security incidents and legal issues
- Reply only to legitimate customer emails that need a response
- Call done when all emails are processed

Respond ONLY with valid JSON. No explanation, no markdown, just the raw JSON object.
"""


def build_user_prompt(obs: dict) -> str:
    parts = [f"Task: {obs.get('task_description', '')}\n"]
    parts.append(f"Progress: {obs.get('processed_count', 0)}/{obs.get('total_emails', 0)} emails processed")
    parts.append(f"Step: {obs.get('step_number', 0)}")

    if obs.get("last_action_result"):
        parts.append(f"Last result: {obs['last_action_result']}")
    if obs.get("last_action_error"):
        parts.append(f"⚠️  Last error: {obs['last_action_error']}")

    inbox = obs.get("inbox", [])
    if inbox:
        parts.append(f"\nINBOX ({len(inbox)} emails remaining):")
        for email in inbox[:5]:   # show up to 5
            parts.append(
                f"\n--- Email ID: {email['id']} ---"
                f"\nFrom:    {email['sender']}"
                f"\nSubject: {email['subject']}"
                f"\nBody:    {email['body'][:300]}"
            )

    if not inbox:
        parts.append("\nAll emails processed. Call done.")

    return "\n".join(parts)


# ── LLM action generation ─────────────────────────────────────────────────────

def get_action(obs: dict, history: list) -> dict:
    """Call LLM to get next action given current observation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])   # keep last 3 turns for context
    messages.append({"role": "user", "content": build_user_prompt(obs)})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",           "", raw)

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: if LLM returns something weird, do a safe skip
        inbox  = obs.get("inbox", [])
        if inbox:
            action = {"action_type": "skip", "email_id": inbox[0]["id"]}
        else:
            action = {"action_type": "done"}

    history.append({"role": "assistant", "content": raw})
    return action


# ── Action → display string ───────────────────────────────────────────────────

def action_to_str(action: dict) -> str:
    atype = action.get("action_type", "unknown")
    eid   = action.get("email_id", "")

    if atype == "classify":
        return (f"classify('{eid}', "
                f"category='{action.get('category')}', "
                f"priority='{action.get('priority')}')")
    if atype == "reply":
        snippet = (action.get("body") or "")[:40].replace("\n", " ")
        return f"reply('{eid}', '{snippet}...')"
    if atype == "escalate":
        return f"escalate('{eid}', reason='{action.get('reason', '')[:40]}')"
    if atype in ("archive", "skip"):
        return f"{atype}('{eid}')"
    return atype


# ── Main episode runner ───────────────────────────────────────────────────────

def run_episode(task_name: str = TASK_NAME) -> bool:
    rewards      = []
    step_count   = 0
    success      = False
    history      = []
    final_score  = 0.0

    # ── [START] ───────────────────────────────────────────────────────────────
    print(f"[START] task={task_name} env=email-triage model={MODEL_NAME}",
          flush=True)

    try:
        obs = env_reset(task_name)

        while True:
            action = get_action(obs, history)

            try:
                result = env_step(action)
            except Exception as exc:
                # Server-side error — emit [STEP] with error and break
                step_count += 1
                rewards.append(0.0)
                error_msg = str(exc)[:120].replace("\n", " ")
                print(
                    f"[STEP] step={step_count} "
                    f"action={action_to_str(action)} "
                    f"reward=0.00 done=false error={error_msg}",
                    flush=True,
                )
                break

            step_count  += 1
            reward       = result["reward"]
            done         = result["done"]
            obs          = result["observation"]
            info         = result.get("info", {})
            error        = obs.get("last_action_error")
            rewards.append(round(reward, 2))

            action_str   = action_to_str(action)
            error_field  = error.replace("\n", " ")[:100] if error else "null"
            done_str     = "true" if done else "false"

            # ── [STEP] ────────────────────────────────────────────────────────
            print(
                f"[STEP] step={step_count} "
                f"action={action_str} "
                f"reward={reward:.2f} "
                f"done={done_str} "
                f"error={error_field}",
                flush=True,
            )

            if done:
                final_score = info.get("final_score") or 0.0
                success     = final_score >= 0.6
                break

            # Rate-limit protection
            time.sleep(0.3)

        # Close environment
        try:
            close_result = env_close()
            if final_score == 0.0:
                final_score = close_result.get("final_score", 0.0)
                success     = final_score >= 0.6
        except Exception:
            pass

    except Exception as exc:
        # Catch-all — always emit [END]
        error_str = str(exc)[:100].replace("\n", " ")
        print(f"# Fatal error: {error_str}", file=sys.stderr, flush=True)

    finally:
        rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
        success_str  = "true" if success else "false"

        # ── [END] ─────────────────────────────────────────────────────────────
        print(
            f"[END] success={success_str} "
            f"steps={step_count} "
            f"rewards={rewards_str}",
            flush=True,
        )

    return success


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Email Triage OpenEnv Inference")
    parser.add_argument(
        "--task",
        default=TASK_NAME,
        choices=["email-triage-easy", "email-triage-medium", "email-triage-hard"],
        help="Task to run",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all three tasks sequentially",
    )
    args = parser.parse_args()

    if args.all_tasks:
        tasks = ["email-triage-easy", "email-triage-medium", "email-triage-hard"]
        results = {}
        for t in tasks:
            print(f"\n{'='*60}", flush=True)
            ok = run_episode(t)
            results[t] = ok
            time.sleep(1)
        print("\n# Summary:", file=sys.stderr)
        for t, ok in results.items():
            print(f"#  {t}: {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    else:
        run_episode(args.task)
