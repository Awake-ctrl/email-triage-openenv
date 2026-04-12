---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server.py
pinned: false
---


## 🌐 Live Demo

Hugging Face Space:
https://huggingface.co/spaces/saiKiran112201044/email-triage-openenv

API Endpoint:
https://saikiran112201044-email-triage-openenv.hf.space

Example health check:
https://saikiran112201044-email-triage-openenv.hf.space/health

# 📧 Email Triage — OpenEnv RL Environment

A real-world email triage environment for the Meta OpenEnv RL Hackathon.
An agent reads an inbox of emails and must classify, reply to, escalate,
and archive them correctly — just like a real customer support engineer.

[![OpenEnv](https://img.shields.io/badge/openenv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)

---

## 🧠 Motivation

Email triage is one of the most universal knowledge-work tasks:
every company routes, prioritises and responds to email.
It requires natural language understanding, business context reasoning,
and multi-step decision-making — ideal for evaluating LLM agents.

Unlike toy benchmarks, this environment penalises:
- Replying to phishing / spam emails
- Failing to escalate security incidents
- Mislabelling urgent issues as low priority
- Wasting steps without making progress

---

## 🗂️ Project Structure

```
openenv-email-triage/
├── inference.py          ← Hackathon submission entry point (root-level, required)
├── server.py             ← FastAPI environment server
├── openenv.yaml          ← OpenEnv metadata
├── requirements.txt
├── Dockerfile
│
├── env/
│   ├── __init__.py
│   ├── environment.py    ← EmailTriageEnv class (reset/step/state/close)
│   ├── models.py         ← Pydantic models: Observation, Action, Email, StepResult
│   └── tasks.py          ← 3 task definitions + programmatic graders
│
└── tests/
    └── test_environment.py
```

---

## 🎮 Tasks

### Task 1 — `email-triage-easy` (Easy)
**Objective:** Classify 5 clearly-labelled emails by category and priority.

| Email | Expected Category | Expected Priority |
|-------|------------------|-------------------|
| Production server DOWN | `bug_report` | `urgent` |
| Free iPhone newsletter | `spam` | `low` |
| Team standup notes | `internal` | `normal` |
| Invoice dispute | `billing` | `high` |
| Dark mode request | `feature_request` | `low` |

- Max steps: **15**
- Reward: classification accuracy only
- Difficulty: **⭐ Easy**

---

### Task 2 — `email-triage-medium` (Medium)
**Objective:** Classify + reply + escalate/archive 6 mixed customer emails.

The agent must:
1. Classify each email with the right category and priority
2. Write a relevant reply for customer emails (keyword-quality graded)
3. Escalate the angry refund customer (3rd time asking)
4. Archive spam/test emails without replying

- Max steps: **30**
- Difficulty: **⭐⭐ Medium**

---

### Task 3 — `email-triage-hard` (Hard)
**Objective:** Handle 8 adversarial emails including phishing, CVE security alerts, and ambiguous contracts.

Key challenges:
- A **phishing email** disguised as a bank security alert — must archive, not reply
- A **CVE-9.8 GitHub alert** — must escalate immediately
- A **potential data breach notification** — must escalate
- An internal CTO asking for analysis of the CVE — must reply with technical context
- A multi-threaded **contract amendment** with an EOD deadline

- Max steps: **40**
- Difficulty: **⭐⭐⭐ Hard**

---

## 🔭 Observation Space

```python
class Observation(BaseModel):
    inbox:               List[Email]       # all unprocessed emails
    current_email:       Optional[Email]   # top of inbox
    processed_count:     int
    total_emails:        int
    step_number:         int
    last_action_result:  str               # human-readable feedback
    last_action_error:   Optional[str]     # set if last action was invalid
    task_description:    str               # injected into LLM prompt
    done:                bool
```

Each `Email` contains: `id`, `sender`, `subject`, `body`, `timestamp`,
`has_attachment`, `thread_id`.

---

## ⚡ Action Space

| Action | Fields | Description |
|--------|--------|-------------|
| `classify` | `email_id`, `category`, `priority` | Assign category + priority |
| `reply` | `email_id`, `body` | Send a reply (min 10 chars) |
| `escalate` | `email_id`, `reason` | Escalate to human agent |
| `archive` | `email_id` | Archive the email (removes from inbox) |
| `skip` | `email_id` | Move to back of inbox |
| `done` | — | Signal task completion |

**Categories:** `bug_report` · `feature_request` · `billing` · `general_inquiry` · `spam` · `internal`

**Priorities:** `urgent` · `high` · `normal` · `low`

---

## 🏆 Reward Function

The reward is **shaped throughout the trajectory**, not just at completion.

| Event | Reward |
|-------|--------|
| Step taken | −0.01 (efficiency penalty) |
| Classify correctly (heuristic) | +0.15 |
| Classify incorrectly (heuristic) | −0.10 |
| Reply to a legitimate email | +0.20 |
| Reply to spam / phishing | −0.05 |
| Escalate correctly | +0.15 |
| Escalate wrongly (escalate spam) | −0.10 |
| Archive spam | +0.10 |
| Archive non-spam | −0.10 |
| Completion bonus | +0.20 × final_score |

Final score is computed by a **deterministic grader** (0.0 → 1.0)
that evaluates classification accuracy, reply keyword quality,
escalation decisions, and archival decisions.

---

## 📊 Baseline Performance

| Task | GPT-4.1-mini | Random Agent |
|------|-------------|--------------|
| `email-triage-easy` | ~0.82 | ~0.12 |
| `email-triage-medium` | ~0.68 | ~0.08 |
| `email-triage-hard` | ~0.55 | ~0.05 |

Success threshold: final_score ≥ 0.60

---

## 🚀 Setup & Usage

### Option A — Docker (recommended)

```bash
# 1. Build
docker build -t email-triage-env .

# 2. Run the environment server
docker run -p 7860:7860 email-triage-env

# 3. In a separate terminal, run inference
export HF_TOKEN=your_hf_or_openai_token
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export ENV_BASE_URL=http://localhost:7860

python inference.py --task email-triage-easy
# or run all three tasks:
python inference.py --all-tasks
```

### Option B — Local Python

```bash
pip install -r requirements.txt

# Terminal 1: start env server
python server.py

# Terminal 2: run inference
export HF_TOKEN=your_token
python inference.py --task email-triage-medium
```

### Option C — Environment only (no server)

```python
from env import EmailTriageEnv
from env.models import Action

env = EmailTriageEnv("email-triage-easy")
obs = env.reset()

while not obs.done:
    # your agent logic here
    action = Action(
        action_type="classify",
        email_id=obs.current_email.id,
        category="bug_report",
        priority="urgent",
    )
    obs, reward, done, info = env.step(action)

print(f"Final score: {env.final_score:.4f}")
```

---

## 🌐 API Endpoints (server.py)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/reset` | Start a new episode `{"task_name": "..."}` |
| `POST` | `/step` | Take an action `{"action": {...}}` |
| `GET` | `/state` | Current environment state |
| `POST` | `/close` | End episode, get final score |

---

## 🧪 Running Tests

```bash
# With pytest installed
pytest tests/ -v

# Or run the built-in logic checks
python -m pytest tests/test_environment.py -v --tb=short
```

---

## 📋 Output Format

The inference script emits exactly the required OpenEnv format:

```
[START] task=email-triage-easy env=email-triage model=gpt-4.1-mini
[STEP] step=1 action=classify('e001', category='bug_report', priority='urgent') reward=0.14 done=false error=null
[STEP] step=2 action=classify('e002', category='spam', priority='low') reward=0.14 done=false error=null
[STEP] step=3 action=classify('e003', category='internal', priority='normal') reward=0.14 done=false error=null
[STEP] step=4 action=classify('e004', category='billing', priority='high') reward=0.14 done=false error=null
[STEP] step=5 action=classify('e005', category='feature_request', priority='low') reward=0.14 done=false error=null
[STEP] step=6 action=done reward=0.19 done=true error=null
[END] success=true steps=6 rewards=0.14,0.14,0.14,0.14,0.14,0.19
```

---

## ⚙️ Environment Variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `HF_TOKEN` | — | **Yes** | HuggingFace / OpenAI API key |
| `API_BASE_URL` | `https://api.openai.com/v1` | No | LLM API endpoint |
| `MODEL_NAME` | `gpt-4.1-mini` | No | Model identifier |
| `ENV_BASE_URL` | `http://localhost:7860` | No | Environment server URL |
| `TASK_NAME` | `email-triage-easy` | No | Default task to run |

---

## 🏗️ Hardware Requirements

This environment runs well within the contest limits:

| Resource | Required | Limit |
|----------|----------|-------|
| CPU | < 0.5 vCPU (idle) | 2 vCPU |
| RAM | ~120 MB | 8 GB |
| Disk | ~50 MB (no model weights) | — |

No GPU required. No model weights downloaded. All LLM calls go to external API.

---

## 📝 License

MIT
