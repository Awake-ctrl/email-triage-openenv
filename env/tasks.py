"""
tasks.py
--------
Three email-triage tasks spanning easy → medium → hard.
Each task exposes:
  - emails        : List[Email]          the inbox
  - description   : str                  shown to the agent
  - max_steps     : int
  - grade(actions) → float              programmatic grader [0.0, 1.0]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
from env.models import Email


# ─────────────────────────────────────────────────────────────
#  TASK 1  —  Easy: classify five obvious emails
# ─────────────────────────────────────────────────────────────

TASK1_EMAILS: List[Email] = [
    Email(
        id="e001",
        sender="john.smith@customer.com",
        subject="URGENT: Production server is DOWN",
        body=(
            "Our entire production environment went offline 10 minutes ago. "
            "Thousands of users are affected. We need immediate help. "
            "This is causing massive revenue loss every minute."
        ),
        timestamp="2024-01-15T09:00:00Z",
    ),
    Email(
        id="e002",
        sender="newsletter@promotions.example.com",
        subject="You won a FREE iPhone 15!!! Click now!!!",
        body=(
            "Congratulations! You have been selected to receive a FREE iPhone 15. "
            "Click the link below to claim your prize immediately. "
            "Offer expires in 24 hours. Don't miss out!"
        ),
        timestamp="2024-01-15T09:05:00Z",
    ),
    Email(
        id="e003",
        sender="alice.dev@company.com",
        subject="Team standup notes - Jan 15",
        body=(
            "Hi team, here are the notes from today's standup:\n"
            "- Alice: working on auth module\n"
            "- Bob: finishing API docs\n"
            "- Carol: reviewing PRs\n"
            "Next standup tomorrow at 9 AM."
        ),
        timestamp="2024-01-15T09:10:00Z",
    ),
    Email(
        id="e004",
        sender="billing@customer-corp.com",
        subject="Invoice #INV-2024-089 dispute",
        body=(
            "Hello, I'm writing to dispute invoice #INV-2024-089 for $4,200. "
            "We were charged for the Enterprise plan but we only signed up for Standard. "
            "Please review and issue a corrected invoice."
        ),
        timestamp="2024-01-15T09:15:00Z",
    ),
    Email(
        id="e005",
        sender="dev@startup.io",
        subject="Feature request: dark mode support",
        body=(
            "Hi! Love the product. One thing that would make it even better "
            "is dark mode support. Many of us work late and the bright interface "
            "is straining. Would this be possible to add?"
        ),
        timestamp="2024-01-15T09:20:00Z",
    ),
]

TASK1_GROUND_TRUTH: Dict[str, Dict[str, str]] = {
    "e001": {"category": "bug_report",       "priority": "urgent"},
    "e002": {"category": "spam",             "priority": "low"},
    "e003": {"category": "internal",         "priority": "normal"},
    "e004": {"category": "billing",          "priority": "high"},
    "e005": {"category": "feature_request",  "priority": "low"},
}


def grade_task1(actions_log: List[Dict[str, Any]]) -> float:
    """
    Score: proportion of emails correctly classified.
    Category match = 0.6 pts, priority match = 0.4 pts per email.
    """
    classified: Dict[str, Dict] = {}
    for a in actions_log:
        if a.get("action_type") == "classify" and a.get("email_id"):
            classified[a["email_id"]] = a

    if not classified:
        return 0.01

    total_score = 0.0
    for eid, truth in TASK1_GROUND_TRUTH.items():
        if eid not in classified:
            continue
        cat_ok  = classified[eid].get("category") == truth["category"]
        pri_ok  = classified[eid].get("priority") == truth["priority"]
        total_score += 0.6 * cat_ok + 0.4 * pri_ok

    # return round(total_score / len(TASK1_GROUND_TRUTH), 4)
    score = total_score / len(TASK1_GROUND_TRUTH)

    # Clamp score into (0,1)
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99
    if score == 0.0:
        score = 0.01
    return float(f"{score:.4f}")


# ─────────────────────────────────────────────────────────────
#  TASK 2  —  Medium: triage + reply to customer emails
# ─────────────────────────────────────────────────────────────

TASK2_EMAILS: List[Email] = [
    Email(
        id="e101",
        sender="frustrated.user@client.com",
        subject="Cannot login - account locked",
        body=(
            "I've been trying to login for the past hour and my account keeps "
            "getting locked. I have an important presentation in 2 hours and I need "
            "access to my files. I've already reset my password twice. Nothing works. "
            "This is completely unacceptable."
        ),
        timestamp="2024-01-15T10:00:00Z",
    ),
    Email(
        id="e102",
        sender="sales.lead@bigcorp.com",
        subject="Interested in Enterprise plan - 500 seats",
        body=(
            "Hi, I'm the IT Director at BigCorp. We're evaluating your product for "
            "company-wide deployment (approx 500 seats). Could you send pricing info "
            "and schedule a demo? Decision needs to be made by end of month."
        ),
        timestamp="2024-01-15T10:05:00Z",
    ),
    Email(
        id="e103",
        sender="angry@customer.net",
        subject="Refund request - 3rd time asking",
        body=(
            "I have emailed you THREE TIMES about my refund for order #ORD-5521. "
            "Each time I get a generic response saying someone will follow up. "
            "It has been 3 weeks. If I don't hear back today I am filing a chargeback "
            "and leaving a 1-star review on every platform."
        ),
        timestamp="2024-01-15T10:10:00Z",
    ),
    Email(
        id="e104",
        sender="developer@techstartup.com",
        subject="API rate limit question",
        body=(
            "Hey team, quick question: what's the rate limit for the /events endpoint "
            "on the Standard plan? Our integration is hitting 429s during peak hours. "
            "Should we upgrade or is there a way to batch requests?"
        ),
        timestamp="2024-01-15T10:15:00Z",
    ),
    Email(
        id="e105",
        sender="ceo@vip-client.com",
        subject="Renewal discussion",
        body=(
            "Hi, our annual contract comes up for renewal next month. We've been happy "
            "overall but want to discuss pricing and new features before committing. "
            "Can we schedule a call with your account management team this week?"
        ),
        timestamp="2024-01-15T10:20:00Z",
    ),
    Email(
        id="e106",
        sender="test@test.com",
        subject="test",
        body="test",
        timestamp="2024-01-15T10:25:00Z",
    ),
]

TASK2_GROUND_TRUTH = {
    "e101": {"category": "bug_report",      "priority": "urgent",   "needs_reply": True,  "needs_escalate": False},
    "e102": {"category": "general_inquiry", "priority": "high",     "needs_reply": True,  "needs_escalate": False},
    "e103": {"category": "billing",         "priority": "urgent",   "needs_reply": True,  "needs_escalate": True},
    "e104": {"category": "general_inquiry", "priority": "normal",   "needs_reply": True,  "needs_escalate": False},
    "e105": {"category": "general_inquiry", "priority": "high",     "needs_reply": True,  "needs_escalate": False},
    "e106": {"category": "spam",            "priority": "low",      "needs_reply": False, "needs_escalate": False},
}

REPLY_KEYWORDS = {
    "e101": ["apologize", "unlock", "immediate", "priority", "help"],
    "e102": ["demo", "pricing", "enterprise", "schedule", "seats"],
    "e103": ["apologize", "refund", "escalate", "priority", "resolve"],
    "e104": ["rate limit", "batch", "429", "upgrade", "standard"],
    "e105": ["schedule", "call", "account", "renewal", "team"],
}


def grade_task2(actions_log: List[Dict[str, Any]]) -> float:
    classified = {}
    replies    = {}
    escalated  = set()

    for a in actions_log:
        eid  = a.get("email_id", "")
        atype = a.get("action_type")
        if atype == "classify":
            classified[eid] = a
        elif atype == "reply":
            replies[eid] = (a.get("body") or "").lower()
        elif atype == "escalate":
            escalated.add(eid)

    score = 0.0
    weight_per_email = 1.0 / len(TASK2_GROUND_TRUTH)

    for eid, truth in TASK2_GROUND_TRUTH.items():
        email_score = 0.0

        # Classification (40%)
        if eid in classified:
            cat_ok = classified[eid].get("category") == truth["category"]
            pri_ok = classified[eid].get("priority") == truth["priority"]
            email_score += 0.40 * (0.6 * cat_ok + 0.4 * pri_ok)

        # Reply quality (40%) — keyword presence
        if truth["needs_reply"] and eid in replies:
            keywords = REPLY_KEYWORDS.get(eid, [])
            if keywords:
                hits = sum(1 for kw in keywords if kw in replies[eid])
                email_score += 0.40 * (hits / len(keywords) )
        elif not truth["needs_reply"] and eid not in replies:
            email_score += 0.40  # correctly did NOT reply to spam/test

        # Escalation (20%)
        if truth["needs_escalate"] and eid in escalated:
            email_score += 0.20
        elif not truth["needs_escalate"] and eid not in escalated:
            email_score += 0.20

        score += email_score * weight_per_email
    score = min(max(score, 0.01), 0.99)
    if score == 0.0:
        score = 0.01
    return float(f"{score:.4f}")


# ─────────────────────────────────────────────────────────────
#  TASK 3  —  Hard: full inbox triage under time pressure
#             Classify + reply + escalate + archive spam
#             with adversarial / ambiguous emails
# ─────────────────────────────────────────────────────────────

TASK3_EMAILS: List[Email] = [
    Email(
        id="e201",
        sender="security@bank-alerts.net",
        subject="Urgent: Verify your account",
        body=(
            "Dear Customer, we detected unusual activity on your account. "
            "Please verify immediately by clicking: http://secure-bank-login.suspicious.xyz "
            "Failure to verify within 1 hour will result in account suspension."
        ),
        timestamp="2024-01-15T11:00:00Z",
    ),
    Email(
        id="e202",
        sender="m.johnson@enterprise-client.com",
        subject="RE: RE: RE: Contract amendment - Section 4.2",
        body=(
            "Per our discussion on Friday, we need the amendment to Section 4.2 "
            "to reflect the new SLA commitments (99.95% uptime). Our legal team "
            "has reviewed and the only blocking item is the indemnification clause. "
            "Can we hop on a 15-min call today? This needs to close by EOD."
        ),
        timestamp="2024-01-15T11:05:00Z",
        thread_id="thread-202",
    ),
    Email(
        id="e203",
        sender="noreply@github.com",
        subject="[GitHub] Security alert: vulnerable dependency",
        body=(
            "A dependency in your repository 'core-api' has a known vulnerability: "
            "CVE-2024-1234 (CVSS 9.8 - CRITICAL). "
            "Affected: lodash < 4.17.21. "
            "Recommendation: Update immediately to patch this remote code execution vulnerability."
        ),
        timestamp="2024-01-15T11:10:00Z",
    ),
    Email(
        id="e204",
        sender="hr@company.com",
        subject="Q1 All-hands meeting - please RSVP",
        body=(
            "Hi everyone, please RSVP for the Q1 all-hands meeting scheduled for "
            "January 30th at 2 PM EST. The agenda includes product roadmap, "
            "team achievements, and OKR review. Lunch will be provided."
        ),
        timestamp="2024-01-15T11:15:00Z",
    ),
    Email(
        id="e205",
        sender="data.breach@darkweb-notify.org",
        subject="Your customer data was leaked",
        body=(
            "We have detected your company's customer database (47,000 records) "
            "being sold on underground forums. This includes names, emails, and "
            "hashed passwords. We recommend immediate action and notifying affected users."
        ),
        timestamp="2024-01-15T11:20:00Z",
    ),
    Email(
        id="e206",
        sender="partner@integration-vendor.com",
        subject="API breaking change in v3.0 - action required",
        body=(
            "This is an advance notice that our API v2.x will be deprecated on "
            "March 1st, 2024. All integrations must migrate to v3.0. "
            "Breaking changes include: new auth flow, response format change for /orders. "
            "Migration guide attached. Please confirm receipt."
        ),
        timestamp="2024-01-15T11:25:00Z",
        has_attachment=True,
    ),
    Email(
        id="e207",
        sender="random@newsletter.biz",
        subject="Top 10 productivity hacks you NEED to know",
        body=(
            "Boost your productivity with these 10 amazing hacks! "
            "Number 3 will blow your mind! Click to read more... "
            "Unsubscribe | View in browser"
        ),
        timestamp="2024-01-15T11:30:00Z",
    ),
    Email(
        id="e208",
        sender="cto@company.com",
        subject="Need your analysis on the GitHub security alert ASAP",
        body=(
            "Hey, I saw the GitHub security alert about CVE-2024-1234 in core-api. "
            "Can you assess the blast radius and whether we need an emergency patch? "
            "I need a summary in 30 minutes for the board call."
        ),
        timestamp="2024-01-15T11:31:00Z",
    ),
]

TASK3_GROUND_TRUTH = {
    "e201": {"category": "spam",            "priority": "low",    "needs_reply": False, "needs_escalate": False, "needs_archive": True},
    "e202": {"category": "general_inquiry", "priority": "urgent", "needs_reply": True,  "needs_escalate": False, "needs_archive": False},
    "e203": {"category": "bug_report",      "priority": "urgent", "needs_reply": False, "needs_escalate": True,  "needs_archive": False},
    "e204": {"category": "internal",        "priority": "normal", "needs_reply": False, "needs_escalate": False, "needs_archive": False},
    "e205": {"category": "general_inquiry", "priority": "urgent", "needs_reply": False, "needs_escalate": True,  "needs_archive": False},
    "e206": {"category": "general_inquiry", "priority": "high",   "needs_reply": True,  "needs_escalate": False, "needs_archive": False},
    "e207": {"category": "spam",            "priority": "low",    "needs_reply": False, "needs_escalate": False, "needs_archive": True},
    "e208": {"category": "internal",        "priority": "urgent", "needs_reply": True,  "needs_escalate": False, "needs_archive": False},
}

TASK3_REPLY_KEYWORDS = {
    "e202": ["call", "today", "amendment", "4.2", "sla", "schedule"],
    "e206": ["confirm", "received", "migration", "v3", "march", "noted"],
    "e208": ["cve", "critical", "patch", "blast radius", "lodash", "assess", "rce", "update"],
}


def grade_task3(actions_log: List[Dict[str, Any]]) -> float:
    classified = {}
    replies    = {}
    escalated  = set()
    archived   = set()

    for a in actions_log:
        eid   = a.get("email_id", "")
        atype = a.get("action_type")
        if atype == "classify":
            classified[eid] = a
        elif atype == "reply":
            replies[eid] = (a.get("body") or "").lower()
        elif atype == "escalate":
            escalated.add(eid)
        elif atype == "archive":
            archived.add(eid)

    score = 0.0
    n = len(TASK3_GROUND_TRUTH)

    for eid, truth in TASK3_GROUND_TRUTH.items():
        e_score = 0.0

        # Classification 30%
        if eid in classified:
            cat_ok = classified[eid].get("category") == truth["category"]
            pri_ok = classified[eid].get("priority") == truth["priority"]
            e_score += 0.30 * (0.6 * cat_ok + 0.4 * pri_ok)

        # Reply quality 25%
        if truth["needs_reply"] and eid in replies:
            kws = TASK3_REPLY_KEYWORDS.get(eid, [])
            hits = sum(1 for kw in kws if kw in replies[eid]) if kws else 1
            e_score += 0.25 * (hits / len(kws)) if kws else 0.25
        elif not truth["needs_reply"] and eid not in replies:
            e_score += 0.25

        # Escalation 25%
        if truth["needs_escalate"] and eid in escalated:
            e_score += 0.25
        elif not truth["needs_escalate"] and eid not in escalated:
            e_score += 0.25

        # Archive spam 20%
        if truth["needs_archive"] and eid in archived:
            e_score += 0.20
        elif not truth["needs_archive"] and eid not in archived:
            e_score += 0.20

        score += e_score / n


    # Clamp score into (0,1)
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    return float(f"{score:.4f}")

# ─────────────────────────────────────────────────────────────
#  Task registry
# ─────────────────────────────────────────────────────────────

@dataclass
class TaskDefinition:
    name:        str
    emails:      List[Email]
    description: str
    max_steps:   int
    grader:      callable
    difficulty:  str


TASKS: Dict[str, TaskDefinition] = {
    "email-triage-easy": TaskDefinition(
        name="email-triage-easy",
        emails=TASK1_EMAILS,
        description=(
            "You are an email triage agent. You have 5 emails in your inbox. "
            "Your job is to classify EACH email with the correct category and priority.\n\n"
            "Categories: bug_report | feature_request | billing | general_inquiry | spam | internal\n"
            "Priorities: urgent | high | normal | low\n\n"
            "Use the classify action for each email, then call done when finished."
        ),
        max_steps=15,
        grader=grade_task1,
        difficulty="easy",
    ),
    "email-triage-medium": TaskDefinition(
        name="email-triage-medium",
        emails=TASK2_EMAILS,
        description=(
            "You are a customer support triage agent. You have 6 emails to handle.\n"
            "For each email you must:\n"
            "  1. classify(email_id, category, priority)\n"
            "  2. reply(email_id, body)  — if the customer needs a response\n"
            "  3. escalate(email_id, reason) — if the issue requires human escalation\n"
            "  4. archive(email_id) — for spam or test emails\n\n"
            "Categories: bug_report | feature_request | billing | general_inquiry | spam | internal\n"
            "Priorities: urgent | high | normal | low\n\n"
            "Call done when all emails are processed."
        ),
        max_steps=30,
        grader=grade_task2,
        difficulty="medium",
    ),
    "email-triage-hard": TaskDefinition(
        name="email-triage-hard",
        emails=TASK3_EMAILS,
        description=(
            "You are a senior support triage agent. You have 8 emails including "
            "phishing attempts, security incidents, and time-critical requests.\n\n"
            "For EACH email:\n"
            "  1. classify(email_id, category, priority)\n"
            "  2. reply(email_id, body)  — for emails requiring a response\n"
            "  3. escalate(email_id, reason) — for security or legal incidents\n"
            "  4. archive(email_id) — spam and phishing emails\n\n"
            "IMPORTANT: Some emails may be phishing — do not reply to them, archive instead.\n"
            "Security vulnerabilities must be escalated immediately.\n\n"
            "Categories: bug_report | feature_request | billing | general_inquiry | spam | internal\n"
            "Priorities: urgent | high | normal | low\n\n"
            "Call done when all emails are processed."
        ),
        max_steps=40,
        grader=grade_task3,
        difficulty="hard",
    ),
}
