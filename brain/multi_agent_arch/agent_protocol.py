from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


ALLOWED_NEXT_ACTIONS = {
    "retriever_agent",
    "rewrite_agent",
    "evidence_agent",
    "answer_agent",
    "verification_agent",
    "finish",
}


def extract_json_block(text: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    text = str(text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return default or {}


def normalize_next_action(value: Any, fallback: str = "finish") -> str:
    action = str(value or "").strip()
    if action in ALLOWED_NEXT_ACTIONS:
        return action
    return fallback


def append_agent_trace(
    state: Dict[str, Any],
    *,
    agent_name: str,
    next_action: str,
    status: str,
    summary: str,
) -> List[Dict[str, Any]]:
    trace = list(state.get("agent_trace", []) or [])
    trace.append(
        {
            "agent": agent_name,
            "next_action": next_action,
            "status": status,
            "summary": summary,
        }
    )
    return trace[-100:]


def build_agent_update(
    state: Dict[str, Any],
    *,
    agent_name: str,
    next_action: str,
    decision_payload: Optional[Dict[str, Any]] = None,
    note: str = "",
    status: str = "ok",
    extra_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    decision_payload = decision_payload or {}
    extra_updates = extra_updates or {}
    next_action = normalize_next_action(next_action)

    agent_decisions = dict(state.get("agent_decisions", {}) or {})
    agent_notes = dict(state.get("agent_notes", {}) or {})
    agent_status = dict(state.get("agent_status", {}) or {})

    agent_decisions[agent_name] = decision_payload
    agent_notes[agent_name] = note
    agent_status[agent_name] = {
        "status": status,
        "next_action": next_action,
    }

    trace = append_agent_trace(
        state,
        agent_name=agent_name,
        next_action=next_action,
        status=status,
        summary=note or str(decision_payload),
    )

    return {
        "active_agent": agent_name,
        "next_action_recommendation": next_action,
        "agent_decisions": agent_decisions,
        "agent_notes": agent_notes,
        "agent_status": agent_status,
        "agent_trace": trace,
        **extra_updates,
    }