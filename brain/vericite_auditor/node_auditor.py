from pathlib import Path
import sys
import re

# Allow this folder to import shared files from ../
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState

llm = build_groq_llm(temperature=0.0)


def _build_context_blocks(docs: list[dict]) -> str:
    blocks = []

    for idx, doc in enumerate(docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown Source")
        section = metadata.get("section_header", "Unknown Section")
        page = metadata.get("page_number", "Unknown Page")

        block = (
            f"DOCUMENT [{idx + 1}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Page: {page}\n"
            f"Text:\n{doc.get('text', '')}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def _parse_audit_response(text: str) -> tuple[bool, str]:
    """
    Expected format:
    DECISION: PASS or FAIL
    FEEDBACK: <short text or NONE>
    """
    decision_match = re.search(r"DECISION\s*:\s*(PASS|FAIL)", text, re.IGNORECASE)
    feedback_match = re.search(r"FEEDBACK\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    decision = decision_match.group(1).upper() if decision_match else "FAIL"
    feedback = feedback_match.group(1).strip() if feedback_match else "Unsupported or overly broad answer."

    if feedback.upper() == "NONE":
        feedback = ""

    return decision == "PASS", feedback


def audit_answer(state: GraphState):
    """
    VeriCite-inspired answer auditor.

    Checks whether the generated answer is fully supported by the retrieved docs.
    If not, returns short feedback for one retry generation.
    """
    query = state["original_query"]
    answer = state.get("generation", "").strip()
    selected_docs = state.get("graded_docs", [])
    verify_retries = state.get("verify_retries", 0)

    print("\n[VeriCite Auditor] Auditing generated answer...")
    print(f"  -> Groq model: {GROQ_MODEL}")

    if not answer or not selected_docs:
        print("[VeriCite Auditor] Empty answer or no docs. Marking FAIL.")
        return {
            "citations_pass": False,
            "auditor_feedback": "The answer is empty or not grounded in retrieved evidence.",
            "verify_retries": verify_retries + 1,
        }

    context = _build_context_blocks(selected_docs)
    prompt = f"""You are auditing whether an academic QA answer is fully supported and sufficiently specific given the retrieved evidence.

User Question:
{query}

Retrieved Documents:
---
{context}
---

Generated Answer:
{answer}

Audit rules:
- PASS only if the answer is both:
  1. fully supported by the retrieved documents, and
  2. sufficiently specific to answer the actual user question.
- FAIL if the answer:
  - contains unsupported or weakly supported claims,
  - is too broad or generic,
  - misses key details needed for the question,
  - gives background instead of the exact asked answer,
  - answers only partially when the retrieved documents support a more complete answer.
- Be strict.
- Prefer FAIL if unsure.
- Minor wording differences are acceptable only if the meaning is clearly supported and complete.

If you FAIL the answer, give one short feedback sentence telling how to tighten or correct it.

Output exactly in this format:
DECISION: PASS or FAIL
FEEDBACK: <short feedback or NONE>
"""
#     prompt = f"""You are auditing whether an academic QA answer is fully supported by retrieved evidence.

# User Question:
# {query}

# Retrieved Documents:
# ---
# {context}
# ---

# Generated Answer:
# {answer}

# Audit rules:
# - PASS if the answer is fully supported by the retrieved documents.
# - FAIL if the answer contains unsupported, overly broad, speculative, or weakly grounded claims.
# - Be strict about unsupported extra details.
# - Minor wording differences are acceptable if the meaning is clearly supported.
# - If FAIL, give one short feedback sentence explaining what to remove or tighten.

# Output exactly in this format:
# DECISION: PASS or FAIL
# FEEDBACK: <short feedback or NONE>
# """

    response = llm.invoke([HumanMessage(content=prompt)])
    passed, feedback = _parse_audit_response(response.content.strip())

    if passed:
        print("[VeriCite Auditor] Audit PASS.")
        return {
            "citations_pass": True,
            "auditor_feedback": "",
        }

    print("[VeriCite Auditor] Audit FAIL.")
    print(f"  -> Feedback: {feedback}")

    return {
        "citations_pass": False,
        "auditor_feedback": feedback,
        "verify_retries": verify_retries + 1,
    }