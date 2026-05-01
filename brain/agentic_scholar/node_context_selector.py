from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL
from state_agentic import AgenticGraphState
from config import GRADE_TOP_K

llm = build_groq_llm(temperature=0.0)


def _build_context_group(label: str, docs: list[dict]) -> str:
    blocks = []
    for idx, doc in enumerate(docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown Source")
        section = metadata.get("section_header", "Unknown Section")
        page = metadata.get("page_number", "Unknown Page")

        block = (
            f"{label} DOCUMENT [{idx + 1}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Page: {page}\n"
            f"Text:\n{doc.get('text', '')}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def select_best_context(state: AgenticGraphState):
    """
    Compare original candidate docs vs rewritten-query docs.
    Keep the better set as candidate_docs for grading.
    """
    query = state["original_query"]
    original_docs = state.get("candidate_docs", [])[:GRADE_TOP_K]
    rewritten_docs = state.get("retrieved_docs", [])[:GRADE_TOP_K]

    print(
        f"\n[Agentic Scholar] Selecting best context set between "
        f"original ({len(original_docs)}) and rewritten ({len(rewritten_docs)})..."
    )
    print(f"  -> Groq model: {GROQ_MODEL}")

    if not rewritten_docs:
        print("[Agentic Scholar] No rewritten docs found. Falling back to original docs.")
        return {"candidate_docs": original_docs}

    if not original_docs:
        print("[Agentic Scholar] No original candidate docs saved. Using rewritten docs.")
        return {"candidate_docs": rewritten_docs}

    original_context = _build_context_group("ORIGINAL", original_docs)
    rewritten_context = _build_context_group("REWRITTEN", rewritten_docs)

    prompt = f"""You are choosing which retrieved document set is better for answering an academic question.

User Question:
{query}

Original-query document set:
---
{original_context}
---

Rewritten-query document set:
---
{rewritten_context}
---

Decision rule:
- Choose ORIGINAL if the original-query set is more directly relevant, specific, and sufficient.
- Choose REWRITTEN if the rewritten-query set is clearly better targeted to the question.
- Prefer the set that is more question-specific and less noisy.
- Output only one word: ORIGINAL or REWRITTEN.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = re.sub(r"[^A-Z]", "", response.content.strip().upper())

    if decision == "REWRITTEN":
        print("[Agentic Scholar] Selected REWRITTEN context set.")
        return {"candidate_docs": rewritten_docs}

    print("[Agentic Scholar] Selected ORIGINAL context set.")
    return {"candidate_docs": original_docs}
