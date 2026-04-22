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
from config import TOP_K

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


def select_best_context(state: GraphState):
    """
    Compare:
    - original top-k docs saved in candidate_docs
    - rewritten-query top-k docs from current retrieved_docs

    Keep the set that is more likely to answer the question accurately.
    """
    query = state["original_query"]
    original_docs = state.get("candidate_docs", [])[:TOP_K]
    rewritten_docs = state.get("retrieved_docs", [])[:TOP_K]

    print(
        f"\n[CRAG Rewrite] Selecting best context set between "
        f"original ({len(original_docs)}) and rewritten ({len(rewritten_docs)})..."
    )
    print(f"  -> Groq model: {GROQ_MODEL}")

    if not rewritten_docs:
        print("[CRAG Rewrite] No rewritten docs found. Falling back to original docs.")
        return {
            "graded_docs": original_docs
        }

    if not original_docs:
        print("[CRAG Rewrite] No original candidate docs saved. Using rewritten docs.")
        return {
            "graded_docs": rewritten_docs
        }

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
    decision = response.content.strip().upper()
    decision = re.sub(r"[^A-Z]", "", decision)

    if decision == "REWRITTEN":
        print("[CRAG Rewrite] Selected REWRITTEN context set.")
        return {
            "graded_docs": rewritten_docs
        }

    print("[CRAG Rewrite] Selected ORIGINAL context set.")
    return {
        "graded_docs": original_docs
    }