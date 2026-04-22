from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState
from config import TOP_K, RETRIEVAL_EVAL_TOP_K, WEAK_SIGNAL_TOP_K

llm = build_groq_llm(temperature=0.0)


def _build_eval_context(docs: list[dict]) -> str:
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


def evaluate_retrieval(state: GraphState):
    """
    CRAG-style retrieval quality gate.

    If original retrieval is sufficient:
    - keep top-k docs directly for answering

    If insufficient:
    - keep top-k original docs as candidate_docs
    - keep weak-signal docs for rewrite prompt
    - trigger rewrite path
    """
    query = state["original_query"]
    retrieved_docs = state.get("retrieved_docs", [])

    candidate_docs = retrieved_docs[:TOP_K]
    eval_docs = retrieved_docs[:RETRIEVAL_EVAL_TOP_K]

    print(
        f"\n[CRAG + VeriCite] Evaluating retrieval quality on "
        f"{len(eval_docs)} docs..."
    )
    print(f"  -> Groq model: {GROQ_MODEL}")

    if not candidate_docs:
        return {
            "candidate_docs": [],
            "weak_signal_docs": [],
            "graded_docs": [],
            "citations_pass": False,
        }

    context = _build_eval_context(eval_docs)

    prompt = f"""You are checking whether retrieved academic document chunks are sufficient to answer a user question.

User Question:
{query}

Retrieved Documents:
---
{context}
---

Decision rule:
- Answer SUFFICIENT if the retrieved documents contain enough grounded, question-specific evidence to answer the question directly.
- Answer INSUFFICIENT if the retrieved documents are too broad, too noisy, off-target, or missing crucial evidence.
- Be strict about specificity.
- Output only one word: SUFFICIENT or INSUFFICIENT.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = re.sub(r"[^A-Z]", "", response.content.strip().upper())

    sufficient = decision == "SUFFICIENT"

    if sufficient:
        print("[CRAG + VeriCite] Retrieval judged SUFFICIENT.")
        return {
            "candidate_docs": candidate_docs,
            "weak_signal_docs": [],
            "graded_docs": candidate_docs,
            "citations_pass": True,
        }

    print("[CRAG + VeriCite] Retrieval judged INSUFFICIENT. Triggering rewrite.")
    return {
        "candidate_docs": candidate_docs,
        "weak_signal_docs": retrieved_docs[:WEAK_SIGNAL_TOP_K],
        "graded_docs": [],
        "citations_pass": False,
    }