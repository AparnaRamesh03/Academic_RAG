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
    CRAG-inspired retrieval quality check.

    Design:
    - take top-k retrieved docs from the original query
    - judge whether they are sufficient to answer the question
    - if sufficient: pass them directly to generation
    - if insufficient: keep them as the original candidate set and trigger rewrite

    Note:
    We reuse `citations_pass` as a generic boolean flag for
    retrieval sufficiency in this architecture.
    """
    query = state["original_query"]
    retrieved_docs = state.get("retrieved_docs", [])

    candidate_docs = retrieved_docs[:TOP_K]
    eval_docs = retrieved_docs[:RETRIEVAL_EVAL_TOP_K]

    print(
        f"\n[CRAG Rewrite] Evaluating retrieval quality on "
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
- Answer SUFFICIENT if the retrieved documents contain enough grounded information to answer the question directly and specifically.
- Answer INSUFFICIENT if the retrieved documents are too broad, too noisy, off-target, or missing crucial question-specific evidence.
- Be strict about specificity. If the question asks for a comparison, explanation, mechanism, or detail that is not clearly supported yet, answer INSUFFICIENT.
- Output only one word: SUFFICIENT or INSUFFICIENT.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().upper()
    decision = re.sub(r"[^A-Z]", "", decision)

    sufficient = decision == "SUFFICIENT"
    # sufficient = False
    if sufficient:
        print("[CRAG Rewrite] Retrieval judged SUFFICIENT.")
        return {
            "candidate_docs": candidate_docs,
            "weak_signal_docs": [],
            "graded_docs": candidate_docs,
            "citations_pass": True,
        }

    print("[CRAG Rewrite] Retrieval judged INSUFFICIENT. Triggering rewrite.")
    return {
        "candidate_docs": candidate_docs,
        "weak_signal_docs": retrieved_docs[:WEAK_SIGNAL_TOP_K],
        "graded_docs": [],
        "citations_pass": False,
    }