from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState

llm = build_groq_llm(temperature=0.0)


def _build_context_blocks(docs: list[dict]) -> str:
    context_blocks = []

    for idx, doc in enumerate(docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown Source")
        section = metadata.get("section_header", "Unknown Section")
        page = metadata.get("page_number", "Unknown Page")
        score = doc.get("score", None)

        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"

        block = (
            f"DOCUMENT [{idx + 1}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Page: {page}\n"
            f"Retrieval Score: {score_text}\n"
            f"Text:\n{doc.get('text', '')}"
        )
        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


def generate(state: GraphState):
    """
    Generate from the final CRAG-selected docs.
    Supports one audit-driven retry.
    """
    query = state["original_query"]
    selected_docs = state.get("graded_docs", [])
    auditor_feedback = state.get("auditor_feedback", "").strip()
    verify_retries = state.get("verify_retries", 0)

    print(
        f"\n[CRAG + VeriCite] Generating answer using "
        f"{len(selected_docs)} selected docs..."
    )
    print(f"  -> Groq model: {GROQ_MODEL}")
    print(f"  -> Audit retry count: {verify_retries}")

    if not selected_docs:
        return {
            "generation": (
                "I could not find enough grounded evidence in the retrieved papers "
                "to answer this question."
            )
        }

    context = _build_context_blocks(selected_docs)

    feedback_block = ""
    if auditor_feedback:
        feedback_block = f"""
Auditor feedback from the previous answer:
{auditor_feedback}

Important retry instruction:
- Rewrite the answer to directly answer the question and remove unsupported, overly broad, vague, or incomplete claims.
- Stay strictly grounded in the retrieved evidence.
- Include the key detail needed by the question if it is supported.
- Prefer a short, exact, evidence-backed answer over a broader answer.
"""

    prompt = f"""You are an expert academic question-answering assistant.

Your task is to answer the user's question using ONLY the retrieved documents below.

Retrieved Documents:
---
{context}
---

User Question:
{query}
{feedback_block}

Rules:
1. Use ONLY the retrieved documents. Do not use outside knowledge.
2. You MAY synthesize across multiple retrieved documents when the answer is directly supported by combining them.
3. Prefer the evidence that is MOST SPECIFIC to the user's question.
4. Start with the direct answer immediately. Do NOT add headings like "Answer", "Clarification", or similar meta commentary.
5. Keep the answer short and evaluation-friendly.
6. For direct fact lookup questions, prefer exactly one sentence.
7. Avoid unnecessary extra details that are not needed to answer the question.
8. If part of a broader answer is not clearly supported, omit it.
9. Do NOT include inline citations or bracket citations in the answer text.
10. Write clean natural prose only.

Now answer the question.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    print("[CRAG + VeriCite] Answer generation complete.")

    return {"generation": answer}