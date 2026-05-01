from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL
from state_agentic import AgenticGraphState

llm = build_groq_llm(temperature=0.0)

# ── Ingest suggestion appended when scholar fallback was used ────────────────
_SCHOLAR_FALLBACK_NOTE = (
    "\n\n---\n"
    "*Note: The local knowledge base did not have sufficient indexed papers "
    "to fully answer this question. The context above was retrieved from "
    "Semantic Scholar and arXiv abstracts (trusted scholarly sources). "
    "For deeper, fully-cited answers, consider ingesting the relevant full "
    "papers into the system via the ingest folder.*"
)


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


def generate(state: AgenticGraphState):
    """
    Generator for agentic_scholar.

    Uses final graded docs (which may include scholarly fallback abstracts).
    Appends an ingest suggestion note when fallback docs were used.
    Supports one audit-driven retry.
    """
    query = state["original_query"]
    selected_docs = state.get("graded_docs", [])
    auditor_feedback = state.get("auditor_feedback", "").strip()
    verify_retries = state.get("verify_retries", 0)
    scholar_used = state.get("scholar_used", False)

    print(
        f"\n[Agentic Scholar] Generating answer using "
        f"{len(selected_docs)} graded docs"
        f"{' (includes scholarly fallback)' if scholar_used else ''}..."
    )
    print(f"  -> Groq model: {GROQ_MODEL}")
    print(f"  -> Audit retry count: {verify_retries}")
    print(f"  -> Scholar fallback used: {scholar_used}")

    if not selected_docs:
        return {
            "generation": (
                "I could not find enough grounded evidence in the retrieved "
                "papers or scholarly APIs to answer this question."
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

    # Append ingest suggestion when scholarly fallback docs were used
    if scholar_used:
        answer += _SCHOLAR_FALLBACK_NOTE

    print("[Agentic Scholar] Answer generation complete.")

    return {"generation": answer}
