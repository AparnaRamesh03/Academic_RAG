from pathlib import Path
import sys

# Allow this folder to import shared files from ../
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState

llm = build_groq_llm(temperature=0.0)


def _build_weak_context(docs: list[dict]) -> str:
    blocks = []

    for idx, doc in enumerate(docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown Source")
        section = metadata.get("section_header", "Unknown Section")
        page = metadata.get("page_number", "Unknown Page")

        block = (
            f"WEAK DOC [{idx + 1}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Page: {page}\n"
            f"Text:\n{doc.get('text', '')}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def rewrite_query(state: GraphState):
    """
    CRAG-inspired fallback query rewrite.

    Important rule:
    - original query remains the main intent
    - rewritten query is only a better retrieval query
    - do not broaden too much or drift into a different question
    """
    original_query = state["original_query"]
    weak_docs = state.get("weak_signal_docs", [])
    retries = state.get("crag_retries", 0)

    print("\n[CRAG Rewrite] Rewriting retrieval query...")
    print(f"  -> Groq model: {GROQ_MODEL}")

    weak_context = _build_weak_context(weak_docs) if weak_docs else "No weak documents available."
    prompt = f"""You are rewriting a user question into a better retrieval query for academic document search.

Original user question:
{original_query}

Weak or partial retrieved evidence:
---
{weak_context}
---

Instructions:
- Rewrite ONLY for retrieval quality.
- Preserve the original user intent exactly.
- Make the query more specific, not broader.
- Prefer short keyword-rich phrasing useful for hybrid search.
- Include important entities, task names, mechanism names, dataset names, or paper terminology if helpful.
- Remove filler words and conversational phrasing.
- Do NOT answer the question.
- Do NOT add commentary.
- Output only one short retrieval query.
- Keep it under 15 words if possible.

Examples:
User question: Why does Norway not fit neatly into the old three-circle English model anymore?
Better retrieval query: Norway English expanding circle fuzzy status domain use higher education business

User question: What does the tabular attention model use to pick features?
Better retrieval query: TabNet sequential attention instance-wise feature selection decision step

User question: Why does the ResNet paper say deeper nets can get worse even when overfitting is not the main issue?
Better retrieval query: ResNet degradation problem deeper plain networks higher training error not overfitting

Now output the rewritten retrieval query only.
"""
#     prompt = f"""You are rewriting a user question into a better retrieval query for academic document search.

# Original user question:
# {original_query}

# Weak or partial retrieved evidence:
# ---
# {weak_context}
# ---

# Instructions:
# - Rewrite ONLY for retrieval quality.
# - Preserve the original user intent exactly.
# - Make the query more specific, not broader.
# - Prefer key entities, technical terms, task names, mechanism names, paper terminology, or dataset names if helpful.
# - Do NOT answer the question.
# - Do NOT add commentary.
# - Output only the rewritten retrieval query on one line.
# """

    response = llm.invoke([HumanMessage(content=prompt)])
    # rewritten_query = response.content.strip()
    rewritten_query = " ".join(rewritten_query.split())
    if not rewritten_query:
        rewritten_query = original_query

    print(f"[CRAG Rewrite] Original query:  {original_query}")
    print(f"[CRAG Rewrite] Rewritten query: {rewritten_query}")

    return {
        "search_query": rewritten_query,
        "crag_retries": retries + 1,
    }