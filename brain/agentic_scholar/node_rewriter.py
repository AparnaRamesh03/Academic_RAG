from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL
from state_agentic import AgenticGraphState

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


def rewrite_query(state: AgenticGraphState):
    original_query = state["original_query"]
    weak_docs = state.get("weak_signal_docs", [])
    retries = state.get("crag_retries", 0)

    print("\n[Agentic Scholar] Rewriting retrieval query...")
    print(f"  -> Groq model: {GROQ_MODEL}")

    weak_context = (
        _build_weak_context(weak_docs)
        if weak_docs
        else "No weak documents available."
    )

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

Now output the rewritten retrieval query only.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten_query = " ".join(response.content.strip().split())

    if not rewritten_query:
        rewritten_query = original_query

    print(f"[Agentic Scholar] Original query:  {original_query}")
    print(f"[Agentic Scholar] Rewritten query: {rewritten_query}")

    return {
        "search_query": rewritten_query,
        "crag_retries": retries + 1,
    }
