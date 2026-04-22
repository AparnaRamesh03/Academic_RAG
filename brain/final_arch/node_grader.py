from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage

from llm_config import build_groq_llm, GROQ_MODEL
from state_shared import GraphState

llm = build_groq_llm(temperature=0.0)


def grade_documents(state: GraphState):
    """
    Self-RAG-inspired relevance grader over candidate_docs.
    """
    query = state["original_query"]
    candidate_docs = state.get("candidate_docs", [])

    print(f"\n[Final Combined] Grading {len(candidate_docs)} candidate docs...")
    print(f"  -> Groq model: {GROQ_MODEL}")

    if not candidate_docs:
        return {"graded_docs": []}

    graded_docs = []

    for idx, doc in enumerate(candidate_docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown Source")
        section = metadata.get("section_header", "Unknown Section")
        page = metadata.get("page_number", "Unknown Page")

        prompt = f"""You are judging whether a retrieved academic document chunk is useful for answering a user question.

User question:
{query}

Retrieved chunk:
Source: {source}
Section: {section}
Page: {page}
Text:
---
{doc.get('text', '')}
---

Instructions:
- Answer YES if this chunk is clearly relevant or likely useful.
- Answer NO if this chunk is clearly irrelevant or too generic.
- Be slightly recall-friendly: if potentially useful, prefer YES.
- Output only one word: YES or NO.
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        decision = re.sub(r"[^A-Z]", "", response.content.strip().upper())

        if decision == "YES":
            graded_docs.append(doc)
            print(f"  -> Doc {idx + 1}: RELEVANT")
        else:
            print(f"  -> Doc {idx + 1}: IRRELEVANT")

    print(f"[Final Combined] Kept {len(graded_docs)} / {len(candidate_docs)} docs.")

    if not graded_docs and candidate_docs:
        print("[Final Combined] Fallback: grader rejected all docs, keeping original candidates.")
        graded_docs = candidate_docs

    return {"graded_docs": graded_docs}