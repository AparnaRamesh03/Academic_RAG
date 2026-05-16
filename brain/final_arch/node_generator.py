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
    Combined generator:
    - uses final graded docs
    - supports one audit-driven retry
    """
    query = state["original_query"]
    selected_docs = state.get("graded_docs", [])
    auditor_feedback = state.get("auditor_feedback", "").strip()
    verify_retries = state.get("verify_retries", 0)
    choices = state.get("choices", None)  # ARC Multiple Choice

    print(
        f"\n[Final Combined] Generating answer using "
        f"{len(selected_docs)} graded docs..."
    )
    print(f"  -> Groq model: {GROQ_MODEL}")
    print(f"  -> Audit retry count: {verify_retries}")
    if choices:
        print(f"  -> MCQ Mode: {len(choices.get('label', []))} options.")

    if not selected_docs:
        return {
            "generation": (
                "Insufficient information."
            )
        }

    context = _build_context_blocks(selected_docs)

    # Handle Multiple Choice formatting
    choices_block = ""
    if choices and "text" in choices and "label" in choices:
        choices_block = "\nMultiple Choice Options:\n"
        for label, text in zip(choices["label"], choices["text"]):
            choices_block += f"{label}: {text}\n"

    feedback_block = ""
    if auditor_feedback:
        feedback_block = f"""
Auditor feedback from the previous answer:
{auditor_feedback}

Important retry instruction:
- Rewrite the answer to directly answer the user question.
- Remove unsupported or weakly grounded claims.
- Remove broad background statements unless they are necessary for the answer.
- Prefer exact, question-specific claims over general explanation.
- Keep only what is clearly supported by the retrieved evidence.
"""

    # Selection rules for MCQ vs Free-Text
    mcq_rule = ""
    if choices:
        mcq_rule = """
14. MULTIPLE CHOICE RULE: This is a multiple-choice question. You MUST select exactly one label from the provided options (e.g., A, B, C, or D). Your entire response MUST be ONLY the single letter of the correct choice.
"""

    prompt = f"""You are an expert academic question-answering assistant.

Your task is to answer the user's question using ONLY the retrieved documents below.

Retrieved Documents:
---
{context}
---

User Question:
{query}
{choices_block}
{feedback_block}

Rules:
1. Use ONLY the retrieved documents. Do not use outside knowledge.
2. You MAY synthesize across multiple retrieved documents when the answer is directly supported by combining them.
3. Prefer the evidence that is MOST SPECIFIC to the user's question.
4. Start with the direct answer immediately. Do NOT add headings like "Answer", "Clarification", or similar meta commentary.
5. Keep the answer extremely concise and evaluation-friendly.
6. ABSOLUTE RULE: If the question is a Yes/No question, your entire response MUST be ONLY "Yes" or "No". Do not explain.
7. If the question asks for a specific name, date, or fact, provide ONLY that specific information.
8. Avoid any conversational filler, introductory phrases, or extra context.
9. If the provided documents do not contain the answer, your entire response MUST be "Insufficient information."
10. If part of a broader answer is not clearly supported, omit it.
11. Do NOT include inline citations or bracket citations in the answer text.
12. Write clean natural prose ONLY for long-form questions. For factual, multi-hop, or yes/no questions, use the minimal number of words possible.
13. STRATEGIC RULE: Your goal is to maximize Token F1 overlap with the ground truth. This means providing ONLY the specific entity name, number, or "Yes"/"No".
{mcq_rule}

### Examples of Expected Output:
Question: Was the CEO of Company X fired in 2023?
Answer: Yes

Question: What is the capital of France?
Answer: Paris

Question: Which company acquired DeepMind?
Answer: Google

Question: Based on the articles, did the stock price increase? (No info in docs)
Answer: Insufficient information.

Now answer the question.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    print("[Final Combined] Answer generation complete.")

    return {"generation": answer}