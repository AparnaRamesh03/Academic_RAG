from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from state import GraphState

llm = ChatOllama(model="qwen2.5:14b", temperature=0)

def verify_citations(state: GraphState):
    """
    Node 5: VeriCite Auditor
    Checks if the generation is fully entailed by the graded_docs.
    """
    generation = state.get("generation", "")
    graded_docs = state["graded_docs"]
    verify_retries = state.get("verify_retries", 0)
    
    print(f"\n[Node 5] VeriCite Auditor running (Attempt {verify_retries + 1}/3)...")
    
    context_blocks = []
    for doc in graded_docs:
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown")
        page = metadata.get("page_number", "Unknown")
        context_blocks.append(f"Source: {source} (Page {page})\nText: {doc['text']}")
        
    context = "\n\n".join(context_blocks)
    
    prompt = f"""You are a strict academic fact-checker.
Compare the following GENERATED ANSWER against the provided SOURCE DOCUMENTS.

SOURCE DOCUMENTS:
---
{context}
---

GENERATED ANSWER:
---
{generation}
---

Your only job is to determine if the GENERATED ANSWER contains any claims, facts, or numbers that are NOT explicitly stated in the SOURCE DOCUMENTS.
(Note: You must ignore [Source: X, Page: Y] citation brackets in the text. You are only auditing the factual claims.)

If every claim is supported, respond strictly with 'PASS'.
If there is ANY hallucinated or unsupported claim, respond strictly with 'FAIL' followed by a brief explanation of what claim failed.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    
    if content.upper().startswith("PASS"):
        print("  -> Verdict: PASS (No hallucinations detected).")
        return {
            "citations_pass": True,
            "auditor_feedback": ""
        }
    else:
        print("  -> Verdict: FAIL (Hallucination detected).")
        print(f"  -> Auditor Note: {content}")
        return {
            "citations_pass": False,
            "auditor_feedback": content,
            "verify_retries": verify_retries + 1
        }
