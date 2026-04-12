from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from state import GraphState

# Standard generation temperature for factual accuracy
llm = ChatOllama(model="qwen2.5:14b", temperature=0)

def generate(state: GraphState):
    """
    Node 4: Generator
    Uses verified graded_docs to generate the final answer with strict inline citations.
    """
    query = state["original_query"]
    graded_docs = state["graded_docs"]
    auditor_feedback = state.get("auditor_feedback", "")
    
    print(f"\n[Node 4] Generating answer using {len(graded_docs)} verified sources...")
    
    # Compile context with explicit numbering and source file mappings
    context_blocks = []
    for idx, doc in enumerate(graded_docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("source_file", "Unknown Source")
        page = metadata.get("page_number", "Unknown Page")
        
        # Include [Source: X, Page: Y] identifier in the context block
        block = f"DOCUMENT [{idx+1}]:\nSource: {source}\nPage: {page}\nText: {doc['text']}"
        context_blocks.append(block)
        
    context = "\n\n".join(context_blocks)
    
    # Inject auditor feedback if this is a regeneration attempt
    if auditor_feedback:
        feedback_chunk = f"\n\nWARNING: Your previous draft failed the factual audit. The auditor stated:\n{auditor_feedback}\nYou MUST revise your answer to completely omit or correct the unsupported claims."
    else:
        feedback_chunk = ""
    
    prompt = f"""You are an expert academic assistant.
Your task is to answer the user's question using ONLY the provided verified documents.{feedback_chunk}

Verified Documents:
---
{context}
---

User Question: {query}

CRITICAL RULES:
1. You MUST NOT use any outside knowledge or hallucinate information. If the answer is not in the documents, state that you do not know.
2. Every time you make a claim, you MUST include an inline citation in exactly this format: [Source: source_name, Page: page_number].
3. For example: "The sky is blue [Source: my_paper.pdf, Page: 4]."
4. Never mention "Document [1]" or "Document 1" in your response. Always use the explicit Source and Page format.
5. Format your response cleanly using Markdown.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()
    
    print("[Node 4] Answer generation complete.")
    
    return {"generation": answer}
