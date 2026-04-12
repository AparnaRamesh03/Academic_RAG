from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from state import GraphState

# Elevated temperature for slightly more creative rewriting
llm = ChatOllama(model="qwen2.5:14b", temperature=0.2)

def crag_rewrite(state: GraphState):
    """
    Node 3: Smart Query Rewriter (CRAG)
    Uses the weak signals to extract jargon or context to rewrite the user's query.
    """
    search_query = state["search_query"]
    original_query = state["original_query"]
    weak_signal_docs = state["weak_signal_docs"]
    crag_retries = state.get("crag_retries", 0)
    
    print(f"\n[Node 3] CRAG Rewrite triggered (Retry {crag_retries + 1}/3).")
    print("  -> Evaluating weak signals to improve search query...")
    
    # Compile weak signals into context
    context = "\n\n".join([doc["text"] for doc in weak_signal_docs])
    
    prompt = f"""You are an expert academic research assistant.
The user asked the following original question: "{original_query}"

Our previous search attempt using the query "{search_query}" failed to find highly relevant documents.
However, we did find some loosely related documents. You must extract any useful keywords, jargon, or context from these loosely related documents to rewrite the search query so it is highly specific and likely to succeed.

Loosely Related Documents (Weak Signals):
---
{context}
---

Provide ONLY the rewritten query text. Do not include quotes, explanations, or introductory text.
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    new_query = response.content.strip()
    
    print(f"  -> Original User Query: {original_query}")
    print(f"  -> Old Search Query: {search_query}")
    print(f"  -> New Search Query: {new_query}")
    
    return {
        "search_query": new_query,
        "crag_retries": crag_retries + 1
    }
