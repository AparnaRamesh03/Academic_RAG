import re
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from state import GraphState

llm = ChatOllama(model="qwen2.5:14b", temperature=0)

def grade_documents(state: GraphState):
    """
    Node 2: Relevance Grader (Self-RAG)
    Uses LLM to filter candidate_docs, keeping only those actually relevant to the query.
    """
    query = state["original_query"]
    candidate_docs = state["candidate_docs"]
    
    print(f"\n[Node 2] Grading {len(candidate_docs)} candidates for relevance...")
    
    graded_docs = []
    
    for idx, doc in enumerate(candidate_docs):
        prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.
Here is the retrieved document:
---
{doc['text']}
---
Here is the user question: {query}

If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Provide ONLY the word 'yes' or 'no'.
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        score = response.content.strip().lower()
        score = re.sub(r'[^a-zA-Z]', '', score)
        
        if score == "yes":
            graded_docs.append(doc)
            print(f"  -> Doc {idx+1}: RELEVANT")
        else:
            print(f"  -> Doc {idx+1}: IRRELEVANT")
            
    print(f"[Node 2] {len(graded_docs)} out of {len(candidate_docs)} docs passed the grader.")
            
    return {"graded_docs": graded_docs}
