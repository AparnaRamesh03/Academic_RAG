from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from graph import build_graph

load_dotenv()

app_graph = build_graph()

app = FastAPI(title="Academic RAG Brain API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Brain API is running.", "endpoints": {"ask": "POST /ask"}}

class QueryRequest(BaseModel):
    query: str
    ablation_mode: str = "full"

class QueryResponse(BaseModel):
    answer: str
    context_used: List[str]

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Invoke the LangGraph brain to process a question."""
    initial_state = {
        "original_query": request.query, 
        "search_query": request.query, 
        "ablation_mode": request.ablation_mode,
        "candidate_docs": [], 
        "weak_signal_docs": [],
        "graded_docs": [],
        "generation": "",
        "crag_retries": 0,
        "verify_retries": 0,
        "citations_pass": True,
        "auditor_feedback": ""
    }
    
    result = app_graph.invoke(initial_state)
    
    contexts = [f"[RELEVANT] {doc['text']}" for doc in result.get("graded_docs", [])]
    
    # Send actual generation to UI
    return QueryResponse(
        answer=result.get("generation", "Error: No answer generated."),
        context_used=contexts
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
