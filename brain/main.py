from typing import List, Optional
from fastapi import FastAPI, HTTPException
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
    return {
        "message": "Brain API is running.",
        "endpoints": {
            "ask": "POST /ask",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "Academic RAG Brain API"
    }

class QueryRequest(BaseModel):
    query: str
    ablation_mode: str = "full"

class CitationItem(BaseModel):
    source_file: str
    page_number: Optional[int | str] = None
    section_header: Optional[str] = None
    excerpt: str
    content_type: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    context_used: List[str]
    citations: List[CitationItem]

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

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

    try:
        result = app_graph.invoke(initial_state)
        graded_docs = result.get("graded_docs", [])

        contexts = [f"[RELEVANT] {doc['text']}" for doc in graded_docs]

        citations = []
        for doc in graded_docs:
            metadata = doc.get("metadata", {})
            citations.append(
                CitationItem(
                    source_file=str(metadata.get("source_file", "Unknown Source")),
                    page_number=metadata.get("page_number", "Unknown Page"),
                    section_header=str(metadata.get("section_header", "Unknown Section")),
                    excerpt=str(doc.get("text", "")),
                    content_type=str(metadata.get("content_type", "text")),
                )
            )

        return QueryResponse(
            answer=result.get("generation", "Error: No answer generated."),
            context_used=contexts,
            citations=citations
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Brain processing failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)