from pathlib import Path
import sys
import os

# Allow this folder to import shared files from ../
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from api_models import QueryRequest, QueryResponse
from citation_utils import build_contexts_from_docs, build_citations_from_docs
from graph import build_graph
from config import ARCHITECTURE_NAME
from qdrant_config import QDRANT_URL, COLLECTION_NAME
from llm_config import GROQ_MODEL

load_dotenv()

app_graph = build_graph()
qdrant_client = QdrantClient(url=QDRANT_URL)

app = FastAPI(title="Academic RAG Brain API - CRAG Rewrite")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Brain API is running.",
        "architecture": ARCHITECTURE_NAME,
        "model_provider": "Groq",
        "model": GROQ_MODEL,
        "collection": COLLECTION_NAME,
        "endpoints": {
            "ask": "POST /ask",
            "health": "GET /health",
            "dependencies_health": "GET /health/dependencies",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Academic RAG Brain API",
        "architecture": ARCHITECTURE_NAME,
    }


@app.get("/health/dependencies")
def dependencies_health():
    qdrant_status = "ok"
    qdrant_error = None

    try:
        qdrant_client.collection_exists(COLLECTION_NAME)
    except Exception as e:
        qdrant_status = "error"
        qdrant_error = str(e)

    groq_api_key_present = bool(os.getenv("GROQ_API_KEY"))

    return {
        "architecture": ARCHITECTURE_NAME,
        "qdrant": {
            "status": qdrant_status,
            "url": QDRANT_URL,
            "collection": COLLECTION_NAME,
            "error": qdrant_error,
        },
        "groq": {
            "status": "ok" if groq_api_key_present else "missing_api_key",
            "model": GROQ_MODEL,
        },
    }


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Sync route on purpose:
    graph invocation and retrieval/generation work are blocking.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    initial_state = {
        "original_query": request.query,
        "search_query": request.query,

        "retrieved_docs": [],
        "candidate_docs": [],
        "weak_signal_docs": [],
        "graded_docs": [],

        "generation": "",
        "crag_retries": 0,
        "verify_retries": 0,
        "citations_pass": True,
        "auditor_feedback": "",
    }

    try:
        result = app_graph.invoke(initial_state)
        final_docs = result.get("graded_docs", [])

        contexts = build_contexts_from_docs(final_docs, prefix="[RETRIEVED]")
        citations = build_citations_from_docs(final_docs)

        return QueryResponse(
            answer=result.get("generation", "Error: No answer generated."),
            context_used=contexts,
            citations=citations,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Brain processing failed.",
                "architecture": ARCHITECTURE_NAME,
                "error": str(e),
            },
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)