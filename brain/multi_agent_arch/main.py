from pathlib import Path
import sys
import os
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
BRAIN_DIR = CURRENT_DIR.parent

for path in [str(CURRENT_DIR), str(BRAIN_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

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

app = FastAPI(title="Academic RAG Brain API - Multi-Agent Architecture")

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


def _build_initial_state(query: str) -> Dict[str, Any]:
    return {
        "original_query": query,
        "search_query": query,
        "retrieved_docs": [],
        "candidate_docs": [],
        "weak_signal_docs": [],
        "graded_docs": [],
        "generation": "",
        "crag_retries": 0,
        "verify_retries": 0,
        "citations_pass": False,
        "auditor_feedback": "",
        "claim_verification": [],
        "step_count": 0,
        "action_history": [],
        "last_action": "",
        "done": False,
        "stop_reason": "",
        "confidence": 0.0,
        "latency_so_far": 0.0,
        "agent_status": {},
        "agent_trace": [],
        "verification_outcome": "",
        "mixed_domain_evidence": False,
        "evidence_source_distribution": {},
    }


def _serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    metadata = doc.get("metadata", {}) or {}
    return {
        "text": doc.get("text", ""),
        "score": doc.get("score"),
        "rerank_score": doc.get("rerank_score"),
        "metadata": {
            "source_file": metadata.get("source_file"),
            "page_number": metadata.get("page_number"),
            "section_header": metadata.get("section_header"),
            "content_type": metadata.get("content_type"),
            "chunk_index": metadata.get("chunk_index"),
            "has_table": metadata.get("has_table"),
            "has_image_description": metadata.get("has_image_description"),
            "continued_from_previous_page": metadata.get("continued_from_previous_page"),
            "previous_page_number": metadata.get("previous_page_number"),
            "figure_number": metadata.get("figure_number"),
            "figure_caption": metadata.get("figure_caption"),
        },
    }


def _serialize_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_serialize_doc(doc) for doc in docs]


@app.get("/")
def root():
    return {
        "message": "Multi-agent brain API is running.",
        "architecture": ARCHITECTURE_NAME,
        "model_provider": "Groq",
        "model": GROQ_MODEL,
        "collection": COLLECTION_NAME,
        "endpoints": {
            "ask": "POST /ask",
            "ask_debug": "POST /ask_debug",
            "health": "GET /health",
            "dependencies_health": "GET /health/dependencies",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Academic RAG Multi-Agent API",
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
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    initial_state = _build_initial_state(request.query)

    try:
        result = app_graph.invoke(initial_state)
        final_docs = result.get("graded_docs", [])

        contexts = build_contexts_from_docs(final_docs, prefix="[FINAL]")
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
                "message": "Multi-agent brain processing failed.",
                "architecture": ARCHITECTURE_NAME,
                "error": str(e),
            },
        )


@app.post("/ask_debug")
def ask_question_debug(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    initial_state = _build_initial_state(request.query)

    try:
        result = app_graph.invoke(initial_state)
        final_docs = result.get("graded_docs", [])

        contexts = build_contexts_from_docs(final_docs, prefix="[FINAL]")
        citations = build_citations_from_docs(final_docs)

        return {
            "architecture": ARCHITECTURE_NAME,
            "query": request.query,
            "original_query": result.get("original_query", request.query),
            "final_search_query": result.get("search_query", request.query),
            "answer": result.get("generation", "Error: No answer generated."),
            "context_used": contexts,
            "citations": [c.model_dump() for c in citations],
            "retrieved_docs": _serialize_docs(result.get("retrieved_docs", [])),
            "candidate_docs": _serialize_docs(result.get("candidate_docs", [])),
            "weak_signal_docs": _serialize_docs(result.get("weak_signal_docs", [])),
            "graded_docs": _serialize_docs(final_docs),
            "crag_retries": result.get("crag_retries", 0),
            "verify_retries": result.get("verify_retries", 0),
            "citations_pass": result.get("citations_pass", False),
            "auditor_feedback": result.get("auditor_feedback", ""),
            "claim_verification": result.get("claim_verification", []),
            "step_count": result.get("step_count", 0),
            "action_history": result.get("action_history", []),
            "last_action": result.get("last_action", ""),
            "done": result.get("done", False),
            "stop_reason": result.get("stop_reason", ""),
            "confidence": result.get("confidence", 0.0),
            "latency_so_far": result.get("latency_so_far", 0.0),
            "agent_status": result.get("agent_status", {}),
            "agent_trace": result.get("agent_trace", []),
            "verification_outcome": result.get("verification_outcome", ""),
            "mixed_domain_evidence": result.get("mixed_domain_evidence", False),
            "evidence_source_distribution": result.get("evidence_source_distribution", {}),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Multi-agent debug processing failed.",
                "architecture": ARCHITECTURE_NAME,
                "error": str(e),
            },
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)