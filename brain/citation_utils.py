from typing import List, Dict, Any
from api_models import CitationItem


def build_contexts_from_docs(docs: List[Dict[str, Any]], prefix: str = "[RETRIEVED]") -> List[str]:
    """
    Build context strings for API responses or saved results.
    """
    return [f"{prefix} {doc['text']}" for doc in docs]


def build_citations_from_docs(docs: List[Dict[str, Any]]) -> List[CitationItem]:
    """
    Build citation metadata objects from retrieved/final docs.
    """
    citations = []

    for doc in docs:
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

    return citations