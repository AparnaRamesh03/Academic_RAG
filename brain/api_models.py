from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


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