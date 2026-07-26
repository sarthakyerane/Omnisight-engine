"""
Semantic query route — the core "recall" endpoint for RIOM.

GET /query?q=<natural language question>&top_k=10&app=chrome.exe

Example queries:
  - "What Python file was I editing this morning?"
  - "Show me the Stripe dashboard I had open"
  - "What error message appeared in my terminal?"
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ai import embeddings
from config import settings

router = APIRouter(prefix="/query", tags=["query"])


class QueryResult(BaseModel):
    frame_id: int
    distance: float
    summary: str
    app_name: str
    ts: str
    tags: str


class QueryResponse(BaseModel):
    query: str
    total_results: int
    results: list[QueryResult]


@router.get("", response_model=QueryResponse, summary="Semantic recall query")
def semantic_query(
    q: str = Query(..., min_length=2, max_length=512, description="Natural-language search query"),
    top_k: int = Query(default=None, ge=1, le=100, description="Maximum results to return"),
    app: str | None = Query(default=None, description="Filter by application name"),
) -> QueryResponse:
    """
    Search your screen history using natural language.

    Results are ordered by semantic similarity (most relevant first).
    Each result includes the frame ID, timestamp, app name, tags, and a
    1-3 sentence summary of what was on screen.

    To retrieve the full screenshot, use `GET /frames/{frame_id}`.
    """
    try:
        raw_results = embeddings.query(
            query_text=q,
            top_k=top_k or settings.QUERY_TOP_K,
            app_filter=app,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    results = [
        QueryResult(
            frame_id=r["frame_id"],
            distance=r["distance"],
            summary=r["summary"],
            app_name=r["app_name"],
            ts=r["ts"],
            tags=r["tags"],
        )
        for r in raw_results
    ]

    return QueryResponse(query=q, total_results=len(results), results=results)
