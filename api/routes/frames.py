"""
Frames route — browse captured frames and their AI analyses.

GET /frames          List recent frames (paginated)
GET /frames/{id}     Get a single frame with full analysis
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from storage.database import SessionLocal
from storage.models import Frame

router = APIRouter(prefix="/frames", tags=["frames"])


# ── Dependency ─────────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Response schemas ──────────────────────────────────────────────────────────

class AnalysisSchema(BaseModel):
    ocr_text: Optional[str]
    summary: Optional[str]
    tags: Optional[str]
    status: str

    model_config = {"from_attributes": True}


class FrameSchema(BaseModel):
    id: int
    ts: datetime
    monitor_id: int
    app_name: str
    window_title: str
    path: str
    phash: str
    analysis: Optional[AnalysisSchema] = None

    model_config = {"from_attributes": True}


class FrameListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    frames: list[FrameSchema]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=FrameListResponse, summary="List recent frames")
def list_frames(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    app: Optional[str] = Query(default=None, description="Filter by application name"),
    db: Session = Depends(get_db),
) -> FrameListResponse:
    """
    Return a paginated list of captured frames, newest first.

    Optionally filter by application name (exact match, case-sensitive).
    """
    # Use SQLAlchemy 2.0 Core select() for forward-compatibility
    stmt = select(Frame)
    if app:
        stmt = stmt.where(Frame.app_name == app)

    total: int = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()

    frames = db.execute(
        stmt.order_by(Frame.ts.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).scalars().all()

    return FrameListResponse(
        total=total,
        page=page,
        page_size=page_size,
        frames=list(frames),
    )


@router.get("/{frame_id}", response_model=FrameSchema, summary="Get a single frame")
def get_frame(frame_id: int, db: Session = Depends(get_db)) -> FrameSchema:
    """
    Retrieve a single frame by ID, including its AI-generated analysis.
    """
    frame = db.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"Frame {frame_id} not found.")
    return frame
