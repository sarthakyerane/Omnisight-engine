"""
SQLAlchemy ORM models for RIOM.

Two tables:
  - frames       : every captured screen frame (written by the capture daemon)
  - frame_analyses: AI-generated OCR + summary + tags for each frame (written by the AI worker)
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp. Replaces deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc)


class Frame(Base):
    """One captured screenshot frame."""

    __tablename__ = "frames"

    id = Column(Integer, primary_key=True, index=True)

    # Timestamp set by the capture loop (not the DB insert time)
    ts = Column(DateTime(timezone=True), nullable=False, default=_utcnow, index=True)

    monitor_id = Column(Integer, nullable=False)
    app_name = Column(String(255), nullable=False, index=True)
    window_title = Column(String(1024), nullable=False)

    # Relative path from DATA_DIR so the DB is portable across machines
    path = Column(String(1024), nullable=False)

    phash = Column(String(64), nullable=False)

    # Inserted-at timestamp (DB-side bookkeeping only)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    # One-to-one relationship with the AI analysis
    analysis = relationship("FrameAnalysis", back_populates="frame", uselist=False)

    __table_args__ = (
        UniqueConstraint("path", name="uq_frame_path"),
        # Most common query pattern: "show me Chrome frames from today"
        Index("ix_frames_app_ts", "app_name", "ts"),
    )


class FrameAnalysis(Base):
    """AI-generated analysis for a single Frame (OCR text + LLM summary + tags)."""

    __tablename__ = "frame_analyses"

    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(
        Integer,
        ForeignKey("frames.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Raw OCR text extracted from the screenshot
    ocr_text = Column(Text, nullable=True)

    # LLM-generated natural-language summary (≤ 3 sentences)
    summary = Column(Text, nullable=True)

    # Comma-separated tags produced by the LLM (e.g. "coding,python,vscode")
    tags = Column(String(512), nullable=True)

    # ChromaDB document ID for the embedding of this frame
    chroma_doc_id = Column(String(128), nullable=True, unique=True)

    # Processing status: pending | processing | done | failed
    status = Column(String(32), nullable=False, default="pending", index=True)

    # Error message if status == "failed"
    error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    frame = relationship("Frame", back_populates="analysis")
