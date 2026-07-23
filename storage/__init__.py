"""
Storage module — SQLAlchemy engine, session factory, and ORM models.

Exports:
    engine       : SQLAlchemy Engine (SQLite WAL mode enabled by default)
    SessionLocal : Session factory (use as a context manager)
    Base         : Declarative base for all ORM models
    Frame        : Captured screenshot frame
    FrameAnalysis: AI-generated analysis (OCR + summary + embedding)
"""
from storage.database import engine, SessionLocal
from storage.models import Base, Frame, FrameAnalysis

__all__ = ["engine", "SessionLocal", "Base", "Frame", "FrameAnalysis"]
