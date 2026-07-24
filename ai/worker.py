"""
AI Worker — consumes frame events from Redis and runs the full AI pipeline.

Pipeline per frame:
  1. Pop event from Redis list "frame_queue" (BRPOP — blocking, no busy-wait)
  2. Mark FrameAnalysis record as "processing"
  3. Run Tesseract OCR on the saved JPEG
  4. Call OpenAI LLM to generate a summary and semantic tags
  5. Embed summary in ChromaDB (sentence-transformers, local inference)
  6. Mark FrameAnalysis record as "done"

On any step failure the record is marked "failed" with an error message so
the operator can inspect and replay if needed.

Running:
    python -m ai.worker
    # or, after pip install -e .
    riom-worker
"""
from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

import redis
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError

from config import settings
from storage.database import SessionLocal, engine
from storage.models import Base, Frame, FrameAnalysis
from ai.ocr import extract_text
from ai.llm import analyse_frame
from ai import embeddings


_QUEUE_KEY = "frame_queue"
_BRPOP_TIMEOUT = 2  # seconds to block before re-checking the shutdown flag


class AIWorker:
    """
    Long-running process that consumes frame events from Redis and
    orchestrates OCR → LLM → embedding for each frame.
    """

    def __init__(self) -> None:
        self._stop = False

    def startup(self) -> None:
        """Create DB tables if needed and connect to Redis."""
        Base.metadata.create_all(bind=engine)
        self._redis = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        self._redis.ping()
        logger.info("AI Worker connected to Redis.")

    def _handle_signal(self, signum, _frame) -> None:
        logger.info(f"Signal {signum} received — shutting down worker.")
        self._stop = True

    def run(self) -> None:
        """Main loop — blocks on BRPOP until a frame event arrives."""
        self.startup()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(f"AI Worker listening on Redis list '{_QUEUE_KEY}'...")

        while not self._stop:
            result = self._redis.brpop(_QUEUE_KEY, timeout=_BRPOP_TIMEOUT)
            if result is None:
                continue  # Timeout — loop and check self._stop

            _, raw_event = result
            try:
                event = json.loads(raw_event)
            except json.JSONDecodeError:
                logger.error(f"Malformed event on queue: {raw_event!r}")
                continue

            self._process_event(event)

        logger.info("AI Worker stopped.")

    def _process_event(self, event: dict) -> None:
        """
        Run the full OCR → LLM → embedding pipeline for one frame.

        Errors are caught per-step; the frame is marked "failed" and processing
        continues with the next event rather than crashing the worker.
        """
        frame_id = event.get("frame_id")
        relative_path = event.get("path", "")
        app_name = event.get("app_name", "Unknown")
        ts_iso = event.get("ts", "")

        logger.info(f"Processing frame {frame_id} ({app_name})")

        absolute_path = Path(settings.DATA_DIR) / relative_path

        with SessionLocal() as db:
            # Ensure the Frame row exists
            frame: Frame | None = db.get(Frame, frame_id)
            if frame is None:
                logger.warning(f"Frame {frame_id} not found in DB — skipping.")
                return

            window_title = frame.window_title

            # Create or fetch FrameAnalysis row
            analysis: FrameAnalysis = frame.analysis or FrameAnalysis(frame_id=frame_id)
            if analysis.status == "done":
                logger.debug(f"Frame {frame_id} already processed — skipping.")
                return

            analysis.status = "processing"
            db.add(analysis)
            try:
                db.commit()
            except SQLAlchemyError as exc:
                logger.error(f"Could not mark frame {frame_id} as processing: {exc}")
                return

        # ── Step 1: OCR ────────────────────────────────────────────────────────
        ocr_text = extract_text(absolute_path)
        logger.debug(
            f"Frame {frame_id} OCR: {len(ocr_text or '')} chars extracted."
        )

        # ── Step 2: LLM ────────────────────────────────────────────────────────
        llm_result = analyse_frame(
            ocr_text=ocr_text,
            app_name=app_name,
            window_title=window_title,
        )

        # ── Step 3: Embed ──────────────────────────────────────────────────────
        chroma_doc_id: str | None = None
        if llm_result and llm_result.summary:
            try:
                chroma_doc_id = embeddings.index_frame(
                    frame_id=frame_id,
                    summary=llm_result.summary,
                    tags=llm_result.tags,
                    app_name=app_name,
                    ts_iso=ts_iso,
                )
            except Exception as exc:
                logger.error(f"Embedding failed for frame {frame_id}: {exc}")
                # Non-fatal — we still save OCR and summary to DB

        # ── Step 4: Persist results ────────────────────────────────────────────
        with SessionLocal() as db:
            analysis = (
                db.query(FrameAnalysis)
                .filter(FrameAnalysis.frame_id == frame_id)
                .first()
            ) or FrameAnalysis(frame_id=frame_id)
            analysis.ocr_text = ocr_text
            analysis.summary = llm_result.summary if llm_result else None
            analysis.tags = ", ".join(llm_result.tags) if llm_result else None
            analysis.chroma_doc_id = chroma_doc_id
            analysis.status = "done"
            analysis.error = None
            db.add(analysis)
            try:
                db.commit()
                logger.info(
                    f"Frame {frame_id} processed — "
                    f"summary={'yes' if analysis.summary else 'no'}, "
                    f"embedded={'yes' if chroma_doc_id else 'no'}."
                )
            except SQLAlchemyError as exc:
                logger.error(f"Failed to persist analysis for frame {frame_id}: {exc}")
                db.rollback()
                self._mark_failed(frame_id, str(exc))

    def _mark_failed(self, frame_id: int, error: str) -> None:
        """Mark a FrameAnalysis row as failed with an error message."""
        with SessionLocal() as db:
            try:
                analysis = (
                    db.query(FrameAnalysis)
                    .filter(FrameAnalysis.frame_id == frame_id)
                    .first()
                )
                if analysis:
                    analysis.status = "failed"
                    analysis.error = error[:1024]
                    db.commit()
            except SQLAlchemyError as exc:
                logger.error(f"Could not mark frame {frame_id} as failed: {exc}")


def main() -> None:
    worker = AIWorker()
    worker.run()


if __name__ == "__main__":
    main()
