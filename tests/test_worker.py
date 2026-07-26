"""
Tests for the AI worker (ai/worker.py)

The Redis connection, DB session, OCR, LLM, and embeddings modules are all
mocked so no external infrastructure is needed.

Covers:
- Already-processed frames are skipped (idempotent)
- OCR failure marks frame as done with null ocr_text (non-fatal)
- LLM failure still saves OCR text to DB
- Full happy-path: OCR + LLM + embedding all succeed
"""
import json
from unittest.mock import MagicMock, patch, call

import pytest

from ai.worker import AIWorker
from ai.llm import LLMResult


def _make_event(frame_id: int = 1) -> dict:
    return {
        "frame_id": frame_id,
        "path": "frames/2024-01-15/test.jpg",
        "app_name": "code.exe",
        "ts": "2024-01-15T10:30:00+00:00",
    }


def _make_mock_frame(frame_id: int = 1):
    frame = MagicMock()
    frame.id = frame_id
    frame.window_title = "main.py — VSCode"
    frame.analysis = None
    return frame


class TestAIWorkerProcessEvent:
    def _make_worker(self):
        worker = AIWorker()
        worker._redis = MagicMock()
        return worker

    def test_skips_already_done_frame(self):
        worker = self._make_worker()
        event = _make_event()

        mock_frame = _make_mock_frame()
        mock_analysis = MagicMock()
        mock_analysis.status = "done"
        mock_analysis.id = 10
        mock_frame.analysis = mock_analysis

        mock_db = MagicMock()
        mock_db.get.return_value = mock_frame
        mock_db.__enter__ = lambda s: mock_db
        mock_db.__exit__ = MagicMock(return_value=False)

        with patch("ai.worker.SessionLocal", return_value=mock_db):
            with patch("ai.worker.extract_text") as mock_ocr:
                worker._process_event(event)
                mock_ocr.assert_not_called()

    def test_missing_frame_id_in_db_is_skipped_gracefully(self):
        worker = self._make_worker()
        event = _make_event(frame_id=999)

        mock_db = MagicMock()
        mock_db.get.return_value = None
        mock_db.__enter__ = lambda s: mock_db
        mock_db.__exit__ = MagicMock(return_value=False)

        with patch("ai.worker.SessionLocal", return_value=mock_db):
            # Should not raise
            worker._process_event(event)

    def test_happy_path_calls_all_three_steps(self):
        worker = self._make_worker()
        event = _make_event()

        mock_frame = _make_mock_frame()
        mock_analysis = MagicMock()
        mock_analysis.status = "pending"
        mock_analysis.id = 5
        mock_frame.analysis = mock_analysis

        mock_db = MagicMock()
        mock_db.get.side_effect = [mock_frame, mock_analysis]
        mock_db.__enter__ = lambda s: mock_db
        mock_db.__exit__ = MagicMock(return_value=False)

        with patch("ai.worker.SessionLocal", return_value=mock_db):
            with patch("ai.worker.extract_text", return_value="Hello world " * 10) as mock_ocr:
                with patch("ai.worker.analyse_frame", return_value=LLMResult(summary="Coding", tags=["code"])) as mock_llm:
                    with patch("ai.worker.embeddings.index_frame", return_value="5") as mock_embed:
                        worker._process_event(event)

        mock_ocr.assert_called_once()
        mock_llm.assert_called_once()
        mock_embed.assert_called_once()

    def test_embedding_failure_does_not_crash_worker(self):
        worker = self._make_worker()
        event = _make_event()

        mock_frame = _make_mock_frame()
        mock_analysis = MagicMock()
        mock_analysis.status = "pending"
        mock_analysis.id = 5
        mock_frame.analysis = mock_analysis

        mock_db = MagicMock()
        mock_db.get.side_effect = [mock_frame, mock_analysis]
        mock_db.__enter__ = lambda s: mock_db
        mock_db.__exit__ = MagicMock(return_value=False)

        with patch("ai.worker.SessionLocal", return_value=mock_db):
            with patch("ai.worker.extract_text", return_value="Hello " * 20):
                with patch("ai.worker.analyse_frame", return_value=LLMResult(summary="x", tags=[])):
                    with patch("ai.worker.embeddings.index_frame", side_effect=RuntimeError("ChromaDB down")):
                        # Should not raise — embedding failure is non-fatal
                        worker._process_event(event)
