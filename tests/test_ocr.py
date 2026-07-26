"""
Tests for ai/ocr.py

Tesseract is mocked to avoid requiring the binary in CI.

Covers:
- extract_text returns a string for a valid image
- extract_text returns None for whitespace-only OCR output
- extract_text returns None and logs error when Tesseract is not found
- extract_text returns None when image file does not exist
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import io

import pytesseract


class TestExtractText:
    def _make_image_bytes(self) -> bytes:
        """Create a small in-memory JPEG for testing."""
        img = Image.new("RGB", (100, 50), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_returns_text_from_valid_image(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(self._make_image_bytes())

        with patch("pytesseract.image_to_string", return_value="Hello World\n"):
            from ai.ocr import extract_text
            result = extract_text(img_path)

        assert result == "Hello World"

    def test_returns_none_for_whitespace_only_ocr(self, tmp_path):
        img_path = tmp_path / "blank.jpg"
        img_path.write_bytes(self._make_image_bytes())

        with patch("pytesseract.image_to_string", return_value="   \n\n  "):
            from ai.ocr import extract_text
            result = extract_text(img_path)

        assert result is None

    def test_returns_none_when_tesseract_not_found(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(self._make_image_bytes())

        with patch(
            "pytesseract.image_to_string",
            side_effect=pytesseract.TesseractNotFoundError(),
        ):
            from ai.ocr import extract_text
            result = extract_text(img_path)

        assert result is None

    def test_returns_none_when_file_not_found(self):
        from ai.ocr import extract_text
        result = extract_text(Path("/nonexistent/path/image.jpg"))
        assert result is None

    def test_returns_none_on_generic_exception(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(self._make_image_bytes())

        with patch("pytesseract.image_to_string", side_effect=RuntimeError("GPU OOM")):
            from ai.ocr import extract_text
            result = extract_text(img_path)

        assert result is None
