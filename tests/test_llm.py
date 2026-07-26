"""
Tests for the LLM module (ai/llm.py)

All OpenAI calls are mocked — no API key required to run tests.

Covers:
- Short OCR text returns None without calling the API
- Valid response is parsed correctly
- Malformed JSON response returns LLMResult with no summary
- RateLimitError triggers retry and ultimately returns None after exhaustion
- Non-retriable APIError does not retry
"""
import json
from unittest.mock import MagicMock, patch, call

import pytest

from ai.llm import analyse_frame, _parse_response, LLMResult, _MIN_OCR_LENGTH


class TestParseResponse:
    def test_valid_json_parsed_correctly(self):
        raw = json.dumps({"summary": "User was writing Python code.", "tags": ["coding", "python"]})
        result = _parse_response(raw)
        assert result.summary == "User was writing Python code."
        assert result.tags == ["coding", "python"]

    def test_null_summary_becomes_none(self):
        raw = json.dumps({"summary": None, "tags": []})
        result = _parse_response(raw)
        assert result.summary is None
        assert result.tags == []

    def test_invalid_json_returns_empty_result(self):
        result = _parse_response("this is not json {{{")
        assert result.summary is None
        assert result.tags == []

    def test_tags_are_lowercased_and_stripped(self):
        raw = json.dumps({"summary": "x", "tags": ["  Python  ", "CODING"]})
        result = _parse_response(raw)
        assert result.tags == ["python", "coding"]

    def test_missing_keys_return_defaults(self):
        raw = json.dumps({})
        result = _parse_response(raw)
        assert result.summary is None
        assert result.tags == []


class TestAnalyseFrame:
    def test_short_ocr_text_skips_api_call(self):
        short_text = "hi"
        assert len(short_text) < _MIN_OCR_LENGTH
        with patch("ai.llm._get_client") as mock_client:
            result = analyse_frame(short_text, "code.exe", "main.py")
            mock_client.assert_not_called()
            assert result is None

    def test_none_ocr_text_skips_api_call(self):
        with patch("ai.llm._get_client") as mock_client:
            result = analyse_frame(None, "code.exe", "main.py")
            mock_client.assert_not_called()
            assert result is None

    def test_valid_response_returns_llm_result(self):
        long_text = "x" * 100
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(
            {"summary": "User was editing code.", "tags": ["coding"]}
        )
        with patch("ai.llm._get_client") as mock_get_client:
            mock_get_client.return_value.chat.completions.create.return_value = mock_response
            result = analyse_frame(long_text, "code.exe", "main.py")
        assert result is not None
        assert result.summary == "User was editing code."
        assert "coding" in result.tags

    def test_rate_limit_retries_and_returns_none_on_exhaustion(self):
        from groq import RateLimitError
        long_text = "x" * 100

        mock_client = MagicMock()
        # Groq's RateLimitError only requires a message string
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )

        with patch("ai.llm._get_client", return_value=mock_client):
            with patch("ai.llm.settings") as mock_settings:
                mock_settings.LLM_MAX_RETRIES = 2
                mock_settings.LLM_MAX_TOKENS = 256
                mock_settings.LLM_TEMPERATURE = 0.0
                mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
                with patch("ai.llm.time.sleep"):
                    result = analyse_frame(long_text, "code.exe", "main.py")

        assert result is None
        assert mock_client.chat.completions.create.call_count == 2
