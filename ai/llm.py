"""
LLM module — generate natural-language summaries and semantic tags for frames.

Uses the Groq API (llama-3.1-8b-instant by default) — ultra-fast inference,
no per-token cost concerns at this scale.

Design decisions:
  - Groq client mirrors the OpenAI SDK interface, so the code is familiar.
  - JSON mode enforces machine-parseable output every time.
  - OCR text is fenced in triple-backticks to prevent prompt injection.
  - Retry with exponential back-off handles transient rate-limit errors.
  - Short/empty OCR text skips the API call entirely to save tokens.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from groq import Groq, RateLimitError, APIError
from loguru import logger

from config import settings

_client: Groq | None = None

# Minimum OCR text length before we bother calling the LLM.
_MIN_OCR_LENGTH = 30

_SYSTEM_PROMPT = """\
You are an AI assistant that analyses screenshots of a computer screen.
You will receive OCR-extracted text from a screenshot.
Your task is to produce a concise JSON object with exactly two keys:
  "summary": A 1-3 sentence plain-English description of what the user was doing.
  "tags":    A list of 3-8 short lowercase tags (e.g. ["coding", "python", "vscode"]).

Rules:
- Be factual and specific. Do not invent information not present in the text.
- If the text is clearly garbled or unreadable, set summary to null and tags to [].
- Return ONLY valid JSON. Do not include markdown fences or explanations.
"""


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


@dataclass
class LLMResult:
    summary: str | None
    tags: list[str]


def analyse_frame(
    ocr_text: str | None,
    app_name: str,
    window_title: str,
) -> LLMResult | None:
    """
    Call the Groq LLM to summarise screen content and generate semantic tags.

    Returns None if OCR text is too short, or if all retries are exhausted.
    Never raises — errors are logged and None is returned.
    """
    if not ocr_text or len(ocr_text.strip()) < _MIN_OCR_LENGTH:
        logger.debug(
            f"Skipping LLM — OCR text too short "
            f"({len(ocr_text or '')} chars, min={_MIN_OCR_LENGTH})."
        )
        return None

    user_message = (
        f"Application: {app_name}\n"
        f"Window title: {window_title}\n\n"
        f"OCR text:\n```\n{ocr_text[:4000]}\n```"
    )

    last_exc: Exception | None = None
    for attempt in range(1, settings.LLM_MAX_RETRIES + 1):
        try:
            client = _get_client()
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            raw_json = response.choices[0].message.content or "{}"
            return _parse_response(raw_json)

        except RateLimitError as exc:
            wait = 2 ** attempt
            logger.warning(
                f"Groq rate limit (attempt {attempt}/{settings.LLM_MAX_RETRIES}). "
                f"Retrying in {wait}s…"
            )
            last_exc = exc
            time.sleep(wait)

        except APIError as exc:
            logger.error(f"Groq API error (attempt {attempt}): {exc}")
            last_exc = exc
            break

        except Exception as exc:
            logger.error(f"Unexpected LLM error (attempt {attempt}): {exc}")
            last_exc = exc
            break

    logger.error(f"LLM analysis failed after {settings.LLM_MAX_RETRIES} attempts: {last_exc}")
    return None


def _parse_response(raw: str) -> LLMResult:
    try:
        data = json.loads(raw)
        summary = data.get("summary") or None
        tags = [str(t).lower().strip() for t in data.get("tags", []) if t]
        return LLMResult(summary=summary, tags=tags)
    except json.JSONDecodeError as exc:
        logger.warning(f"LLM returned invalid JSON: {exc}. Raw: {raw[:200]!r}")
        return LLMResult(summary=None, tags=[])
