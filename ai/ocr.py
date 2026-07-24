"""
OCR module — extract text from screenshot images using Tesseract.

Design decisions:
  - pytesseract is a thin wrapper around the Tesseract binary, which must be
    installed separately (see README). It gives production-quality accuracy
    without requiring a GPU.
  - Images are pre-processed (grayscale + slight upscale for small text) to
    improve Tesseract accuracy on UI screenshots.
  - Empty / whitespace-only results are normalised to None so callers can
    distinguish "OCR ran but found nothing" from "OCR not yet run".
"""
from __future__ import annotations

from pathlib import Path

import pytesseract
from loguru import logger
from PIL import Image, ImageFilter

from config import settings

# Allow an explicit path override for Windows installations
if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

# Tesseract page-segmentation mode 6: assume a single uniform block of text.
# Works well for most desktop UI screenshots.
_TESSERACT_CONFIG = "--psm 6"


def _preprocess(img: Image.Image) -> Image.Image:
    """
    Convert to grayscale and sharpen slightly.

    Tesseract performs significantly better on clean, high-contrast greyscale
    images. For small fonts common in IDEs or terminals, mild sharpening helps.
    """
    grey = img.convert("L")
    # Only upscale if the image is smaller than 1280px wide (rare, but possible
    # on very low-res monitors or if a cropped region is passed in).
    if grey.width < 1280:
        scale = 1280 / grey.width
        grey = grey.resize(
            (int(grey.width * scale), int(grey.height * scale)),
            resample=Image.LANCZOS,
        )
    return grey.filter(ImageFilter.SHARPEN)


def extract_text(image_path: Path | str) -> str | None:
    """
    Run Tesseract OCR on the image at *image_path*.

    Returns:
        The extracted text string, or None if:
        - Tesseract is not installed / not found on PATH
        - The image yields no readable text
        - Any other OCR error occurs

    This function never raises — errors are logged and None is returned so
    the caller can mark the frame as failed and continue.
    """
    try:
        img = Image.open(image_path)
        preprocessed = _preprocess(img)
        raw: str = pytesseract.image_to_string(preprocessed, config=_TESSERACT_CONFIG)
        text = raw.strip()
        return text if text else None
    except pytesseract.TesseractNotFoundError:
        logger.error(
            "Tesseract not found. Install it from https://github.com/UB-Mannheim/tesseract/wiki "
            "and set TESSERACT_CMD in your .env if it is not on PATH."
        )
        return None
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as exc:
        logger.error(f"OCR failed for {image_path}: {exc}")
        return None
