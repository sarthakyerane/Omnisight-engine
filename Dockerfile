FROM python:3.11-slim

WORKDIR /app

# Install Tesseract OCR binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (no pywin32 — Linux container)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pydantic-settings \
    loguru \
    SQLAlchemy \
    redis \
    Pillow \
    ImageHash \
    psutil \
    pytesseract \
    groq \
    sentence-transformers \
    chromadb \
    fastapi \
    "uvicorn[standard]" \
    pydantic

COPY . .

# Default: override with docker-compose command
CMD ["python", "-m", "ai.worker"]
