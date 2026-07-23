"""
Configuration settings for RIOM (Real-time Intelligent Observation & Memory).

All values are loaded from the .env file (or environment variables).
Copy .env.example to .env and fill in required values before running.
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Capture Daemon ─────────────────────────────────────────────────────────
    IDLE_TIMEOUT_SECONDS: int = Field(default=30, gt=0)
    CAPTURE_INTERVAL_SECONDS: float = Field(default=2.0, gt=0.0)
    PHASH_THRESHOLD: int = Field(default=5, ge=0)

    # Anchor DATA_DIR to the repo root, not the working directory, so it is
    # stable when the daemon runs as a service from an arbitrary CWD.
    DATA_DIR: str = str(Path(__file__).parent / "data")

    # ── Privacy ────────────────────────────────────────────────────────────────
    DENYLIST_APPS: list[str] = [
        "1password", "bitwarden", "lastpass", "keepass", "dashlane",
        "incognito", "private browsing", "bank", "finance",
    ]

    # ── Storage ────────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./ambient_memory.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # ── AI / LLM ───────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_MAX_TOKENS: int = Field(default=256, gt=0)
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_MAX_RETRIES: int = Field(default=3, ge=1)

    # ── OCR ────────────────────────────────────────────────────────────────────
    # Leave empty to auto-detect Tesseract on PATH
    TESSERACT_CMD: str = ""

    # ── Embeddings ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── API ────────────────────────────────────────────────────────────────────
    API_HOST: str = "127.0.0.1"
    API_PORT: int = Field(default=8000, gt=0, lt=65536)
    QUERY_TOP_K: int = Field(default=10, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra fields so future .env entries don't break old code
        extra="ignore",
    )


settings = Settings()
