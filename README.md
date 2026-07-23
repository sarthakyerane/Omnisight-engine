# RIOM — Real-time Intelligent Observation & Memory

> **Ambient screen understanding.** RIOM silently captures your screen, extracts meaning with OCR and an LLM, and lets you recall anything you've seen using plain English.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RIOM System                                   │
│                                                                      │
│  ┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐  │
│  │  Capture Daemon │────▶│    Redis     │────▶│   AI Worker      │  │
│  │  (capture/)     │     │  frame_queue │     │  (ai/worker.py)  │  │
│  │                 │     └──────────────┘     │                  │  │
│  │  - mss          │                          │  1. Tesseract OCR│  │
│  │  - pHash dedup  │     ┌──────────────┐     │  2. OpenAI LLM   │  │
│  │  - Privacy list │────▶│   SQLite DB  │◀────│  3. ChromaDB     │  │
│  │  - Tray icon    │     │  (frames +   │     │     embedding    │  │
│  └─────────────────┘     │   analyses)  │     └──────────────────┘  │
│                          └──────────────┘                            │
│                                    │                                 │
│                          ┌─────────▼──────┐                         │
│                          │  Query API     │                         │
│                          │  (FastAPI)     │                         │
│                          │                │                         │
│                          │  GET /query    │  ← "What was I          │
│                          │  GET /frames   │     reading at 3pm?"    │
│                          └────────────────┘                         │
└──────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | |
| Tesseract | ≥ 5.0 | [Download](https://github.com/UB-Mannheim/tesseract/wiki) (Windows) |
| Redis | ≥ 7.0 | `docker run -p 6379:6379 redis` or local install |
| OpenAI API Key | — | Set in `.env` |

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/your-username/riom.git
cd riom

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

# 3. Install the package in editable mode
pip install -e ".[dev]"

# 4. Configure
cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum

# 5. Verify Tesseract is on PATH
tesseract --version
# If not, set TESSERACT_CMD in .env
```

## Running

Open three terminals (or use a process manager like Honcho / systemd):

```bash
# Terminal 1 — Capture daemon (requires Windows or adapted platform module)
riom-capture

# Terminal 2 — AI worker (processes frames from Redis queue)
riom-worker

# Terminal 3 — Query API
riom-api
# → http://127.0.0.1:8000/docs for interactive API explorer
```

## Querying Your Screen History

```bash
# Semantic search
curl "http://localhost:8000/query?q=python+code+I+was+editing"

# Filter by application
curl "http://localhost:8000/query?q=dashboard&app=chrome.exe"

# List recent frames
curl "http://localhost:8000/frames?page=1&page_size=20"
```

## Running Tests

```bash
pytest tests/ -v
```

## Privacy

- The **privacy denylist** (`DENYLIST_APPS` in `.env`) prevents capture for password managers, banking sites, and private browsing sessions.
- Captured frames are stored **locally only** — no data leaves your machine except for the OCR text sent to OpenAI.
- To keep all processing local, replace `ai/llm.py` with a call to a local model (e.g., Ollama / llama.cpp).
- Set a short `DATA_DIR` retention policy (via cron / task scheduler) to automatically delete old screenshots.

## Configuration Reference

See [`.env.example`](.env.example) for all available settings with documentation.

## Project Structure

```
riom/
├── capture/           # Screenshot capture daemon
│   ├── main.py        # CaptureDaemon orchestrator
│   ├── change_detection.py  # pHash deduplication
│   ├── tray_app.py    # System tray icon
│   └── platform_*.py  # OS-specific idle/window detection
├── storage/           # Database layer
│   ├── models.py      # SQLAlchemy ORM models (Frame, FrameAnalysis)
│   └── database.py    # Engine + session factory
├── ai/                # AI processing pipeline
│   ├── ocr.py         # Tesseract OCR
│   ├── llm.py         # OpenAI summarisation + tagging
│   ├── embeddings.py  # ChromaDB vector store
│   └── worker.py      # Redis consumer orchestrator
├── api/               # FastAPI query interface
│   ├── main.py        # App assembly + CORS
│   └── routes/
│       ├── query.py   # GET /query — semantic search
│       └── frames.py  # GET /frames — browse history
├── tests/             # pytest test suite
├── config.py          # Pydantic settings (validated)
├── pyproject.toml     # Package manifest
├── requirements.txt   # Pinned dependencies
└── .env.example       # Configuration template
```

## License

MIT
