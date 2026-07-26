"""
RIOM Query API — FastAPI application.

Provides endpoints to:
  - Query captured frames semantically  GET /query
  - List recent frames                  GET /frames
  - Get a single frame + analysis       GET /frames/{frame_id}

Running:
    uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
    # or
    riom-api
"""
