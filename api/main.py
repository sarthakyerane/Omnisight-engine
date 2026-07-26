"""
RIOM FastAPI application — assembles all routers and middleware.

Running locally:
    uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

Production:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import query, frames
from config import settings

app = FastAPI(
    title="RIOM — Ambient Screen Memory",
    description=(
        "Query your screen history using natural language. "
        "The AI pipeline processes captured screenshots with OCR and an LLM, "
        "then makes them searchable via semantic embeddings."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — lock down to localhost only in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://{settings.API_HOST}:{settings.API_PORT}", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(frames.router)


@app.get("/health", tags=["meta"], summary="Health check")
def health() -> dict:
    """Returns 200 OK when the API is running."""
    return {"status": "ok", "version": "0.1.0"}


def main() -> None:
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
