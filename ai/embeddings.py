"""
Embeddings module — store and query frame summaries in ChromaDB.

Design decisions:
  - sentence-transformers (all-MiniLM-L6-v2) runs locally with no API cost,
    no data leaving the machine, and ~50ms inference per frame.
  - ChromaDB is an embedded vector database — zero infrastructure required.
    Data persists to CHROMA_PERSIST_DIR on disk.
  - Each document stored is the LLM-generated summary + tags, NOT the raw OCR
    text. Summaries are more semantically coherent and produce better recall.
  - The ChromaDB document ID is the Frame database ID so results can be joined
    back to the SQL database for full metadata retrieval.
"""
from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger

from config import settings

_COLLECTION_NAME = "screen_frames"

_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )
        _collection = _chroma_client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{_COLLECTION_NAME}' loaded "
            f"({_collection.count()} documents)."
        )
    return _collection


def index_frame(
    frame_id: int,
    summary: str,
    tags: list[str],
    app_name: str,
    ts_iso: str,
) -> str:
    """
    Embed and store a frame summary in ChromaDB.

    The document text is a concatenation of the summary and tags to give the
    embedding model the richest possible signal.

    Returns:
        The ChromaDB document ID (str form of frame_id).

    Raises:
        Exception: propagated from ChromaDB so the caller can mark the frame failed.
    """
    doc_id = str(frame_id)
    document = f"{summary}\nTags: {', '.join(tags)}"

    collection = _get_collection()
    collection.upsert(
        ids=[doc_id],
        documents=[document],
        metadatas=[{
            "frame_id": frame_id,
            "app_name": app_name,
            "ts": ts_iso,
            "tags": ", ".join(tags),
        }],
    )
    logger.debug(f"Frame {frame_id} indexed in ChromaDB (doc_id={doc_id}).")
    return doc_id


def query(
    query_text: str,
    top_k: int | None = None,
    app_filter: str | None = None,
) -> list[dict]:
    """
    Perform a semantic similarity search over all indexed frames.

    Args:
        query_text:  Natural-language query (e.g. "Python code I was editing").
        top_k:       Max results to return (defaults to settings.QUERY_TOP_K).
        app_filter:  If set, filter results to only this app name.

    Returns:
        List of result dicts, each containing:
          { "frame_id", "distance", "summary", "app_name", "ts", "tags" }
        Ordered by ascending distance (most similar first).
    """
    k = top_k or settings.QUERY_TOP_K
    collection = _get_collection()

    if collection.count() == 0:
        return []

    where_filter = {"app_name": app_filter} if app_filter else None

    results = collection.query(
        query_texts=[query_text],
        n_results=min(k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    output: list[dict] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
        output.append({
            "frame_id": meta.get("frame_id"),
            "distance": round(dist, 4),
            "summary": doc,
            "app_name": meta.get("app_name", ""),
            "ts": meta.get("ts", ""),
            "tags": meta.get("tags", ""),
        })

    return output
