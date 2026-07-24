"""
AI Pipeline for RIOM.

Exposes three sub-modules:
  - ocr        : Extract text from screenshots (Tesseract)
  - llm        : Summarise and tag screen content (OpenAI)
  - embeddings : Embed and store text in ChromaDB for semantic recall
  - worker     : Redis consumer that orchestrates the full pipeline
"""
