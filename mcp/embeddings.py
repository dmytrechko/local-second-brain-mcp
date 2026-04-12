"""Local embeddings via sentence-transformers (all-MiniLM-L6-v2)."""

import os

_model = None  # lazy-loaded


def get_embedding(text: str) -> list[float]:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(os.getenv("LOCAL_MODEL", "all-MiniLM-L6-v2"))
    return _model.encode(text[:8000], normalize_embeddings=True).tolist()
