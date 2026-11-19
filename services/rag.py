"""Retrieval over a small resume-writing knowledge base (FAISS, with a NumPy fallback)."""

from __future__ import annotations

import hashlib
import logging
import os
import threading

import numpy as np

from services.embeddings import Embedder
from services.text import chunk_text

logger = logging.getLogger(__name__)

BUILTIN_DOCUMENTS = [
    "Resumes should emphasize quantifiable achievements and impact.",
    "Use strong action verbs such as developed, implemented, optimized, and designed.",
    "Tailor each resume to the specific job description for better ATS scores.",
    "Include both technical and soft skills relevant to the role.",
    "Keep formatting simple and clean for ATS compatibility.",
    "For technical roles, highlight projects with measurable outcomes.",
]


def load_documents(path: str) -> list[str]:
    """Chunks of the knowledge file, or the built-in tips if it is missing/empty."""
    try:
        with open(path, encoding="utf-8") as fh:
            chunks = chunk_text(fh.read())
        if chunks:
            logger.info("Loaded %d knowledge chunks from %s", len(chunks), path)
            return chunks
        logger.warning("%s is empty; using built-in knowledge.", path)
    except FileNotFoundError:
        logger.info("No knowledge file at %s; using built-in knowledge.", path)
    except OSError as exc:
        logger.warning("Could not read %s (%s); using built-in knowledge.", path, exc)
    return list(BUILTIN_DOCUMENTS)


class KnowledgeBase:
    """Lazily built vector index. All public methods are safe to call and never raise."""

    def __init__(self, embedder: Embedder, knowledge_path: str, cache_dir: str):
        self.embedder = embedder
        self.documents = load_documents(knowledge_path)
        self.cache_dir = cache_dir
        self._vectors: np.ndarray | None = None
        self._faiss_index = None
        self._ready = False
        self._lock = threading.Lock()

    @property
    def cache_path(self) -> str:
        # Fingerprint the model and the documents so edits invalidate the cache.
        digest = hashlib.sha256(
            "\x1f".join([self.embedder.model_name, *self.documents]).encode("utf-8")
        ).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"rag_embeddings_{digest}.npy")

    def ensure_ready(self) -> bool:
        if self._ready:
            return True
        with self._lock:
            if not self._ready:
                self._build()
        return self._ready

    def _build(self) -> None:
        vectors = self._load_cache()
        if vectors is None:
            vectors = self.embedder.encode(self.documents)
            if vectors is None:
                return
            self._save_cache(vectors)

        try:
            import faiss

            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self._faiss_index = index
            logger.info("FAISS index ready: %d vectors, dim=%d", index.ntotal, vectors.shape[1])
        except Exception as exc:
            logger.info("FAISS unavailable (%s); using NumPy search.", exc)
        self._vectors = vectors
        self._ready = True

    def _load_cache(self) -> np.ndarray | None:
        path = self.cache_path
        if not os.path.exists(path):
            return None
        try:
            arr = np.load(path, allow_pickle=False).astype(np.float32)
        except Exception as exc:
            logger.warning("Ignoring unreadable embedding cache %s: %s", path, exc)
            return None
        if arr.ndim != 2 or arr.shape[0] != len(self.documents) or arr.shape[1] == 0:
            logger.warning("Ignoring embedding cache with unexpected shape %s", arr.shape)
            return None
        return arr

    def _save_cache(self, vectors: np.ndarray) -> None:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(self.cache_path, vectors)
        except OSError as exc:
            logger.warning("Could not write embedding cache: %s", exc)

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        if not self.ensure_ready() or self._vectors is None:
            return []
        q = self.embedder.encode([query])
        if q is None or q.shape[1] != self._vectors.shape[1]:
            return []
        k = min(top_k, len(self.documents))
        try:
            if self._faiss_index is not None:
                _, idx = self._faiss_index.search(q, k)
                order = [i for i in idx[0] if 0 <= i < len(self.documents)]
            else:
                order = list(np.argsort(self._vectors @ q[0])[::-1][:k])
        except Exception as exc:
            logger.error("Knowledge retrieval failed: %s", exc)
            return []
        return [self.documents[i] for i in order]
