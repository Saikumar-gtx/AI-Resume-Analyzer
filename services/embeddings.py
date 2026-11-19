"""Lazily loaded SentenceTransformer model shared by scoring and RAG."""

from __future__ import annotations

import logging
import threading

import numpy as np

from services.text import chunk_text

logger = logging.getLogger(__name__)


class Embedder:
    """Thread-safe, lazily loaded sentence embedding model. Never raises."""

    def __init__(self, model_name: str, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.enabled and self._get_model() is not None

    def _get_model(self):
        if not self.enabled or self._failed:
            return None
        if self._model is None:
            with self._lock:
                if self._model is None and not self._failed:
                    try:
                        from sentence_transformers import SentenceTransformer

                        logger.info("Loading embedding model %s", self.model_name)
                        try:
                            # Use the local cache without contacting Hugging Face (much faster).
                            self._model = SentenceTransformer(self.model_name, local_files_only=True)
                        except Exception:
                            logger.info("Model not cached yet; downloading %s", self.model_name)
                            self._model = SentenceTransformer(self.model_name)
                    except Exception as exc:
                        logger.error("Embedding model unavailable (%s); semantic features disabled.", exc)
                        self._failed = True
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray | None:
        """Return L2-normalised float32 embeddings of shape (len(texts), dim)."""
        model = self._get_model()
        if model is None or not texts:
            return None
        try:
            vectors = model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(vectors, dtype=np.float32)
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            return None

    def resume_job_similarity(self, resume_text: str, job_text: str) -> float | None:
        """
        Cosine similarity between a resume and a job description.

        The model only reads ~256 tokens per input, so the resume is chunked and
        the mean of the three best-matching chunks is used.
        """
        chunks = chunk_text(resume_text, max_chars=700)[:60]
        if not chunks:
            return None
        vectors = self.encode([job_text, *chunks])
        if vectors is None:
            return None
        sims = vectors[1:] @ vectors[0]
        top = np.sort(sims)[::-1][:3]
        return float(top.mean())
