"""Application configuration, read from environment variables (and `.env`)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

_DEV_SECRET = "dev-insecure-secret-change-me"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _groq_models() -> list[str]:
    """Primary model first, then fallbacks; duplicates removed, order kept."""
    models = [
        os.environ.get("GROQ_PRIMARY_MODEL", "openai/gpt-oss-120b"),
        os.environ.get("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b"),
    ]
    return list(dict.fromkeys(m.strip() for m in models if m and m.strip()))


class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", _DEV_SECRET)

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'instance' / 'resume_analyzer.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "5"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    ALLOWED_EXTENSIONS = frozenset({"pdf", "docx", "txt"})

    # LLM (Groq, OpenAI-compatible REST API)
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or None
    GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_MODELS = _groq_models()
    GROQ_TIMEOUT = int(os.environ.get("GROQ_TIMEOUT", "60"))

    # Local Ollama, used when every Groq model fails (or no Groq key is set)
    OLLAMA_ENABLED = _env_bool("OLLAMA_ENABLED", True)
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b")
    OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "420"))

    # Embeddings / RAG
    ENABLE_EMBEDDINGS = _env_bool("ENABLE_EMBEDDINGS", True)
    EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    RAG_WARMUP = _env_bool("RAG_WARMUP", True)
    RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "4"))
    KNOWLEDGE_FILE = str(BASE_DIR / "knowledge" / "resume_guidelines.txt")
    EMBED_CACHE_DIR = str(BASE_DIR / "instance")

    JOBS_SEED_FILE = str(BASE_DIR / "data" / "jobs.json")

    ENABLE_HISTORY = _env_bool("ENABLE_HISTORY", True)
    CSRF_ENABLED = _env_bool("CSRF_ENABLED", True)

    # Background feedback (False = write feedback inside the request, e.g. for tests)
    ASYNC_FEEDBACK = _env_bool("ASYNC_FEEDBACK", True)
    FEEDBACK_WORKERS = int(os.environ.get("FEEDBACK_WORKERS", "2"))
    # Max analyses per client per 10 minutes (0 disables the limit)
    RATE_LIMIT_PER_10_MIN = int(os.environ.get("RATE_LIMIT_PER_10_MIN", "10"))
    # Reuse an identical (same resume + same job) AI report for this many days (0 disables)
    CACHE_DAYS = int(os.environ.get("CACHE_DAYS", "7"))
    # Delete reports older than this many days at startup (0 keeps them forever)
    RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "0"))
    MIN_JOB_DESCRIPTION_CHARS = 80

    @classmethod
    def uses_dev_secret(cls, secret: str) -> bool:
        return secret == _DEV_SECRET
