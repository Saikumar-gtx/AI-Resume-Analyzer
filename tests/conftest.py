import io
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from models import db  # noqa: E402

SAMPLE_RESUME = """Jane Doe
jane.doe@example.com | +1 555 123 4567 | github.com/janedoe

Summary
Backend engineer with 5 years of experience building Python services.

Experience
Senior Software Engineer, Acme Corp (2021 - present)
- Built REST APIs in Flask and FastAPI serving 2M requests per day.
- Migrated services to Docker and Kubernetes, cutting deploy time by 60%.
- Optimized PostgreSQL queries, reducing p95 latency by 45%.
- Led a team of 4 engineers and introduced pytest-based CI/CD with GitHub Actions.

Projects
- Built a Redis-backed rate limiter used by 10k users.

Skills
Python, Flask, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS, Git

Education
B.Sc. Computer Science, State University, 2019
"""


@pytest.fixture()
def app(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
            "ENABLE_EMBEDDINGS": False,
            "RAG_WARMUP": False,
            "GROQ_API_KEY": None,
            "OLLAMA_ENABLED": False,
            "ASYNC_FEEDBACK": False,
            "RATE_LIMIT_PER_10_MIN": 0,
            "EMBED_CACHE_DIR": str(tmp_path),
        }
    )
    # Never call real LLMs from tests.
    assert not app.extensions["analyzer"]["llm"].available
    yield app
    with app.app_context():
        db.session.remove()
        db.engine.dispose()


@pytest.fixture()
def client(app):
    return app.test_client()


def csrf_token(client) -> str:
    html = client.get("/analyze").get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def upload(text: str = SAMPLE_RESUME, name: str = "resume.txt"):
    return (io.BytesIO(text.encode("utf-8")), name)
