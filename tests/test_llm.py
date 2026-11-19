import json

import pytest
import requests

from services.advisor import generate_feedback
from services.llm import GroqClient, LLMChain, LLMError, OllamaClient
from services.scoring import score_resume
from tests.conftest import SAMPLE_RESUME

GOOD_JSON = json.dumps(
    {
        "summary": "Strong backend profile.",
        "strengths": ["Python APIs"],
        "gaps": ["No Kafka"],
        "recommendations": ["Add Kafka", "Quantify impact", "Add projects", "Group skills", "Add summary"],
    }
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeProvider:
    def __init__(self, name, result=None, error=None, available=True):
        self.name, self.result, self.error, self._available = name, result, error, available
        self.calls = 0

    @property
    def available(self):
        return self._available

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.error:
            raise LLMError(self.error)
        return self.result, self.name


# --- chain ------------------------------------------------------------------

def test_chain_falls_back_to_next_provider():
    groq = FakeProvider("groq", error="HTTP 503")
    ollama = FakeProvider("ollama/mistral:7b", result="hi")
    assert LLMChain([groq, ollama]).complete([]) == ("hi", "ollama/mistral:7b")
    assert groq.calls == 1 and ollama.calls == 1


def test_chain_stops_at_first_success():
    groq = FakeProvider("groq", result="from groq")
    ollama = FakeProvider("ollama", result="from ollama")
    assert LLMChain([groq, ollama]).complete([])[0] == "from groq"
    assert ollama.calls == 0


def test_chain_skips_unavailable_and_reports_all_errors():
    chain = LLMChain([FakeProvider("groq", available=False), FakeProvider("ollama", error="not running")])
    with pytest.raises(LLMError, match="not running"):
        chain.complete([])
    assert not LLMChain([FakeProvider("x", available=False)]).available


def test_groq_auth_error_still_falls_back_to_ollama(monkeypatch):
    groq = GroqClient("bad-key", ["m1", "m2"], "https://groq.invalid")
    monkeypatch.setattr(groq._session, "post", lambda *a, **k: FakeResponse(401, {"error": "bad key"}))
    ollama = FakeProvider("ollama/mistral:7b", result=GOOD_JSON)
    assert LLMChain([groq, ollama]).complete([])[1] == "ollama/mistral:7b"


# --- Ollama client ----------------------------------------------------------

def test_ollama_success_requests_json(monkeypatch):
    client = OllamaClient("http://localhost:11434/", "mistral:7b")
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(url=url, body=json)
        return FakeResponse(200, {"message": {"content": GOOD_JSON}})

    monkeypatch.setattr(client._session, "post", fake_post)
    text, model = client.complete([{"role": "user", "content": "hi"}], max_tokens=100)
    assert model == "ollama/mistral:7b" and "Strong backend" in text
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["body"]["format"] == "json" and seen["body"]["stream"] is False
    assert seen["body"]["options"]["num_predict"] == 100


def test_ollama_not_running(monkeypatch):
    client = OllamaClient("http://localhost:11434", "mistral:7b")

    def refuse(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(client._session, "post", refuse)
    with pytest.raises(LLMError, match="not running"):
        client.complete([])


def test_ollama_missing_model(monkeypatch):
    client = OllamaClient("http://localhost:11434", "nope")
    monkeypatch.setattr(client._session, "post", lambda *a, **k: FakeResponse(404, {"error": "model not found"}))
    with pytest.raises(LLMError, match="ollama pull nope"):
        client.complete([])


def test_ollama_disabled():
    client = OllamaClient("http://localhost:11434", "mistral:7b", enabled=False)
    assert not client.available and not client.ping()


def test_ollama_ping_matches_tag(monkeypatch):
    client = OllamaClient("http://localhost:11434", "mistral:7b")
    monkeypatch.setattr(
        client._session, "get", lambda *a, **k: FakeResponse(200, {"models": [{"name": "mistral:7b"}]})
    )
    assert client.ping()


# --- advisor integration ----------------------------------------------------

def test_feedback_comes_from_ollama_when_groq_fails():
    breakdown = score_resume(SAMPLE_RESUME, "Python Kafka APIs", "Python;Kafka")
    chain = LLMChain(
        [FakeProvider("openai/gpt-oss-120b", error="HTTP 500"), FakeProvider("ollama/mistral:7b", result=GOOD_JSON)]
    )
    fb = generate_feedback(SAMPLE_RESUME, "Backend Developer", "Python Kafka APIs", breakdown, None, chain)
    assert fb.source == "ai" and fb.model == "ollama/mistral:7b"
    assert len(fb.recommendations) == 5


def test_feedback_uses_rules_when_everything_fails():
    breakdown = score_resume(SAMPLE_RESUME, "Python Kafka APIs", "Python;Kafka")
    chain = LLMChain([FakeProvider("groq", error="down"), FakeProvider("ollama", error="not running")])
    fb = generate_feedback(SAMPLE_RESUME, "Backend Developer", "Python Kafka APIs", breakdown, None, chain)
    assert fb.source == "rules"
