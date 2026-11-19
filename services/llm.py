"""
LLM clients.

* ``GroqClient``   – Groq's OpenAI-compatible API (hosted open-weight models).
* ``OllamaClient`` – a local Ollama server, used as an offline fallback.
* ``LLMChain``     – tries providers in order and returns the first answer.

Every client exposes ``available`` and ``complete(messages, ...) -> (text, model)``
and raises ``LLMError`` on failure.
"""

from __future__ import annotations

import logging
import re

import requests

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    pass


class GroqClient:
    def __init__(self, api_key: str | None, models: list[str], base_url: str, timeout: int = 60):
        self.api_key = api_key
        self.models = models
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.models)

    def complete(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 2048) -> tuple[str, str]:
        """
        Return ``(text, model)`` from the first model that answers.

        Falls through to the next model on network errors, rate limits, server
        errors and unknown-model errors. Authentication errors stop immediately.
        """
        if not self.available:
            raise LLMError("No Groq API key configured.")

        last_error: Exception | None = None
        for model in self.models:
            try:
                resp = self._session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                logger.warning("Groq request with %s failed: %s", model, exc)
                last_error = exc
                continue

            if resp.status_code in (401, 403):
                raise LLMError(f"Groq rejected the API key (HTTP {resp.status_code}).")
            if resp.status_code >= 400:
                logger.warning("Groq model %s returned HTTP %s: %.200s", model, resp.status_code, resp.text)
                last_error = LLMError(f"HTTP {resp.status_code}")
                continue

            try:
                content = resp.json()["choices"][0]["message"]["content"] or ""
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                logger.warning("Unexpected Groq response from %s: %s", model, exc)
                last_error = exc
                continue

            content = _THINK_RE.sub("", content).strip()
            if content:
                logger.info("Groq completion succeeded with %s", model)
                return content, model
            last_error = LLMError(f"{model} returned an empty response")

        raise LLMError(f"All Groq models failed: {last_error}")


class OllamaClient:
    """Client for a local Ollama server's native ``/api/chat`` endpoint."""

    def __init__(self, base_url: str, model: str, timeout: int = 180, enabled: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.enabled = enabled
        self._session = requests.Session()

    @property
    def available(self) -> bool:
        # Reachability is only known at call time; a stopped server fails fast.
        return bool(self.enabled and self.model)

    def ping(self, timeout: float = 1.5) -> bool:
        """True if the server answers and has the configured model pulled."""
        if not self.available:
            return False
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=timeout)
            names = {m.get("name", "") for m in resp.json().get("models", [])}
        except (requests.RequestException, ValueError):
            return False
        return any(n == self.model or n.split(":")[0] == self.model for n in names)

    def complete(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 2048) -> tuple[str, str]:
        if not self.available:
            raise LLMError("Ollama fallback is disabled.")
        try:
            resp = self._session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    # The advisor asks for JSON; constrain small local models to valid JSON.
                    "format": "json",
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=self.timeout,
            )
        except requests.ConnectionError as exc:
            raise LLMError(f"Ollama is not running at {self.base_url}.") from exc
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        if resp.status_code == 404:
            raise LLMError(f"Ollama model '{self.model}' not found. Run: ollama pull {self.model}")
        if resp.status_code >= 400:
            raise LLMError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            content = resp.json()["message"]["content"] or ""
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response: {exc}") from exc

        content = _THINK_RE.sub("", content).strip()
        if not content:
            raise LLMError(f"Ollama model {self.model} returned an empty response.")
        logger.info("Ollama completion succeeded with %s", self.model)
        return content, f"ollama/{self.model}"


class LLMChain:
    """Try each provider in order (e.g. Groq primary → Groq fallback → Ollama)."""

    def __init__(self, providers: list):
        self.providers = providers

    @property
    def available(self) -> bool:
        return any(p.available for p in self.providers)

    def complete(self, messages: list[dict], **kwargs) -> tuple[str, str]:
        errors = []
        for provider in self.providers:
            if not provider.available:
                continue
            try:
                return provider.complete(messages, **kwargs)
            except LLMError as exc:
                logger.warning("%s failed: %s", type(provider).__name__, exc)
                errors.append(f"{type(provider).__name__}: {exc}")
        raise LLMError("; ".join(errors) or "No LLM provider is configured.")
