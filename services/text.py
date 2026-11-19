"""Text clean-up and chunking helpers shared by the parser, scorer and RAG index."""

from __future__ import annotations

import re

_SPACES_RE = re.compile(r"[ \t\f\v  -​]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _MANY_NEWLINES_RE.sub("\n\n", "\n".join(lines)).strip()


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """
    Split text into chunks of at most ``max_chars``.

    Paragraphs (blank-line separated) are kept whole when they fit; longer ones
    are split on sentence boundaries, and very long sentences at the hard limit.
    """
    chunks: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue

        current = ""
        for sentence in _SENTENCE_SPLIT_RE.split(para):
            if len(current) + len(sentence) + 1 <= max_chars:
                current = f"{current} {sentence}".strip()
                continue
            if current:
                chunks.append(current)
            if len(sentence) > max_chars:
                chunks.extend(
                    sentence[i : i + max_chars].strip()
                    for i in range(0, len(sentence), max_chars)
                )
                current = ""
            else:
                current = sentence
        if current:
            chunks.append(current)

    return [c for c in chunks if c]
