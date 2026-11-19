"""Extract plain text (and basic contact details) from uploaded resumes."""

from __future__ import annotations

import io
import logging
import re
from typing import BinaryIO

from services.text import clean_text

logger = logging.getLogger(__name__)

MIN_TEXT_CHARS = 50

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)")


class ResumeParseError(ValueError):
    """Raised with a user-facing message when a resume cannot be read."""


def file_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def extract_text(stream: BinaryIO, extension: str) -> str:
    """Return cleaned text from a PDF, DOCX or TXT stream."""
    data = stream.read()
    if not data:
        raise ResumeParseError("The uploaded file is empty.")

    ext = extension.lower().lstrip(".")
    if ext == "pdf":
        text = _extract_pdf(data)
    elif ext == "docx":
        text = _extract_docx(data)
    elif ext == "txt":
        text = _decode_text(data)
    else:
        raise ResumeParseError("Unsupported file type. Upload a PDF, DOCX or TXT file.")

    text = clean_text(text)
    if len(text) < MIN_TEXT_CHARS:
        hint = " If it is a scanned PDF, export it as a text-based PDF or DOCX." if ext == "pdf" else ""
        raise ResumeParseError("We couldn't find enough readable text in that resume." + hint)
    return text


def extract_contact(text: str) -> dict[str, str | None]:
    email = _EMAIL_RE.search(text)
    phone = _PHONE_RE.search(text)
    return {
        "email": email.group(0)[:120] if email else None,
        "phone": " ".join(phone.group(1).split())[:50] if phone else None,
    }


def _extract_pdf(data: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text

        return pdf_extract_text(io.BytesIO(data)) or ""
    except Exception as exc:  # pdfminer raises a variety of exception types
        logger.warning("PDF extraction failed: %s", exc)
        raise ResumeParseError("That PDF could not be read. It may be corrupted or password-protected.") from exc


def _extract_docx(data: bytes) -> str:
    try:
        import docx

        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        logger.warning("DOCX extraction failed: %s", exc)
        raise ResumeParseError("That DOCX file could not be read.") from exc

    parts = [p.text for p in document.paragraphs]
    # Many resume templates put content in tables; include those cells too.
    for table in document.tables:
        for row in table.rows:
            cells = list(dict.fromkeys(cell.text.strip() for cell in row.cells if cell.text.strip()))
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="ignore")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="ignore")
