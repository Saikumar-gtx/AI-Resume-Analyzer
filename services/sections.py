"""Split resume text into its usual sections (experience, projects, skills, ...)."""

from __future__ import annotations

import re

# Canonical section -> heading phrases that introduce it.
_HEADINGS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "executive summary", "professional summary", "profile", "objective", "about me"),
    "experience": (
        "experience", "work experience", "professional experience", "employment", "employment history",
        "work history", "internships", "internship", "relevant experience",
    ),
    "projects": ("projects", "personal projects", "academic projects", "key projects", "selected projects"),
    "skills": (
        "skills", "technical skills", "skills & technologies", "skills and technologies", "technologies",
        "tech stack", "core competencies", "tools", "skills & tools",
    ),
    "education": ("education", "academic background", "qualifications"),
    "certifications": ("certifications", "certificates", "licenses"),
    "achievements": ("achievements", "awards", "honors", "honours", "accomplishments"),
    "research": ("research", "publications", "research experience"),
    "activities": ("activities", "leadership", "volunteering", "extracurricular activities"),
}

_LOOKUP = {phrase: section for section, phrases in _HEADINGS.items() for phrase in phrases}
_HEADING_RE = re.compile(r"^[\W\d_]*([a-z][a-z &/]+?)\s*[:\-–—]?\s*$")

# Sections where using a skill counts as evidence (vs. just listing it).
EVIDENCE_SECTIONS = frozenset({"summary", "experience", "projects", "research", "achievements", "activities", "other"})


def _heading(line: str) -> str | None:
    if len(line) > 40:
        return None
    m = _HEADING_RE.match(line.strip().lower())
    return _LOOKUP.get(m.group(1).strip()) if m else None


def split_sections(text: str) -> dict[str, str]:
    """
    Return ``{section: text}``. Text before the first recognised heading goes to
    ``"other"``. An empty dict means no headings were found at all.
    """
    sections: dict[str, list[str]] = {}
    current = "other"
    found_heading = False
    for line in text.splitlines():
        section = _heading(line)
        if section:
            current, found_heading = section, True
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    if not found_heading:
        return {}
    return {name: "\n".join(lines).strip() for name, lines in sections.items() if "\n".join(lines).strip()}


def evidence_text(sections: dict[str, str]) -> str:
    """Text from the sections where skills are *used* rather than listed."""
    return "\n".join(body for name, body in sections.items() if name in EVIDENCE_SECTIONS)
