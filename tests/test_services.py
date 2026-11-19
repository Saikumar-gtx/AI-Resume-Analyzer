import io

import pytest

from services.advisor import parse_feedback, rule_based_feedback
from services.resume_parser import ResumeParseError, extract_contact, extract_text
from services.scoring import extract_keywords, match_skills, parse_skills, score_resume
from services.text import chunk_text, clean_text
from tests.conftest import SAMPLE_RESUME

# --- text -------------------------------------------------------------------

def test_clean_text_normalises_whitespace():
    assert clean_text("a  b \r\n\r\n\r\n\r\nc  ") == "a b\n\nc"


def test_chunk_text_respects_limit():
    text = "First paragraph.\n\n" + " ".join([f"Sentence number {i}." for i in range(100)])
    chunks = chunk_text(text, max_chars=120)
    assert chunks[0] == "First paragraph."
    assert all(len(c) <= 120 for c in chunks)


# --- parser -----------------------------------------------------------------

def test_extract_text_txt():
    assert "Jane Doe" in extract_text(io.BytesIO(SAMPLE_RESUME.encode()), "txt")


def test_extract_text_docx_includes_tables():
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Jane Doe - Backend engineer with Python and Flask experience.")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Skills"
    table.cell(0, 1).text = "Kubernetes, Terraform"
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    text = extract_text(buf, "docx")
    assert "Kubernetes" in text and "Flask" in text


def test_extract_text_rejects_empty_and_short():
    with pytest.raises(ResumeParseError):
        extract_text(io.BytesIO(b""), "txt")
    with pytest.raises(ResumeParseError):
        extract_text(io.BytesIO(b"too short"), "txt")


def test_extract_text_rejects_bad_pdf():
    with pytest.raises(ResumeParseError):
        extract_text(io.BytesIO(b"not really a pdf" * 10), "pdf")


def test_extract_contact():
    contact = extract_contact(SAMPLE_RESUME)
    assert contact["email"] == "jane.doe@example.com"
    assert contact["phone"] == "+1 555 123 4567"


# --- scoring ----------------------------------------------------------------

def test_parse_skills_with_aliases_and_labels():
    skills = parse_skills("Python;scikit-learn|sklearn;Cloud=AWS|GCP")
    assert [s.name for s in skills] == ["Python", "scikit-learn", "Cloud"]
    assert skills[2].aliases == ("AWS", "GCP")


@pytest.mark.parametrize(
    "text, skill, expected",
    [
        ("I write JavaScript daily", "Java", False),
        ("Strong C++ and Rust", "C", False),
        ("Embedded C, C++", "C", True),
        ("Used Node.js and React", "Node.js", True),
        ("CI/CD with Jenkins", "CI/CD", True),
        ("used sklearn for models", "scikit-learn|sklearn", True),
        ("Power-BI dashboards", "Power BI", True),
        ("Built apps on LLMs", "LLM", True),
        ("fine-tuned transformer models", "Transformers", True),
        ("Microservices and APIs", "Microservices", True),
        ("Kubernetes", "Kubernetes", True),
    ],
)
def test_match_skills_boundaries(text, skill, expected):
    matched, _ = match_skills(text, parse_skills(skill))
    assert bool(matched) is expected


def test_extract_keywords_skips_stop_words():
    keywords = extract_keywords("Strong experience with Kubernetes and Kubernetes clusters and Terraform.")
    assert keywords[0] == "kubernetes"
    assert "strong" not in keywords and "experience" not in keywords


def test_score_resume_rewards_relevant_resume():
    desc = "Build REST APIs in Python with Flask, PostgreSQL, Redis, Docker and Kubernetes on AWS."
    skills = "Python;Flask;PostgreSQL;Redis;Docker;Kubernetes;AWS;Kafka"
    good = score_resume(SAMPLE_RESUME, desc, skills)
    bad = score_resume("Pastry chef with ten years of croissant and baguette experience. " * 3, desc, skills)
    assert good.overall > 70 > bad.overall
    assert "Kafka" in good.missing_skills
    assert good.semantic_score is None


def test_score_resume_uses_semantic_component():
    breakdown = score_resume(SAMPLE_RESUME, "Python developer", "Python", semantic_similarity=0.75)
    assert breakdown.semantic_score == 100


# --- advisor ----------------------------------------------------------------

def test_parse_feedback_json():
    raw = 'Sure!\n{"summary": "Good fit.", "strengths": ["A"], "gaps": [], "recommendations": ["- Do X", "2. Do Y"]}'
    fb = parse_feedback(raw)
    assert fb.summary == "Good fit."
    assert fb.recommendations == ["Do X", "Do Y"]


def test_parse_feedback_text_sections():
    raw = "Summary:\nSolid backend profile.\n\nRecommendations:\n• First\n• Second\n1) Third"
    fb = parse_feedback(raw)
    assert fb.summary == "Solid backend profile."
    assert fb.recommendations == ["First", "Second", "Third"]


def test_parse_feedback_rejects_garbage():
    assert parse_feedback("I cannot help with that.") is None


def test_rule_based_feedback_mentions_missing_skills():
    breakdown = score_resume(SAMPLE_RESUME, "Kafka streaming", "Python;Kafka")
    fb = rule_based_feedback(SAMPLE_RESUME, "Backend Engineer", breakdown)
    assert fb.source == "rules"
    assert 1 <= len(fb.recommendations) <= 5
    assert any("Kafka" in r for r in fb.recommendations)
