"""
Resume-to-job scoring.

The overall score blends up to three signals:

* **Skill coverage** – share of the job's curated skills found in the resume. A skill
  used in experience/projects counts fully; one that only appears in the skills
  list counts ``LISTED_ONLY_CREDIT``.
* **Keyword coverage** – share of the job description's most frequent terms found.
* **Semantic similarity** – embedding similarity between resume and job description
  (optional; skipped when the embedding model is unavailable).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from services.sections import evidence_text, split_sections

WEIGHTS = {"skills": 0.5, "keywords": 0.2, "semantic": 0.3}

LISTED_ONLY_CREDIT = 0.6

# Cosine similarities from all-MiniLM-L6-v2 for resume/job pairs: unrelated pairs
# sit near 0.05–0.15, decent matches around 0.45–0.6 and near-identical text
# above 0.75. Map 0.15–0.75 onto 0–1 so good and great resumes stay distinguishable.
SEMANTIC_FLOOR = 0.15
SEMANTIC_CEILING = 0.75

STOP_WORDS = frozenset(
    """
    a about above across after all also an and any are as at be been being both but by
    can could do does during each either etc for from has have having how if in including
    into is it its may more most must not of on one or other our over per plus such than
    that the their them then there these they this those through to too under up upon us
    using via was we well were what when where which while who will with within without
    would you your
    ability able across based best build building closely collaborate collaborating
    communication comfortable deep demonstrated design designing develop developing
    development drive ensure environment environments excellent experience experienced
    familiar familiarity good great hands-on high highly ideal including knowledge
    looking maintain manage new nice preferred proficiency proficient proven related
    relevant required responsible role skills solid strong success successful team teams
    understanding work working world year years
    actionable apply applies backed behind builds care cares clear closely complex
    contribute contributes creates defines delivers effective efficient end extract
    focus grasp help helps large modern operates operate plus practices quality robust
    runs ship ships sound support supports through uses wide works write writes
    high-quality high-traffic well-documented well-tested
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9+#]*(?:[.\-/][a-z0-9+#]+)*")


@dataclass(frozen=True)
class Skill:
    name: str
    aliases: tuple[str, ...]


@dataclass
class ScoreBreakdown:
    overall: int
    skill_score: int | None
    keyword_score: int
    semantic_score: int | None
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    # Subset of matched_skills found only in the skills list, not in experience/projects.
    listed_only_skills: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)

    @property
    def band(self) -> str:
        return score_band(self.overall)

    def to_dict(self) -> dict:
        return asdict(self)


def score_band(score: float) -> str:
    if score >= 75:
        return "strong"
    if score >= 50:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def parse_skills(raw: str | None) -> list[Skill]:
    """
    Parse a skills string into Skill objects.

    Format: ``;`` separates skills and ``|`` separates aliases. The first alias is
    the display name unless an explicit one is given with ``=``:
    ``"Python;scikit-learn|sklearn;Cloud platform=AWS|GCP|Azure"``.
    """
    skills: list[Skill] = []
    for entry in (raw or "").split(";"):
        label, _, alias_part = entry.rpartition("=")
        aliases = tuple(a.strip() for a in alias_part.split("|") if a.strip())
        if aliases:
            skills.append(Skill(name=label.strip() or aliases[0], aliases=aliases))
    return skills


@lru_cache(maxsize=2048)
def _alias_pattern(alias: str) -> re.Pattern[str]:
    base, plural = alias.lower(), ""
    if len(base) > 2 and base[-1].isalpha():
        # Accept singular and plural: "LLM" ~ "LLMs", "Transformers" ~ "transformer".
        if len(base) > 3 and base.endswith("s") and not base.endswith("ss"):
            base = base[:-1]
        plural = "(?:e?s)?"
    escaped = re.escape(base).replace(r"\ ", r"[\s\-]+")
    # Don't let "c" match inside "c++"/"c#", or "java" inside "javascript".
    return re.compile(rf"(?<![a-z0-9]){escaped}{plural}(?![a-z0-9+#])")


def match_skills(text: str, skills: list[Skill]) -> tuple[list[str], list[str]]:
    lowered = text.lower()
    matched, missing = [], []
    for skill in skills:
        if any(_alias_pattern(a).search(lowered) for a in skill.aliases):
            matched.append(skill.name)
        else:
            missing.append(skill.name)
    return matched, missing


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return [t.strip(".-/") for t in _TOKEN_RE.findall(text.lower())]


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def extract_keywords(text: str, limit: int = 15) -> list[str]:
    """Most frequent meaningful terms of ``text`` (ties keep first-seen order)."""
    tokens = [t for t in tokenize(text) if len(t) > 2 and t not in STOP_WORDS]
    counts: Counter[str] = Counter()
    surface: dict[str, str] = {}  # stem -> first-seen word, for display
    for token in tokens:
        stem = _stem(token)
        counts[stem] += 1
        surface.setdefault(stem, token)
    return [surface[stem] for stem, _ in counts.most_common(limit)]


def match_keywords(resume_text: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    resume_stems = {_stem(t) for t in tokenize(resume_text)}
    matched = [k for k in keywords if _stem(k) in resume_stems]
    missing = [k for k in keywords if _stem(k) not in resume_stems]
    return matched, missing


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------

def semantic_to_unit(similarity: float) -> float:
    unit = (similarity - SEMANTIC_FLOOR) / (SEMANTIC_CEILING - SEMANTIC_FLOOR)
    return max(0.0, min(1.0, unit))


def score_resume(
    resume_text: str,
    job_description: str,
    job_skills: str | None,
    semantic_similarity: float | None = None,
) -> ScoreBreakdown:
    components: dict[str, float] = {}

    skills = parse_skills(job_skills)
    matched_skills, missing_skills = match_skills(resume_text, skills)
    listed_only: list[str] = []
    sections = split_sections(resume_text)
    if sections and "skills" in sections:
        demonstrated, _ = match_skills(evidence_text(sections), skills)
        listed_only = [s for s in matched_skills if s not in demonstrated]
    if skills:
        credit = len(matched_skills) - len(listed_only) + LISTED_ONLY_CREDIT * len(listed_only)
        components["skills"] = credit / len(skills)

    keywords = extract_keywords(job_description)
    matched_kw, missing_kw = match_keywords(resume_text, keywords)
    components["keywords"] = len(matched_kw) / len(keywords) if keywords else 0.0

    if semantic_similarity is not None:
        components["semantic"] = semantic_to_unit(semantic_similarity)

    total_weight = sum(WEIGHTS[k] for k in components)
    overall = sum(WEIGHTS[k] * v for k, v in components.items()) / total_weight

    def pct(key: str) -> int | None:
        return round(components[key] * 100) if key in components else None

    return ScoreBreakdown(
        overall=round(overall * 100),
        skill_score=pct("skills"),
        keyword_score=pct("keywords") or 0,
        semantic_score=pct("semantic"),
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        listed_only_skills=listed_only,
        matched_keywords=matched_kw,
        missing_keywords=missing_kw,
    )


# ---------------------------------------------------------------------------
# Skills for a pasted job description
# ---------------------------------------------------------------------------

def skills_from_text(text: str, vocabulary: list[str], limit: int = 15) -> str:
    """
    Pick the catalogue skills mentioned in ``text`` (e.g. a pasted job posting)
    and return them in skills-string format, ordered by first mention.
    """
    lowered = text.lower()
    found: dict[str, tuple[int, str]] = {}
    seen_aliases: set[str] = set()
    candidates: list[Skill] = []
    for entry in vocabulary:
        if "=" in entry:
            # Umbrella entries ("Web framework=Flask|FastAPI|Django") are too vague as a
            # single skill, so treat each named tool as a skill of its own.
            candidates.extend(Skill(a, (a,)) for a in parse_skills(entry)[0].aliases if len(a) > 2)
        else:
            candidates.extend(parse_skills(entry))
    for skill in candidates:
        hits = [(m.start(), a) for a in skill.aliases if (m := _alias_pattern(a).search(lowered))]
        if not hits:
            continue
        if any(a == skill.aliases[0] for _, a in hits):
            # The canonical name is in the posting, so later aliases are synonyms
            # ("Kubernetes|k8s") and the resume may use any of them.
            name, aliases = skill.aliases[0], skill.aliases
        else:
            # Only an alternative matched ("Jenkins|GitHub Actions" with "GitHub Actions"
            # in the text): require exactly that tool, under its own name.
            _, alias = min(hits)
            if alias.islower() and alias.replace(" ", "").isalpha():
                continue  # descriptive words ("experiments", "monitoring") aren't tools
            name, aliases = alias, (alias,)
        lowered_aliases = {a.lower() for a in aliases}
        if name in found or lowered_aliases & seen_aliases:
            continue
        found[name] = (min(pos for pos, _ in hits), "|".join(aliases))
        seen_aliases |= lowered_aliases
    ordered = sorted(found.values())[:limit]
    return ";".join(raw for _, raw in ordered)
